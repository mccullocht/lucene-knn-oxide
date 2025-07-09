//! FFI interfaces for implementing lucene HNSW codecs.
//!
//! With this implementation it is necessary to parse the field metadata once on each side of the
//! FFI boundary since this implementation does not interact with IndexInput.

#![feature(stdarch_aarch64_prefetch)]

pub mod direct_monotonic_reader;
pub mod vint;

use std::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};
use std::collections::BinaryHeap;
use std::iter::FusedIterator;

use ahash::AHashSet;
use bytes::Buf;
use min_max_heap::MinMaxHeap;
use simsimd::SpatialSimilarity;

use crate::direct_monotonic_reader::DirectMonotonicReader;
use crate::vint::VIntBuf;

#[allow(dead_code)]
pub struct FieldHnswIndex {
    /// Raw encoded index bytes for this field.
    data: &'static [u8],

    /// Encoding of the raw vector data.
    encoding: VectorEncoding,
    /// Similarity function used to compute this index.
    similarity: VectorSimilarity,

    /// Dimensionality of the vector.
    dimensions: usize,
    /// Number of vectors in the index.
    len: usize,
    /// Maximum number of edges on each vertex in the index.
    /// This number is doubled for the lowest level of the index.
    max_edges: usize,
    /// Information about levels 1+.
    upper_levels: Vec<GraphUpperLevel>,
    /// Locate the starting offset in data for any (level, ord) pair.
    offsets_decoder: DirectMonotonicReader<'static>,
}

impl FieldHnswIndex {
    // XXX consider accepting vector data as well and putting all the upper level vectors in heap
    // memory per-level. this is ~7% memory increase but < 1% if only for level > 1. Only a few
    // percent of scoring happens here so it may not be worth it, even if this improves caching
    // and reduces memory latency.
    // XXX consider recording stats in this struct and printing them on drop.
    fn new(mut field_meta: &'static [u8], index_data: &'static [u8]) -> Option<Self> {
        let encoding: VectorEncoding = field_meta
            .get_i32_le()
            .try_into()
            .expect("valid vector encoding");
        if !encoding.is_supported() {
            return None;
        }

        let similarity: VectorSimilarity = field_meta
            .get_i32_le()
            .try_into()
            .expect("valid vector similarity");
        if !similarity.is_supported() {
            return None;
        }

        let index_data_off: usize = field_meta
            .get_vi64()
            .expect("index_data_off")
            .try_into()
            .expect("non-negative offset");
        let index_data_len: usize = field_meta
            .get_vi64()
            .expect("index_data_len")
            .try_into()
            .expect("non-negative length");
        let data = &index_data[index_data_off..(index_data_off + index_data_len)];

        let dimensions = field_meta
            .get_vi32()
            .expect("dimensions")
            .try_into()
            .unwrap();
        let len = field_meta.get_i32_le().try_into().unwrap();
        let max_edges = field_meta
            .get_vi32()
            .expect("max_edges")
            .try_into()
            .unwrap();

        // l0 has no representation in the meta stream but all subsequent levels do.
        let levels: usize = field_meta.get_vi32().expect("levels").try_into().unwrap();
        let mut upper_levels = Vec::with_capacity(levels - 1);
        let mut initial_offset = len;
        for _ in 1..levels {
            let ords_len = field_meta
                .get_vi32()
                .expect("level nodes_len")
                .try_into()
                .unwrap();
            let mut ords = Vec::with_capacity(ords_len);
            let mut sum = 0u32;
            for _ in 0..ords_len {
                sum = sum
                    .checked_add(
                        field_meta
                            .get_vi32()
                            .expect("node delta")
                            .try_into()
                            .expect("positive ord"),
                    )
                    .expect("node len overflow");
                ords.push(sum);
            }
            upper_levels.push(GraphUpperLevel {
                initial_offset,
                ords,
            });
            initial_offset += ords_len;
        }
        let offsets_decoder = DirectMonotonicReader::new(initial_offset, field_meta, index_data);

        Some(Self {
            data,
            encoding,
            similarity,
            dimensions,
            len,
            max_edges,
            upper_levels,
            offsets_decoder,
        })
    }

    /// Return the ordinal of the entry point to the graph.
    ///
    /// The entry point is always present at the highest level.
    pub fn entry_point(&self) -> u32 {
        if let Some(top_level) = self.upper_levels.last() {
            top_level.ords[0]
        } else {
            0
        }
    }

    /// Obtain an iterator over the edges of `ord` at `level`.
    ///
    /// *Panics* if `ord` is not present in `level`.
    pub fn edge_iter(&self, level: usize, ord: u32) -> HnswVertexEdgeIter<'_> {
        let offset_ord = if level == 0 {
            ord as usize
        } else {
            self.upper_levels[level - 1].get_offset(ord)
        };
        let it =
            HnswVertexEdgeIter::new(&self.data[self.offsets_decoder.get(offset_ord) as usize..]);
        assert!(
            it.len() <= self.max_edges * 2,
            "vertex {}:{} has too many edges ({})",
            level,
            ord,
            it.len()
        );
        it
    }
}

/// Representation of level 1+.
struct GraphUpperLevel {
    /// First offset for this level in the offset decoder.
    initial_offset: usize,
    /// Sorted list of ordinals in the level. This can be use to map back to an offset.
    ords: Vec<u32>,
}

impl GraphUpperLevel {
    /// Get the offset for ord.
    ///
    /// *Panics* if ord is not present in this level.
    fn get_offset(&self, ord: u32) -> usize {
        self.initial_offset + self.ords.binary_search(&ord).expect("ord exists in level")
    }
}

/// Iterator over edges on a vertex. See [FieldHnswIndex::edge_iter].
pub struct HnswVertexEdgeIter<'a> {
    edges: usize,
    last: u32,
    buf: &'a [u8],
}

impl<'a> HnswVertexEdgeIter<'a> {
    #[inline(always)]
    fn new(mut buf: &'a [u8]) -> Self {
        let edges = buf.get_vi32().expect("vertex edge len").try_into().unwrap();
        Self {
            edges,
            last: 0,
            buf,
        }
    }
}

impl Iterator for HnswVertexEdgeIter<'_> {
    type Item = u32;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.edges == 0 {
            return None;
        }

        self.last = self
            .last
            .checked_add(
                self.buf
                    .get_vi32()
                    .expect("verted edge next")
                    .try_into()
                    .expect("valid u32"),
            )
            .expect("no overflow");
        self.edges -= 1;
        Some(self.last)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.edges, Some(self.edges))
    }
}

impl ExactSizeIterator for HnswVertexEdgeIter<'_> {}

impl FusedIterator for HnswVertexEdgeIter<'_> {}

#[allow(dead_code)]
pub struct FieldVectorData {
    encoding: VectorEncoding,
    similarity: VectorSimilarity,
    dimensions: usize,
    len: usize,
    vector_data: &'static [f32],
}

impl FieldVectorData {
    /// Lucene writes f32 vector data in little-endian order and ensures it is aligned.
    /// We require little-endian order and cast the data to an f32 slice.
    /// API accommodations are required if we allow big-endian or unaligned input.
    #[cfg(not(target_endian = "little"))]
    const MUST_BE_LITTLE_ENDIAN: usize = 0usize - 1;

    pub fn new(mut field_meta: &'static [u8], vector_data: &'static [u8]) -> Option<Self> {
        let encoding: VectorEncoding = field_meta
            .get_i32_le()
            .try_into()
            .expect("valid vector encoding");
        if !encoding.is_supported() {
            eprintln!("Encoding {:?} is not supported", encoding);
            return None;
        }

        let similarity: VectorSimilarity = field_meta
            .get_i32_le()
            .try_into()
            .expect("valid vector similarity");
        if !similarity.is_supported() {
            eprintln!("Similarity {:?} is not supported", similarity);
            return None;
        }

        let data_off: usize = field_meta
            .get_vi64()
            .expect("vector_data_offset")
            .try_into()
            .unwrap();
        let data_len: usize = field_meta
            .get_vi64()
            .expect("vector_data_length")
            .try_into()
            .unwrap();

        let dimensions = field_meta
            .get_vi32()
            .expect("dimensions")
            .try_into()
            .unwrap();
        let len = field_meta.get_u32_le().try_into().unwrap();

        let stride = dimensions * encoding.byte_size();
        assert_eq!(
            data_len,
            len * stride,
            "Encoded vector data size not the same as vector size * len. encoding {:?} dimensions {} stride {} len {}",
            encoding,
            dimensions,
            stride,
            len
        );

        let docs_with_field_offset = field_meta.get_i64_le();
        assert!(
            docs_with_field_offset < 0,
            "Sparse vector representation is not supported"
        );

        let vector_data =
            bytemuck::try_cast_slice::<_, f32>(&vector_data[data_off..(data_off + data_len)])
                .expect("vector data is f32 aligned");

        // TODO: read direct monotonic ord-to-doc map and IndexedDISI.
        // These actually contain the _same_ information so I'm not sure why we have them both.
        Some(Self {
            encoding,
            similarity,
            dimensions,
            len,
            vector_data,
        })
    }

    #[inline(always)]
    pub fn get(&self, index: usize) -> &[f32] {
        assert!(index < self.len, "index={} len={}", index, self.len);
        let start = index * self.dimensions;
        &self.vector_data[start..(start + self.dimensions)]
    }

    #[inline(always)]
    pub fn score_ord(&self, query: &[f32], ord: u32) -> Neighbor {
        Neighbor {
            vertex: ord,
            score: self.similarity.score(query, self.get(ord as usize)),
        }
    }

    #[inline(always)]
    pub fn score_many(&self, query: &[f32], ords: &[u32], scores: &mut [f32]) {
        let mut ord_chunks = ords.chunks_exact(8);
        let mut score_chunks = scores.chunks_exact_mut(8);
        for (ord_chunk, score_chunk) in ord_chunks.by_ref().zip(score_chunks.by_ref()) {
            let docs = [
                self.get(ord_chunk[0] as usize),
                self.get(ord_chunk[1] as usize),
                self.get(ord_chunk[2] as usize),
                self.get(ord_chunk[3] as usize),
                self.get(ord_chunk[4] as usize),
                self.get(ord_chunk[5] as usize),
                self.get(ord_chunk[6] as usize),
                self.get(ord_chunk[7] as usize),
            ];
            score_chunk.copy_from_slice(&self.similarity.score_many::<8, 128>(query, docs));
        }
        for (ord, score) in ord_chunks
            .remainder()
            .iter()
            .zip(score_chunks.into_remainder().iter_mut())
        {
            *score = self.similarity.score(query, self.get(*ord as usize))
        }
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum VectorEncoding {
    Float,
    Byte,
}

impl VectorEncoding {
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Float)
    }

    pub fn byte_size(&self) -> usize {
        match self {
            Self::Float => 4,
            Self::Byte => 1,
        }
    }
}

impl TryFrom<i32> for VectorEncoding {
    type Error = i32;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Byte),
            1 => Ok(Self::Float),
            _ => Err(value),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum VectorSimilarity {
    Euclidean,
    DotProduct,
    Cosine,
    MaximumInnerProduct,
}

impl VectorSimilarity {
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Euclidean | Self::DotProduct)
    }

    pub fn score(&self, q: &[f32], d: &[f32]) -> f32 {
        assert_eq!(q.len(), d.len());
        match self {
            Self::Euclidean => 1.0f32 / (1.0f32 + SpatialSimilarity::l2sq(q, d).unwrap() as f32),
            Self::DotProduct => {
                // I tried a manual aarch64 SIMD implementation where I unrolled the loop (16d)
                // and it was not any faster, maybe actually slower.
                // XXX 0.0f32.max((1.0f32 + SpatialSimilarity::dot(q, d).unwrap() as f32) / 2.0f32)
                0.0f32.max((1.0f32 + Self::dot(q, d)) / 2.0f32)
            }
            Self::Cosine => unimplemented!(),
            Self::MaximumInnerProduct => unimplemented!(),
        }
    }

    pub fn score_many<const N: usize, const P: usize>(
        &self,
        q: &[f32],
        docs: [&[f32]; N],
    ) -> [f32; N] {
        match self {
            Self::DotProduct => {
                let mut scores = Self::dot_many::<N, P>(q, docs);
                for n in 0..N {
                    scores[n] = 0.0f32.max(1.0f32 + scores[n]) / 2.0f32;
                }
                scores
            }
            _ => unimplemented!(),
        }
    }

    fn dot(q: &[f32], d: &[f32]) -> f32 {
        unsafe {
            let mut dot = vdupq_n_f32(0.0);
            let mut pp = d.as_ptr().add(128);
            let pp_end = d.as_ptr().add(d.len());
            for (qc, dc) in q.chunks(4).zip(d.chunks(4)) {
                if pp < pp_end {
                    core::arch::aarch64::_prefetch::<0, 3>(pp as *const i8);
                    pp = pp.add(16);
                }
                let qc = vld1q_f32(qc.as_ptr());
                let dc = vld1q_f32(dc.as_ptr());
                dot = vfmaq_f32(dot, qc, dc);
            }
            vaddvq_f32(dot)
        }
    }

    fn dot_many<const N: usize, const P: usize>(q: &[f32], docs: [&[f32]; N]) -> [f32; N] {
        unsafe {
            for offset in (0..P).step_by(16) {
                for n in 0..N {
                    core::arch::aarch64::_prefetch::<0, 3>(
                        docs[n].as_ptr().add(offset) as *const i8
                    );
                }
            }
            let mut dot = [vdupq_n_f32(0.0); N];
            for offset in (0..q.len()).step_by(4) {
                let prefetch_offset = P + offset;
                if prefetch_offset % 16 == 0 && prefetch_offset < q.len() {
                    for n in 0..N {
                        core::arch::aarch64::_prefetch::<0, 3>(
                            docs[n].as_ptr().add(prefetch_offset) as *const i8,
                        );
                    }
                }

                let qv = vld1q_f32(q.as_ptr().add(offset));
                for n in 0..N {
                    dot[n] = vfmaq_f32(dot[n], qv, vld1q_f32(docs[n].as_ptr().add(offset)));
                }
            }

            let mut scores = [0.0f32; N];
            for n in 0..N {
                scores[n] = vaddvq_f32(dot[n]);
            }
            scores
        }
    }
}

impl TryFrom<i32> for VectorSimilarity {
    type Error = i32;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Euclidean),
            1 => Ok(Self::DotProduct),
            2 => Ok(Self::Cosine),
            3 => Ok(Self::MaximumInnerProduct),
            _ => Err(value),
        }
    }
}

// NB: this is shared across the FFI boundary.
#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct Neighbor {
    vertex: u32,
    score: f32,
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.vertex == other.vertex && self.score.total_cmp(&other.score).is_eq()
    }
}

impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Neighbor {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .total_cmp(&other.score)
            .reverse()
            .then(self.vertex.cmp(&other.vertex))
    }
}

pub struct NeighborQueue {
    queue: BinaryHeap<Neighbor>,
    limit: usize,
}

impl NeighborQueue {
    pub fn with_limit(limit: usize) -> Self {
        assert_ne!(limit, 0);
        Self {
            queue: BinaryHeap::with_capacity(limit),
            limit,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, neighbor: Neighbor) {
        if self.queue.len() < self.limit {
            self.queue.push(neighbor);
        } else {
            let mut top = self.queue.peek_mut().expect("queue not empty");
            if neighbor < *top {
                *top = neighbor;
            }
        }
    }

    #[inline(always)]
    pub fn min_similarity(&self) -> f32 {
        self.queue.peek().map(|n| n.score).unwrap_or(f32::MIN)
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn into_sorted_iter(self) -> impl Iterator<Item = Neighbor> {
        self.queue.into_sorted_vec().into_iter()
    }
}

/// A read-only version of FixedBitSet
#[repr(transparent)]
struct FixedBitSet<'a>(&'a [u64]);

impl<'a> FixedBitSet<'a> {
    pub fn new(bits: &'a [u64]) -> Self {
        Self(bits)
    }

    pub fn get(&self, index: usize) -> bool {
        (self.0[index / 64] & (1u64 << (index % 64))) != 0
    }

    pub fn iter(&self) -> FixedBitSetIter<'_> {
        FixedBitSetIter::new(self.0)
    }

    pub fn cardinality(&self) -> usize {
        self.0.iter().map(|w| w.count_ones()).sum::<u32>() as usize
    }
}

pub struct FixedBitSetIter<'a> {
    bits: &'a [u64],
    base: usize,
    word: u64,
}

impl<'a> FixedBitSetIter<'a> {
    fn new(bits: &'a [u64]) -> Self {
        let (word, bits) = bits.split_first().unwrap_or((&0, &[]));
        Self {
            bits,
            base: 0,
            word: *word,
        }
    }
}

impl<'a> Iterator for FixedBitSetIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        while self.word == 0 {
            // Get the next word. Keep advancing so long as a word is available and it is zero.
            if let Some((word, bits)) = self.bits.split_first() {
                self.word = *word;
                self.bits = bits;
                self.base += 64;
            } else {
                return None;
            }
        }

        // Extract the least significant bits and remove it from word to advance iteration.
        let bitidx = self.word.trailing_zeros();
        self.word ^= 1u64 << bitidx;
        Some(self.base + bitidx as usize)
    }
}

impl FusedIterator for FixedBitSetIter<'_> {}

/// Search for `query` in this field and return up to `results.len()` neighbors.
/// If `accept_ords` is specified, contains a bitmap of LSB u64s indicating if each ordinal is accepted.
/// Returns the number of valid results added to neighbors.
#[inline(never)]
fn search(
    index: &FieldHnswIndex,
    vector_data: &FieldVectorData,
    query: &[f32],
    accept_ords: Option<FixedBitSet<'_>>,
    results: &mut [Neighbor],
) -> usize {
    let mut neighbor_queue = NeighborQueue::with_limit(results.len());
    let accepted_count = accept_ords
        .as_ref()
        .map(|b| b.cardinality())
        .unwrap_or(index.len);
    let exhaustive_search =
        ((index.len as f64).ln() * results.len() as f64) as usize >= accepted_count;
    if exhaustive_search {
        if let Some(filter_ords) = accept_ords.as_ref() {
            search_exhaustively(vector_data, query, filter_ords.iter(), &mut neighbor_queue);
        } else {
            search_exhaustively(vector_data, query, 0..vector_data.len, &mut neighbor_queue);
        }
    } else {
        search_index(
            index,
            vector_data,
            query,
            accept_ords.as_ref(),
            &mut neighbor_queue,
        );
    }

    let result_len = neighbor_queue.len().min(results.len());
    for (i, o) in neighbor_queue.into_sorted_iter().zip(results.iter_mut()) {
        *o = i;
    }
    result_len
}

#[inline(never)]
fn search_exhaustively(
    vector_data: &FieldVectorData,
    query: &[f32],
    ords: impl Iterator<Item = usize>,
    queue: &mut NeighborQueue,
) {
    for i in ords {
        queue.push(vector_data.score_ord(query, i as u32))
    }
}

#[inline(never)]
fn search_index(
    index: &FieldHnswIndex,
    vector_data: &FieldVectorData,
    query: &[f32],
    accept_ords: Option<&FixedBitSet<'_>>,
    queue: &mut NeighborQueue,
) {
    let best_entry_point = find_best_entry_point(index, vector_data, query);
    let mut visited = AHashSet::with_capacity(10_000);
    let mut candidates = MinMaxHeap::with_capacity(queue.len() + 1);
    candidates.push(best_entry_point);
    visited.insert(best_entry_point.vertex);
    if accept_ords
        .map(|s| s.get(best_entry_point.vertex as usize))
        .unwrap_or(true)
    {
        queue.push(best_entry_point);
    }
    let mut ords = Vec::with_capacity(index.max_edges * 2);
    let mut scores = Vec::with_capacity(index.max_edges * 2);
    while let Some(candidate) = candidates.pop_min() {
        // If the best candidate is worse than the worst result, break.
        if candidate.score < queue.min_similarity() {
            break;
        }

        for vertex in index.edge_iter(0, candidate.vertex) {
            if !visited.insert(vertex) {
                continue;
            }

            ords.push(vertex);
        }

        scores.resize(ords.len(), 0.0);
        vector_data.score_many(query, &ords, &mut scores);
        for (vertex, score) in ords.drain(..).zip(scores.drain(..)) {
            let n = Neighbor { vertex, score };
            // This differs from lucene in that we limit the length of the queue.
            if candidates.len() < queue.len() {
                candidates.push(n);
            } else {
                candidates.push_pop_max(n);
            }
            if accept_ords.map(|s| s.get(vertex as usize)).unwrap_or(true) {
                queue.push(n);
            }
        }
    }
}

#[inline(never)]
fn find_best_entry_point(
    index: &FieldHnswIndex,
    vector_data: &FieldVectorData,
    query: &[f32],
) -> Neighbor {
    let mut best_entry_point = vector_data.score_ord(query, index.entry_point());
    // NB: don't perform a visited check like lucene does here. It does prevent us from re-scoring
    // vectors but otherwise has very little upside.
    for level in (1..(index.upper_levels.len() + 1)).rev() {
        let mut found_better = true;
        while found_better {
            found_better = false;
            for vertex in index.edge_iter(level, best_entry_point.vertex) {
                let n = vector_data.score_ord(query, vertex);
                if n < best_entry_point {
                    best_entry_point = n;
                    found_better = true;
                }
            }
        }
    }
    best_entry_point
}

/// FFI call to initialize a new FieldHnswIndex.
///
/// # Safety
/// * All pointers must be valid and match their corresponding length.
/// * All passed pointers must live _at least_ as long as the returned object.
/// * The returned pointer is owned by the caller. To free resources owned the returned pointer
///   [lucene_knn_oxide_close_field_hnsw_index] must be called.
#[unsafe(no_mangle)]
pub extern "C" fn lucene_knn_oxide_new_field_hnsw_index(
    field_index_meta_ptr: *const u8,
    field_index_meta_len: usize,
    index_data_ptr: *const u8,
    index_data_len: usize,
) -> *mut FieldHnswIndex {
    // NB: we make the lifetime static here to avoid managing the lifetime of an object passed
    // across an FFI bound. This is _mostly fine_ since the lifetime has been externalized to the
    // caller but we have to be careful not to let the lifetime escape field.
    let field_index_meta: &'static [u8] =
        unsafe { std::slice::from_raw_parts(field_index_meta_ptr, field_index_meta_len) };
    let index_data: &'static [u8] =
        unsafe { std::slice::from_raw_parts(index_data_ptr, index_data_len) };

    let field = FieldHnswIndex::new(field_index_meta, index_data).map(Box::new);
    field.map(Box::into_raw).unwrap_or(std::ptr::null_mut())
}

/// FFI call to close an FieldHnswIndex and free related resources.
///
/// # Safety
/// * The passed pointer must have been created using [lucene_knn_oxide_new_field_hnsw_index]
#[unsafe(no_mangle)]
pub extern "C" fn lucene_knn_oxide_close_field_hnsw_index(field: *mut FieldHnswIndex) {
    if !field.is_null() {
        let _ = unsafe { Box::from_raw(field) };
    }
}

/// FFI call to initialize a new FieldVectorData.
///
/// # Safety
/// * All pointers must be valid and match their corresponding length.
/// * All passed pointers must live _at least_ as long as the returned object.
/// * The returned pointer is owned by the caller. To free resources owned the returned pointer
///   [lucene_knn_oxide_close_field_vector_data] must be called.
#[unsafe(no_mangle)]
pub extern "C" fn lucene_knn_oxide_new_field_vector_data(
    field_vector_meta_ptr: *const u8,
    field_vector_meta_len: usize,
    vector_data_ptr: *const u8,
    vector_data_len: usize,
) -> *mut FieldVectorData {
    // NB: we make the lifetime static here to avoid managing the lifetime of an object passed
    // across an FFI bound. This is _mostly fine_ since the lifetime has been externalized to the
    // caller but we have to be careful not to let the lifetime escape field.
    let field_vector_meta: &'static [u8] =
        unsafe { std::slice::from_raw_parts(field_vector_meta_ptr, field_vector_meta_len) };
    let vector_data: &'static [u8] =
        unsafe { std::slice::from_raw_parts(vector_data_ptr, vector_data_len) };

    let field = FieldVectorData::new(field_vector_meta, vector_data).map(Box::new);
    field.map(Box::into_raw).unwrap_or(std::ptr::null_mut())
}

/// FFI call to close an FieldVectorData and free related resources.
///
/// # Safety
/// * The passed pointer must have been created using [lucene_knn_oxide_new_field_vector_data]
#[unsafe(no_mangle)]
pub extern "C" fn lucene_knn_oxide_close_field_vector_data(field: *mut FieldVectorData) {
    if !field.is_null() {
        let _ = unsafe { Box::from_raw(field) };
    }
}

/// FFI call to search an hnsw field.
///
/// Accepts a query, an optional bitmap of accepted ordinals, and a neighbor list.
/// Returns the number of neighbors written.
///
/// # Safety
/// * `index` may not be null.
/// * `vector_data` may not be null.
/// * `query_ptr` may not be null and must be appropriately aligned.
/// * `accept_ords_ptr` must be appropriately aligned if it is not null.
/// * `neighbor_ptr` may not be null and must be appropriately aligned.
/// * All lengths must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lucene_knn_oxide_search_hnsw_field(
    index: *mut FieldHnswIndex,
    vector_data: *mut FieldVectorData,
    query_ptr: *const f32,
    query_len: usize,
    accept_ords_ptr: *const u64,
    accept_ords_len: usize,
    neighbor_ptr: *mut Neighbor,
    neighbor_len: usize,
) -> usize {
    let index = unsafe { index.as_ref().expect("index is not null") };
    let vector_data = unsafe { vector_data.as_ref().expect("vector_data is not null") };
    let query = unsafe { std::slice::from_raw_parts(query_ptr, query_len) };
    let accept_ords = if accept_ords_ptr != std::ptr::null() {
        Some(FixedBitSet::new(unsafe {
            std::slice::from_raw_parts(accept_ords_ptr, accept_ords_len)
        }))
    } else {
        None
    };
    let results = unsafe { std::slice::from_raw_parts_mut(neighbor_ptr, neighbor_len) };
    search(index, vector_data, query, accept_ords, results)
}
