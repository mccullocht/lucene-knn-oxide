// https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/util/packed/DirectMonotonicReader.java
// https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/util/packed/DirectReader.java

#[derive(Copy, Clone, Default)]
struct BlockMeta {
    min: u64,
    avg: f32,
    bpv: u8,
    offset: u64,
}

impl BlockMeta {
    const CODED_SIZE: usize = std::mem::size_of::<u64>()
        + std::mem::size_of::<f32>()
        + std::mem::size_of::<u8>()
        + std::mem::size_of::<u64>();

    // NB: this may read up to 3 bytes beyond the end of the value.
    fn get(&self, data: &[u8], index: usize) -> u64 {
        match self.bpv {
            1 => ((data[self.offset as usize + (index / 8)] >> (index & 7)) & 0x1).into(),
            2 => (data[self.offset as usize + (index / 4)] as u64 >> ((index & 3) * 2)) & 0x3,
            4 => (data[self.offset as usize + (index / 2)] as u64 >> ((index & 1) * 4)) & 0xf,
            8 => data[self.offset as usize + index] as u64 & 0xff,
            12 => self.get_bit_sized::<12>(data, index),
            16 => self.get_bit_sized::<16>(data, index),
            20 => self.get_bit_sized::<20>(data, index),
            24 => self.get_bit_sized::<24>(data, index),
            28 => self.get_bit_sized::<28>(data, index),
            32 => self.get_bit_sized::<32>(data, index),
            40 => self.get_bit_sized::<40>(data, index),
            48 => self.get_bit_sized::<48>(data, index),
            56 => self.get_bit_sized::<56>(data, index),
            64 => self.get_bit_sized::<64>(data, index),
            _ => unreachable!(),
        }
        // DirectReader takes input, bpv, and offset at construction.
        // takes block_index at get.
    }

    fn get_bit_sized<const N: usize>(&self, data: &[u8], index: usize) -> u64 {
        let mut buf = [0u8; 8];
        let start = self.offset as usize + ((index * N) / 8);
        let len = N.div_ceil(8);
        buf[..len].copy_from_slice(&data[start..(start + len)]);
        let shift = match N {
            1 => index & 7,
            2 => index & 3,
            _ if N % 4 == 0 => index & 1,
            8 => 0,
            _ => unreachable!(),
        };
        let value = u64::from_le_bytes(buf) >> shift;
        if N == 64 {
            value
        } else {
            value & ((1 << N) - 1)
        }
    }
}

impl From<&[u8]> for BlockMeta {
    fn from(value: &[u8]) -> Self {
        let (value, min) = value.split_at(std::mem::size_of::<u64>());
        let (value, avg) = value.split_at(std::mem::size_of::<f32>());
        let (value, bpv) = value.split_at(std::mem::size_of::<u8>());
        let (_, offset) = value.split_at(std::mem::size_of::<u64>());
        Self {
            min: u64::from_le_bytes(min.try_into().unwrap()),
            avg: f32::from_le_bytes(avg.try_into().unwrap()),
            bpv: u8::from_le_bytes(bpv.try_into().unwrap()),
            offset: u64::from_le_bytes(offset.try_into().unwrap()),
        }
    }
}

pub struct Meta {
    block_shift: usize,
    block_mask: usize,
    blocks: Box<[BlockMeta]>,
}

impl Meta {
    pub fn new(values_len: usize, mut block_shift: usize, input: &[u8]) -> Self {
        let block_len = (values_len + ((1 << block_shift) - 1)) >> block_shift;
        let mut blocks = Vec::with_capacity(block_len);
        let mut all_zeros = true;
        for c in input.chunks(BlockMeta::CODED_SIZE).take(block_len) {
            let b = BlockMeta::from(c);
            all_zeros = all_zeros && b.min == 0 && b.avg == 0.0 && b.bpv == 0;
            blocks.push(b);
        }

        if all_zeros {
            block_shift = 63;
            blocks = vec![BlockMeta::default(); 1];
        }

        Self {
            block_shift,
            block_mask: (1 << block_shift) - 1,
            blocks: blocks.into_boxed_slice(),
        }
    }
}

pub struct DirectMonotonicReader<'a> {
    meta: Meta,
    data: &'a [u8],
}

impl<'a> DirectMonotonicReader<'a> {
    pub fn new(meta: Meta, data: &'a [u8]) -> Self {
        Self { meta, data }
    }

    pub fn get(&self, index: usize) -> u64 {
        let block = &self.meta.blocks[index >> self.meta.block_shift];
        let block_index = index & self.meta.block_mask;
        let delta: u64 = block.get(self.data, block_index);
        block.min + (block.avg * block_index as f32) as u64 + delta
    }
}
