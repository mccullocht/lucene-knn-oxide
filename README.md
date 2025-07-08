# LuceneKnnOxide

A proof-of-concept grade implementation of `Lucene99HnswVectorsFormat` with a native read path.

The write path is just a copy of the Lucene99 implementation, on the read path we use a native
extension to search each segment provided that the backing input source is actually an mmap based
`MemorySegment`. Only a subset of functionality is supported: the field must be a float field, the
vectors must be exactly 1-1 assigned to docids, and not all similarity functions are supported.

On a corpus of 1M 1536d float vectors merged down to a single segment with dot product similarity
on an M4 Mac this is about 10% faster than the pure Java implementation. It is likely that a 
significant chunk of this is simply being able to score vectors without copying them off of the
heap.
