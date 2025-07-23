package com.github.mccullocht.nvector;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.ByteOrder;
import java.util.Optional;

import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MemorySegmentAccessInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;

class NativeVectorAccess {
    private static final MethodHandle NEW_DATA_FIELD_METHOD;
    private static final MethodHandle CLOSE_DATA_FIELD_METHOD;
    private static final MethodHandle NEW_INDEX_FIELD_METHOD;
    private static final MethodHandle CLOSE_INDEX_FIELD_METHOD;

    private static final MethodHandle SEARCH_INDEX_METHOD;
    private static final MethodHandle BULK_SCORE_METHOD;

    static {
        try {
            SymbolLookup nativeLib = loadNativeLib();
            Linker linker = Linker.nativeLinker();

            MemorySegment newFieldAddr = nativeLib.find("lucene_knn_oxide_new_field_vector_data").orElseThrow();
            FunctionDescriptor newFielDesc = FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                    ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG);
            NEW_DATA_FIELD_METHOD = linker.downcallHandle(newFieldAddr, newFielDesc);

            MemorySegment closeFieldAddr = nativeLib.find("lucene_knn_oxide_close_field_vector_data").orElseThrow();
            FunctionDescriptor closeFielDesc = FunctionDescriptor.ofVoid(ValueLayout.ADDRESS);
            CLOSE_DATA_FIELD_METHOD = linker.downcallHandle(closeFieldAddr, closeFielDesc);

            newFieldAddr = nativeLib.find("lucene_knn_oxide_new_field_hnsw_index").orElseThrow();
            newFielDesc = FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                    ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG);
            NEW_INDEX_FIELD_METHOD = linker.downcallHandle(newFieldAddr, newFielDesc);

            closeFieldAddr = nativeLib.find("lucene_knn_oxide_close_field_hnsw_index").orElseThrow();
            closeFielDesc = FunctionDescriptor.ofVoid(ValueLayout.ADDRESS);
            CLOSE_INDEX_FIELD_METHOD = linker.downcallHandle(closeFieldAddr, closeFielDesc);

            MemorySegment searchIndexAddr = nativeLib.find("lucene_knn_oxide_search_hnsw_field").orElseThrow();
            FunctionDescriptor searchIndesDesc = FunctionDescriptor.of(
                    ValueLayout.JAVA_LONG, // returns count of returned results
                    ValueLayout.ADDRESS, // index ptr
                    ValueLayout.ADDRESS, // vector data ptr
                    ValueLayout.ADDRESS, // query ptr
                    ValueLayout.JAVA_LONG, // query len
                    ValueLayout.ADDRESS, // ords bitmap ptr
                    ValueLayout.JAVA_LONG, // ords bitmap len
                    ValueLayout.ADDRESS, // results ptr
                    ValueLayout.JAVA_LONG // results len
            );
            SEARCH_INDEX_METHOD = linker.downcallHandle(searchIndexAddr, searchIndesDesc);

            MemorySegment bulkScoreAddr = nativeLib.find("lucene_knn_oxide_bulk_score").orElseThrow();
            FunctionDescriptor bulkScoreDesc = FunctionDescriptor.ofVoid(
                    ValueLayout.ADDRESS, // vector data ptr
                    ValueLayout.ADDRESS, // query ptr
                    ValueLayout.JAVA_LONG, // query len
                    ValueLayout.ADDRESS, // neighbors ptr
                    ValueLayout.JAVA_LONG // neighbors len
            );
            BULK_SCORE_METHOD = linker.downcallHandle(bulkScoreAddr, bulkScoreDesc);
        } catch (IOException e) {
            throw new UnsatisfiedLinkError("Could not link native vector access lib.");
        }
    }

    private static SymbolLookup loadNativeLib() throws IOException {
        String libJarPath = "/META-INF/natives/darwin/aarch64/liblucene_knn_oxide.dylib";
        try (InputStream in = NativeVectorAccess.class.getResourceAsStream(libJarPath)) {
            File libFile = File.createTempFile("liblucene_knn_oxide.dylib", "");
            libFile.deleteOnExit();

            // i wish you'd give me some sugar, this is just ridiculous.
            try (OutputStream out = new FileOutputStream(libFile)) {
                byte[] buf = new byte[4096];
                int read;
                while ((read = in.read(buf)) != -1) {
                    out.write(buf, 0, read);
                }
            }

            return SymbolLookup.libraryLookup(libFile.getAbsolutePath(), Arena.global());
        }
    }

    public static class IndexField {
        public static Optional<IndexField> create(IndexInput fieldMeta, IndexInput indexData, Arena arena)
                throws IOException {
            return newField(NEW_INDEX_FIELD_METHOD, CLOSE_INDEX_FIELD_METHOD, fieldMeta, indexData, arena)
                    .map(IndexField::new);
        }

        private IndexField(MemorySegment ptr) {
            this.ptr = ptr;
        }

        private MemorySegment ptr;
    }

    public static class VectorDataField {
        public static Optional<VectorDataField> create(IndexInput fieldMeta, IndexInput vectorData, Arena arena)
                throws IOException {
            return newField(NEW_DATA_FIELD_METHOD, CLOSE_DATA_FIELD_METHOD, fieldMeta, vectorData, arena)
                    .map(VectorDataField::new);
        }

        private VectorDataField(MemorySegment ptr) {
            this.ptr = ptr;
        }

        private MemorySegment ptr;
    }

    private static final ValueLayout.OfInt ORD_LAYOUT = ValueLayout.JAVA_INT.withName("ord")
            .withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfFloat SCORE_LAYOUT = ValueLayout.JAVA_FLOAT.withName("score")
            .withOrder(ByteOrder.nativeOrder());
    private static final MemoryLayout NEIGHBOR_LAYOUT = MemoryLayout.structLayout(
            ORD_LAYOUT,
            SCORE_LAYOUT);

    public static void search(IndexField index, VectorDataField vectors, float[] query, KnnCollector collector,
            Bits acceptDocs) {
        try (Arena searchArena = Arena.ofConfined()) {
            var querySegment = vectorAsMemorySegment(query, searchArena);

            // XXX we are assuming 1:1 doc:ord which is fine for now but wrong in the long
            // run. it's also offensive to have to re-serialize this.
            FixedBitSet acceptOrds = switch (acceptDocs) {
                case null -> null;
                case FixedBitSet set -> set;
                case Bits bits -> FixedBitSet.copyOf(bits);
            };
            MemorySegment acceptOrdsSegment = null;
            long acceptOrdsLen = 0;
            if (acceptOrds != null) {
                acceptOrdsSegment = searchArena.allocateFrom(ValueLayout.JAVA_LONG, acceptOrds.getBits());
                acceptOrdsLen = acceptOrds.getBits().length;
            } else {
                acceptOrdsSegment = MemorySegment.ofAddress(0);
            }

            var resultsLayout = MemoryLayout.sequenceLayout(collector.k(), NEIGHBOR_LAYOUT);
            var resultsSegment = searchArena.allocate(resultsLayout);
            try {
                long numResults = (long) SEARCH_INDEX_METHOD.invokeExact(index.ptr, vectors.ptr, querySegment,
                        (long) query.length, acceptOrdsSegment, acceptOrdsLen, resultsSegment, (long) collector.k());
                // TODO: there has to be a nicer way than hard coding the offsets.
                resultsSegment.elements(NEIGHBOR_LAYOUT).limit(numResults).forEach(n -> {
                    collector.collect(n.get(ORD_LAYOUT, 0), n.get(SCORE_LAYOUT, 4));
                });
            } catch (Throwable t) {
                throw new RuntimeException("unreachable! (search)", t);
            }
        }
    }

    public static MemorySegment vectorAsMemorySegment(float[] vector, Arena arena) {
        return arena.allocateFrom(ValueLayout.JAVA_FLOAT, vector);
    }

    public static void scoreOrds(VectorDataField vectors, MemorySegment query, int[] ords, float[] scores,
            int numOrds) {
        try (var arena = Arena.ofConfined()) {
            var ordsSegment = arena.allocateFrom(ValueLayout.JAVA_INT, ords);
            var scoresSegment = arena.allocate(ValueLayout.JAVA_FLOAT, numOrds);
            try {
                BULK_SCORE_METHOD.invokeExact(vectors.ptr, query, query.byteSize() / ValueLayout.JAVA_FLOAT.byteSize(),
                        ordsSegment, scoresSegment, (long) numOrds);
            } catch (Throwable t) {
                throw new RuntimeException("unreachable! (scoreOrds)", t);
            }
            for (int i = 0; i < numOrds; i++) {
                scores[i] = scoresSegment.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            }
        }
    }

    private static Optional<MemorySegment> newField(MethodHandle newFieldMethod, MethodHandle closeFieldMethod,
            IndexInput fieldMeta, IndexInput data, Arena arena) throws IOException {
        var metaSegment = segmentFromInput(fieldMeta, fieldMeta.getFilePointer(),
                fieldMeta.length() - fieldMeta.getFilePointer());
        if (metaSegment.isEmpty()) {
            return Optional.empty();
        }
        var dataSegment = segmentFromInput(data, 0, data.length());
        if (dataSegment.isEmpty()) {
            return Optional.empty();
        }
        try {
            var ptr = (MemorySegment) newFieldMethod.invokeExact(metaSegment.get(), metaSegment.get().byteSize(),
                    dataSegment.get(),
                    dataSegment.get().byteSize());
            if (ptr.address() == 0) {
                return Optional.empty();
            }
            ptr.reinterpret(arena, s -> {
                try {
                    closeFieldMethod.invokeExact(s);
                } catch (Throwable t) {
                    throw new RuntimeException("unreachable! (close field)", t);
                }
            });
            return Optional.of(ptr);
        } catch (Throwable t) {
            throw new RuntimeException("unreachable! (new field)", t);
        }
    }

    private static Optional<MemorySegment> segmentFromInput(IndexInput input, long off, long len) throws IOException {
        if (!(input instanceof MemorySegmentAccessInput segmentInput)) {
            return Optional.empty();
        }
        return Optional.ofNullable(segmentInput.segmentSliceOrNull(off, len));
    }
}
