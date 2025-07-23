/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.github.mccullocht.nvector;

import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.readSimilarityFunction;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.readVectorEncoding;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Optional;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene95.OffHeapByteVectorValues;
import org.apache.lucene.codecs.lucene95.OffHeapFloatVectorValues;
import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.internal.hppc.IntObjectHashMap;
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.FileDataHint;
import org.apache.lucene.store.FileTypeHint;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.ReadAdvice;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

import com.github.mccullocht.nvector.NativeVectorAccess.VectorDataField;

/**
 * Reads vectors from the index segments.
 *
 * @lucene.experimental
 */
public final class NativeFlatVectorsReader extends FlatVectorsReader {
  private static final long SHALLOW_SIZE = RamUsageEstimator.shallowSizeOfInstance(NativeFlatVectorsFormat.class);

  private final IntObjectHashMap<FieldEntry> fields = new IntObjectHashMap<>();
  private final IndexInput vectorData;
  private final FieldInfos fieldInfos;

  private final Arena arena = Arena.ofAuto();

  public NativeFlatVectorsReader(SegmentReadState state, FlatVectorsScorer scorer)
      throws IOException {
    super(scorer);
    this.fieldInfos = state.fieldInfos;
    boolean success = false;
    try {
      vectorData = openDataInput(
          state,
          NativeFlatVectorsFormat.VECTOR_DATA_EXTENSION,
          NativeFlatVectorsFormat.VECTOR_DATA_CODEC_NAME,
          // Flat formats are used to randomly access vectors from their node ID that is
          // stored
          // in the HNSW graph.
          state.context.withHints(FileTypeHint.DATA, FileDataHint.KNN_VECTORS, DataAccessHint.RANDOM));
      readMetadata(state);
      success = true;
    } finally {
      if (success == false) {
        IOUtils.closeWhileHandlingException(this);
      }
    }
  }

  private int readMetadata(SegmentReadState state) throws IOException {
    String metaFileName = IndexFileNames.segmentFileName(
        state.segmentInfo.name, state.segmentSuffix, NativeFlatVectorsFormat.META_EXTENSION);
    int versionMeta = -1;
    // TODOJ: reimplement checksum support.
    // Checksummed inputs are weird and they are buffered and we have no idea how
    // large an input is until we read the whole thing.
    try (var meta = state.directory.openInput(metaFileName, IOContext.READONCE)) {
      try {
        versionMeta = CodecUtil.checkIndexHeader(
            meta,
            NativeFlatVectorsFormat.META_CODEC_NAME,
            NativeFlatVectorsFormat.VERSION_START,
            NativeFlatVectorsFormat.VERSION_CURRENT,
            state.segmentInfo.getId(),
            state.segmentSuffix);
        readFields(meta, state.fieldInfos);
      } catch (Throwable exception) {
        throw new RuntimeException(exception);
      }
    }
    return versionMeta;
  }

  private static IndexInput openDataInput(
      SegmentReadState state,
      String fileExtension,
      String codecName,
      IOContext context)
      throws IOException {
    String fileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, fileExtension);
    IndexInput in = state.directory.openInput(fileName, context);
    boolean success = false;
    try {
      // TODOJ: we should check the meta version against the data version.
      CodecUtil.checkIndexHeader(
          in,
          codecName,
          NativeFlatVectorsFormat.VERSION_START,
          NativeFlatVectorsFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      CodecUtil.retrieveChecksum(in);
      success = true;
      return in;
    } finally {
      if (success == false) {
        IOUtils.closeWhileHandlingException(in);
      }
    }
  }

  private void readFields(IndexInput meta, FieldInfos infos) throws IOException {
    for (int fieldNumber = meta.readInt(); fieldNumber != -1; fieldNumber = meta.readInt()) {
      FieldInfo info = infos.fieldInfo(fieldNumber);
      if (info == null) {
        throw new CorruptIndexException("Invalid field number: " + fieldNumber, meta);
      }
      FieldEntry fieldEntry = FieldEntry.create(meta, this.vectorData, this.arena, info);
      fields.put(info.number, fieldEntry);
    }
  }

  @Override
  public long ramBytesUsed() {
    return NativeFlatVectorsReader.SHALLOW_SIZE + fields.ramBytesUsed();
  }

  @Override
  public void checkIntegrity() throws IOException {
    CodecUtil.checksumEntireFile(vectorData);
  }

  @Override
  public NativeFlatVectorsReader getMergeInstance() {
    try {
      // Update the read advice since vectors are guaranteed to be accessed
      // sequentially for merge
      this.vectorData.updateReadAdvice(ReadAdvice.SEQUENTIAL);
      return this;
    } catch (IOException exception) {
      throw new UncheckedIOException(exception);
    }
  }

  FieldEntry getFieldEntry(String field, VectorEncoding expectedEncoding) {
    final FieldInfo info = fieldInfos.fieldInfo(field);
    FieldEntry fieldEntry = null;
    if (info == null || (fieldEntry = fields.get(info.number)) == null) {
      throw new IllegalArgumentException("field=\"" + field + "\" not found");
    }
    if (fieldEntry.vectorEncoding != expectedEncoding) {
      throw new IllegalArgumentException(
          "field=\""
              + field
              + "\" is encoded as: "
              + fieldEntry.vectorEncoding
              + " expected: "
              + expectedEncoding);
    }
    return fieldEntry;
  }

  @Override
  public FloatVectorValues getFloatVectorValues(String field) throws IOException {
    final FieldEntry fieldEntry = getFieldEntry(field, VectorEncoding.FLOAT32);
    return OffHeapFloatVectorValues.load(
        fieldEntry.similarityFunction,
        vectorScorer,
        fieldEntry.ordToDoc,
        fieldEntry.vectorEncoding,
        fieldEntry.dimension,
        fieldEntry.vectorDataOffset,
        fieldEntry.vectorDataLength,
        vectorData);
  }

  @Override
  public ByteVectorValues getByteVectorValues(String field) throws IOException {
    final FieldEntry fieldEntry = getFieldEntry(field, VectorEncoding.BYTE);
    return OffHeapByteVectorValues.load(
        fieldEntry.similarityFunction,
        vectorScorer,
        fieldEntry.ordToDoc,
        fieldEntry.vectorEncoding,
        fieldEntry.dimension,
        fieldEntry.vectorDataOffset,
        fieldEntry.vectorDataLength,
        vectorData);
  }

  @Override
  public RandomVectorScorer getRandomVectorScorer(String field, float[] target) throws IOException {
    final FieldEntry fieldEntry = getFieldEntry(field, VectorEncoding.FLOAT32);
    if (fieldEntry.nativeField.isPresent()) {
      return new RandomVectorScorer() {
        // Use an auto arena; RandomVectorScorer is not closeable so we can't be sure
        // that native segments will be freed.
        private Arena arena = Arena.ofAuto();
        private MemorySegment query = NativeVectorAccess.vectorAsMemorySegment(target, arena);
        private VectorDataField field = fieldEntry.nativeField.get();

        @Override
        public int maxOrd() {
          return fieldEntry.size;
        }

        @Override
        public float score(int node) throws IOException {
          float[] scores = { 0.0f };
          bulkScore(new int[] { node }, scores, 1);
          return scores[0];
        }

        @Override
        public void bulkScore(int[] nodes, float[] scores, int numNodes) throws IOException {
          NativeVectorAccess.scoreOrds(this.field, query, nodes, scores, numNodes);
        }
      };
    } else {
      return vectorScorer.getRandomVectorScorer(
          fieldEntry.similarityFunction,
          OffHeapFloatVectorValues.load(
              fieldEntry.similarityFunction,
              vectorScorer,
              fieldEntry.ordToDoc,
              fieldEntry.vectorEncoding,
              fieldEntry.dimension,
              fieldEntry.vectorDataOffset,
              fieldEntry.vectorDataLength,
              vectorData),
          target);
    }
  }

  @Override
  public RandomVectorScorer getRandomVectorScorer(String field, byte[] target) throws IOException {
    final FieldEntry fieldEntry = getFieldEntry(field, VectorEncoding.BYTE);
    return vectorScorer.getRandomVectorScorer(
        fieldEntry.similarityFunction,
        OffHeapByteVectorValues.load(
            fieldEntry.similarityFunction,
            vectorScorer,
            fieldEntry.ordToDoc,
            fieldEntry.vectorEncoding,
            fieldEntry.dimension,
            fieldEntry.vectorDataOffset,
            fieldEntry.vectorDataLength,
            vectorData),
        target);
  }

  @Override
  public void finishMerge() throws IOException {
    // This makes sure that the access pattern hint is reverted back since HNSW
    // implementation
    // needs it
    this.vectorData.updateReadAdvice(ReadAdvice.RANDOM);
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(vectorData);
  }

  record FieldEntry(
      VectorSimilarityFunction similarityFunction,
      VectorEncoding vectorEncoding,
      long vectorDataOffset,
      long vectorDataLength,
      int dimension,
      int size,
      OrdToDocDISIReaderConfiguration ordToDoc,
      Optional<NativeVectorAccess.VectorDataField> nativeField,
      FieldInfo info) {

    FieldEntry {
      if (similarityFunction != info.getVectorSimilarityFunction()) {
        throw new IllegalStateException(
            "Inconsistent vector similarity function for field=\""
                + info.name
                + "\"; "
                + similarityFunction
                + " != "
                + info.getVectorSimilarityFunction());
      }
      int infoVectorDimension = info.getVectorDimension();
      if (infoVectorDimension != dimension) {
        throw new IllegalStateException(
            "Inconsistent vector dimension for field=\""
                + info.name
                + "\"; "
                + infoVectorDimension
                + " != "
                + dimension);
      }

      int byteSize = switch (info.getVectorEncoding()) {
        case BYTE -> Byte.BYTES;
        case FLOAT32 -> Float.BYTES;
      };
      long vectorBytes = Math.multiplyExact((long) infoVectorDimension, byteSize);
      long numBytes = Math.multiplyExact(vectorBytes, size);
      if (numBytes != vectorDataLength) {
        throw new IllegalStateException(
            "Vector data length "
                + vectorDataLength
                + " not matching size="
                + size
                + " * dim="
                + dimension
                + " * byteSize="
                + byteSize
                + " = "
                + numBytes);
      }
    }

    static FieldEntry create(IndexInput input, IndexInput data, Arena arena, FieldInfo info) throws IOException {
      final var nativeField = NativeVectorAccess.VectorDataField.create(input, data, arena);
      final VectorEncoding vectorEncoding = readVectorEncoding(input);
      final VectorSimilarityFunction similarityFunction = readSimilarityFunction(input);
      final var vectorDataOffset = input.readVLong();
      final var vectorDataLength = input.readVLong();
      final var dimension = input.readVInt();
      final var size = input.readInt();
      final var ordToDoc = OrdToDocDISIReaderConfiguration.fromStoredMeta(input, size);

      return new FieldEntry(
          similarityFunction,
          vectorEncoding,
          vectorDataOffset,
          vectorDataLength,
          dimension,
          size,
          ordToDoc,
          nativeField,
          info);
    }
  }
}