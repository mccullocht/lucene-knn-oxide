package com.github.mccullocht.nvector;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.tests.index.BaseKnnVectorsFormatTestCase;
import org.apache.lucene.tests.util.TestUtil;

public class NativeHnswVectorsFormatTest extends BaseKnnVectorsFormatTestCase {
    private static final KnnVectorsFormat FORMAT = new NativeHnswVectorsFormat();

    @Override
    protected Codec getCodec() {
        return TestUtil.alwaysKnnVectorsFormat(FORMAT);
    }
}
