/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2021 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.tools.utils.original.mnditerator;


import juicebox.data.ChromosomeHandler;
import juicebox.tools.utils.original.Chunk;

import java.io.IOException;
import java.util.Map;

/**
 * @author Jim Robinson
 * @since 9/24/11
 */
public class RandomAccessAsciiPairIterator extends AsciiPairIterator {
    public RandomAccessAsciiPairIterator(String path, Map<String, Integer> chromosomeOrdinals, long mndIndex, int mndChunk, ChromosomeHandler handler) throws IOException {
        super(path, chromosomeOrdinals, new Chunk(mndIndex, mndChunk), handler);
    }

    /*

    to use RandomAccessFile, we only need to adjust constructor below, don't duplicate all code from asciipairiterator
    however, since this isn't even being used, I'm just commenting it out - MSS

    public AsciiPairIterator(String path, Map<String, Integer> chromosomeOrdinals, long mndIndex,
                             int mndChunk, ChromosomeHandler handler) throws IOException {
        this.handler = handler;
        if (path.endsWith(".gz")) {
            System.err.println("Multithreading with indexed mnd currently only works with unzipped mnd");
            System.exit(70);
        } else {
            reader = new RandomAccessFile(path, "r");
            reader.getChannel().position(mndIndex);
            this.mndStart = mndIndex;
            this.mndChunkSize = mndChunk;
            this.stopAfterChunk = true;
        }
        this.chromosomeOrdinals = chromosomeOrdinals;
        advance();
    }

     */
}
