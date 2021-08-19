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

import java.io.IOException;
import java.util.Iterator;
import java.util.Map;

/**
 * @author Jim Robinson
 * @since 9/24/11
 */
public interface PairIterator extends Iterator<AlignmentPair> {

    boolean hasNext();

    AlignmentPair next();

    void remove();

    void close();

    static PairIterator getIterator(String file, Map<String, Integer> chromosomeIndexes, ChromosomeHandler chromosomeHandler) throws IOException {
        if (file.endsWith(".bin")) {
            return new BinPairIterator(file);
        } else if (file.endsWith(".bn")) {
            return new ShortBinPairIterator(file);
        } else {
            return new AsciiPairIterator(file, chromosomeIndexes, chromosomeHandler, false);
        }
    }
}
