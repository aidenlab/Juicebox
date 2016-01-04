/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2016 Broad Institute, Aiden Lab
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
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.tools.utils.original;

import htsjdk.tribble.util.LittleEndianInputStream;

import java.io.BufferedInputStream;
import java.io.EOFException;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Map;
//import java.util.zip.GZIPInputStream;

/**
 * @author Jim Robinson
 * @date 4/7/12
 */
public class BinPairIterator implements PairIterator {

    private LittleEndianInputStream is;
    private AlignmentPair next;
    private AlignmentPair preNext;

    /**
     * TODO -- chromosomeIndexes is ignored for now, but should be used to map the chromosome stored in the
     * TODO -- bin pair file with an integer index. The current assumption is the chromosome map in
     * TODO -- the bin pair file is the same being used for the hic file,  a fragile assumption.
     *
     * @param path
     * @param chromosomeIndexes
     * @throws IOException
     */
    public BinPairIterator(String path, Map<String, Integer> chromosomeIndexes) throws IOException {
        is = new LittleEndianInputStream(new BufferedInputStream(new FileInputStream(path)));
        advance();
    }

    public boolean hasNext() {
        return preNext != null || next != null;  //To change body of implemented methods use File | Settings | File Templates.
    }

    public AlignmentPair next() {
        if (preNext == null) {
            AlignmentPair retValue = next;
            advance();
            return retValue;
        } else {
            AlignmentPair retValue = preNext;
            preNext = null;
            return retValue;
        }
    }

    @Override
    public void push(AlignmentPair pair) {
        if (preNext != null) {
            throw new RuntimeException("Cannot push more than one alignment pair back on stack");
        } else {
            preNext = pair;
        }
    }

    public void remove() {
        //To change body of implemented methods use File | Settings | File Templates.
    }

    public void close() {
        if (is != null) try {
            is.close();
        } catch (IOException e) {
            e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
        }
    }

    private void advance() {

        try {
            boolean str1 = (is.readByte() != 0);
            int chr1 = is.readInt();
            int pos1 = is.readInt();
            int frag1 = is.readInt();
            boolean str2 = (is.readByte() != 0);
            int chr2 = is.readInt();
            int pos2 = is.readInt();
            int frag2 = is.readInt();
            next = new AlignmentPair(str1, chr1, pos1, frag1, 1000, str2, chr2, pos2, frag2, 1000);
        } catch (IOException e) {
            next = null;
            if (!(e instanceof EOFException)) {
                e.printStackTrace();
            }
        }
    }
}
