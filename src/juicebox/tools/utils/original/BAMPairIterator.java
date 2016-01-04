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

import htsjdk.samtools.util.CloseableIterator;
import org.broad.igv.sam.Alignment;
import org.broad.igv.sam.ReadMate;
import org.broad.igv.sam.reader.AlignmentReader;
import org.broad.igv.sam.reader.AlignmentReaderFactory;

import java.io.IOException;
import java.util.Map;

/**
 * TODO - should this be deleted?
 * Also, the chromosomeOrdinals map seems to always be empty
 *
 * @author Jim Robinson
 * @date 9/24/11
 */
public class BAMPairIterator implements PairIterator {

    private AlignmentPair nextPair = null;
    private AlignmentPair preNext = null;
    private CloseableIterator<?> iterator;
    private AlignmentReader<?> reader;
    // Map of name -> index
    private Map<String, Integer> chromosomeOrdinals;

    public BAMPairIterator(String path) throws IOException {

        this.reader = AlignmentReaderFactory.getReader(path, false);

        this.iterator = reader.iterator();
        advance();
    }

    private void advance() {

        while (iterator.hasNext()) {
            Alignment alignment = (Alignment) iterator.next();

            final ReadMate mate = alignment.getMate();
            if (alignment.isPaired() && alignment.isMapped() && alignment.getMappingQuality() > 0 &&
                    mate != null && mate.isMapped()) {
                // Skip "normal" insert sizes
                if ((!alignment.getChr().equals(mate.getChr())) || alignment.getInferredInsertSize() > 1000) {

                    // Each pair is represented twice in the file,  keep the record with the "leftmost" coordinate

                    if ((alignment.getChr().equals(mate.getChr()) && alignment.getStart() < mate.getStart()) ||
                            (alignment.getChr().compareTo(mate.getChr()) < 0)) {
                        final String chrom1 = alignment.getChr();
                        final String chrom2 = mate.getChr();
                        if (chromosomeOrdinals.containsKey(chrom1) && chromosomeOrdinals.containsKey(chrom2)) {
                            int chr1 = chromosomeOrdinals.get(chrom1);
                            int chr2 = chromosomeOrdinals.get(chrom2);
                            //  nextPair = new AlignmentPair(chr1, alignment.getStart(), chr2, mate.getStart());
                        }
                        return;
                    }
                }
            }

        }
        nextPair = null;

    }

    public boolean hasNext() {
        return preNext != null || nextPair != null;  //To change body of implemented methods use File | Settings | File Templates.
    }

    public AlignmentPair next() {
        if (preNext == null) {
            AlignmentPair p = nextPair;
            advance();
            return p;
        } else {
            AlignmentPair p = preNext;
            preNext = null;
            return p;
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
        // Not implemented
    }

    public void close() {
        iterator.close();
        try {
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
        }
    }

}
