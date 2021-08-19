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

import java.io.BufferedReader;
import java.io.IOException;

public abstract class GenericPairIterator implements PairIterator {

    protected final MNDFileParser mndFileParser;
    protected AlignmentPair nextPair = null;
    protected BufferedReader reader;

    public GenericPairIterator(MNDFileParser mndFileParser) {
        this.mndFileParser = mndFileParser;
    }

    public boolean hasNext() {
        return nextPair != null;
    }

    public AlignmentPair next() {
        AlignmentPair p = nextPair;
        advance();
        return p;
    }

    protected void advance() {

        try {
            String nextLine = reader.readLine();
            nextLine = validateLine(nextLine);
            if (nextLine != null) {
                nextPair = mndFileParser.parse(nextLine);
                return;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        nextPair = null;
    }

    protected String validateLine(String nextLine) {
        return nextLine;
    }

    public void remove() {
        // Not implemented
    }

    public void close() {
        try {
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
        }
    }
}
