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
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.data;

import htsjdk.tribble.util.LittleEndianInputStream;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

public class NormFactorMapReader {
    private final int version, nFactors;
    private final Map<Integer, Double> normFactors = new LinkedHashMap<>();

    public NormFactorMapReader(int nFactors, int version, LittleEndianInputStream dis)
            throws IOException {
        this.version = version;
        this.nFactors = nFactors;

        for (int j = 0; j < nFactors; j++) {
            int chrIdx = dis.readInt();
            if (version > 8) {
                normFactors.put(chrIdx, (double) dis.readFloat());
            } else {
                normFactors.put(chrIdx, dis.readDouble());
            }
        }
    }

    public Map<Integer, Double> getNormFactors() {
        return normFactors;
    }

    public int getOffset() {
        if (version > 8) {
            return 8 * nFactors;
        } else {
            return 12 * nFactors;
        }
    }
}
