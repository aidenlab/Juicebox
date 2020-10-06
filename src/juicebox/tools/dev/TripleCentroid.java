/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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

package juicebox.tools.dev;

import java.util.HashSet;
import java.util.Set;

public class TripleCentroid {

    private final Integer chrIndex;
    private final Set<Integer> x1s = new HashSet<>();
    private final Set<Integer> x2s = new HashSet<>();
    private final Set<Integer> x3s = new HashSet<>();
    private Integer x1;
    private Integer x2;
    private Integer x3;

    public TripleCentroid(Integer chrIndex, Integer x1, Integer x2, Integer x3) {
        this.chrIndex = chrIndex;
        this.x1 = x1;
        this.x2 = x2;
        this.x3 = x3;
        x1s.add(x1);
        x2s.add(x2);
        x3s.add(x3);
    }

    public void consumeDuplicate(IntraChromTriple triple) {
        x1s.add(triple.getX1());
        x2s.add(triple.getX2());
        x3s.add(triple.getX3());
        x1 = average(x1s);
        x2 = average(x2s);
        x3 = average(x3s);
    }

    private Integer average(Set<Integer> xs) {
        int accum = 0;
        for (int val : xs) {
            accum += val;
        }
        return accum / xs.size();
    }

    public boolean hasOverlapWithTriple(IntraChromTriple triple, int tolerance) {

        return Math.abs(triple.getX1() - x1) < tolerance &&
                Math.abs(triple.getX2() - x2) < tolerance &&
                Math.abs(triple.getX3() - x3) < tolerance;
    }

    public IntraChromTriple toIntraChromTriple() {
        return new IntraChromTriple(chrIndex, x1, x2, x3);
    }
}
