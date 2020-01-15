/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2018 Broad Institute, Aiden Lab
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

import juicebox.data.ChromosomeHandler;
import juicebox.data.feature.Feature;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class IntraChromTriple extends Feature implements Comparable<IntraChromTriple> {

    private final Integer x1, x2, x3, chrIndex;

    public IntraChromTriple(int chrIndex, int x1, int x2, int x3) {
        this.chrIndex = chrIndex;
        this.x1 = x1;
        this.x2 = x2;
        this.x3 = x3;
    }

    private IntraChromTriple(int chrIndex, List<Integer> ints) {
        this.chrIndex = chrIndex;
        Collections.sort(ints);
        x1 = ints.get(0);
        x2 = ints.get(1);
        x3 = ints.get(2);
    }

    /**
     * assuming syntax
     * chr1 x1 chr2 x2 chr3 x3
     */
    public static IntraChromTriple parse(String line, ChromosomeHandler handler) {
        String[] parsedText = line.split("\\s+");
        int chrIndex;
        if (parsedText[0].equalsIgnoreCase(parsedText[2]) && parsedText[2].equalsIgnoreCase(parsedText[4])) {
            chrIndex = handler.getChromosomeFromName(parsedText[0]).getIndex();
        } else {
            return null;
        }
        Integer y1 = Integer.parseInt(parsedText[1]);
        Integer y2 = Integer.parseInt(parsedText[3]);
        Integer y3 = Integer.parseInt(parsedText[5]);

        if (Math.abs(y1 - y2) > 20 && Math.abs(y1 - y3) > 20 && Math.abs(y2 - y3) > 20) {
            List<Integer> ints = new ArrayList<>();
            ints.add(y1);
            ints.add(y2);
            ints.add(y3);
            return new IntraChromTriple(chrIndex, ints);
        }

        return null;
    }

    @Override
    public int compareTo(IntraChromTriple o) {
        int comparison = chrIndex.compareTo(o.chrIndex);
        if (comparison == 0) comparison = x1.compareTo(o.x1);
        if (comparison == 0) comparison = x2.compareTo(o.x2);
        if (comparison == 0) comparison = x3.compareTo(o.x3);
        return comparison;
    }

    @Override
    public String getKey() {
        return "" + chrIndex;
    }

    @Override
    public String toString() {
        return chrIndex + "\t" + x1 + "\t" + x2 + "\t" + x3;
    }

    @Override
    public Feature deepClone() {
        return new IntraChromTriple(chrIndex, x1, x2, x3);
    }

    public Integer getX1() {
        return x1;
    }

    public Integer getX2() {
        return x2;
    }

    public Integer getX3() {
        return x3;
    }

    public TripleCentroid toTripleCentroid() {
        return new TripleCentroid(chrIndex, x1, x2, x3);
    }
}
