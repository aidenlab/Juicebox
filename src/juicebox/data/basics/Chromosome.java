/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab
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

package juicebox.data.basics;

import org.broad.igv.feature.Cytoband;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public class Chromosome {
    private String name;
    private int index;
    private int length = 0;
    private List<Cytoband> cytobands;

    public Chromosome(int index, String name, int length) {
        this.index = index;
        this.name = name;
        this.length = length;
        Cytoband cytoband = new Cytoband(name);
        cytoband.setStart(0);
        cytoband.setEnd(length);
        this.cytobands = Arrays.asList(cytoband);
    }

    public int getIndex() {
        return this.index;
    }

    public void setIndex(int ii) {
        this.index = ii;
    }

    public List<Cytoband> getCytobands() {
        return this.cytobands;
    }

    public void setCytobands(List<Cytoband> cytobands) {
        this.cytobands = cytobands;
    }

    public int getLength() {
        return this.length;
    }

    public String getName() {
        return this.name;
    }

    public String toString() {
        return this.name;
    }

    public boolean equals(Object obj) {
        return obj instanceof org.broad.igv.feature.Chromosome && ((org.broad.igv.feature.Chromosome) obj).getIndex() == this.getIndex() && ((org.broad.igv.feature.Chromosome) obj).getLength() == this.getLength();
    }

    public int hashCode() {
        return Objects.hash(this.index, this.length);
    }

    public org.broad.igv.feature.Chromosome toIGVChromosome() {
        return new org.broad.igv.feature.Chromosome(index, name, length);
    }
}

