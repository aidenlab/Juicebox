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

package juicebox.data.basics;
import java.util.Objects;

public class Chromosome {
    private final String name;
    private int index;
    private long length = 0;
    
    public Chromosome(int index, String name, long length) {
        this.index = index;
        this.name = name;
        this.length = length;
    }

    public int getIndex() {
        return this.index;
    }

    public void setIndex(int ii) {
        this.index = ii;
    }
    
    public long getLength() {
        return this.length;
    }

    public String getName() {
        return this.name;
    }

    public String toString() {
        return this.name;
    }

    public boolean equals(Object obj) {
        return obj instanceof Chromosome && ((Chromosome) obj).getIndex() == this.getIndex() && ((Chromosome) obj).getLength() == this.getLength();
    }

    public int hashCode() {
        return Objects.hash(this.index, this.length);
    }

    public org.broad.igv.feature.Chromosome toIGVChromosome() {
        return new org.broad.igv.feature.Chromosome(index, name, (int) length); // assumed for IGV
    }
}

