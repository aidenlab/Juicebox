/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2023 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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

package juicebox.tools.utils.original.stats;

public class PointTriple extends Triple<Integer, Integer, Integer> {

    public PointTriple(Integer first, Integer second, Integer third) {
        super(first, second, third);
    }

    public Integer getFirst() {
        return super.getFirst();
    }

    public Integer getSecond() {
        return super.getSecond();
    }

    public Integer getThird() {
        return super.getThird();
    }

    @Override
    public boolean equals(Object o) {
        if (o == null || getClass() != o.getClass()) return false;

        PointTriple triple = (PointTriple) o;

        if (!this.getFirst().equals(triple.getFirst())) return false;
        if (!this.getSecond().equals(triple.getSecond())) return false;
        return this.getThird().equals(triple.getThird());
    }

    @Override
    public int hashCode() {
        int result = this.getFirst().hashCode();
        result = 10 * result + this.getSecond().hashCode();
        result = 10 * result + this.getThird().hashCode();
        return result;
    }
}
