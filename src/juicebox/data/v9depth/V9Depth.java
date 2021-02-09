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

package juicebox.data.v9depth;

public abstract class V9Depth {
    protected final int blockBinCount;
    protected double BASE;

    V9Depth(int blockBinCount) {
        this.blockBinCount = blockBinCount;
    }

    public static V9Depth setDepthMethod(int depthBase, int blockBinCount) {
        if (depthBase > 1) {
            return new LogDepth(depthBase, blockBinCount);
        } else if (depthBase < 0) {
            return new ConstantDepth(-depthBase, blockBinCount);
        }

        // Default
        return new LogDepth(2, blockBinCount);
    }

    public int getDepth(int val1, int val2) {
        return logBase(Math.abs(val1 - val2) / Math.sqrt(2) / blockBinCount);
    }

    protected abstract int logBase(double value);
}
