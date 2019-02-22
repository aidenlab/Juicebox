/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2019 Broad Institute, Aiden Lab
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

package juicebox.tools.utils.juicer.hiccups;

import juicebox.data.MatrixZoomData;
import org.broad.igv.feature.Chromosome;

public class HiCCUPSRegionContainer {
    private final MatrixZoomData zd;
    private final double[] normalizationVector;
    private final double[] expectedVector;
    private final int[] rowBounds;
    private final int[] columnBounds;
    private final Chromosome chromosome;

    public HiCCUPSRegionContainer(Chromosome chromosome, MatrixZoomData zd, double[] normalizationVector, double[] expectedVector,
                                  int[] rowBounds, int[] columnBounds) {
        this.chromosome = chromosome;
        this.zd = zd;
        this.normalizationVector = normalizationVector;
        this.expectedVector = expectedVector;
        this.rowBounds = rowBounds;
        this.columnBounds = columnBounds;
    }

    public MatrixZoomData getZd() {
        return zd;
    }

    public double[] getNormalizationVector() {
        return normalizationVector;
    }

    public double[] getExpectedVector() {
        return expectedVector;
    }

    public int[] getRowBounds() {
        return rowBounds;
    }

    public int[] getColumnBounds() {
        return columnBounds;
    }

    public Chromosome getChromosome() {
        return chromosome;
    }
}
