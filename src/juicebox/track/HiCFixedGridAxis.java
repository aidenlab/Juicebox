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

package juicebox.track;

import org.broad.igv.Globals;

/**
 * @author jrobinso
 *         Date: 9/14/12
 *         Time: 8:54 AM
 */
public class HiCFixedGridAxis implements HiCGridAxis {

    private final int binCount;
    private final int binSize;
    private final int igvZoom;
    private final int[] sites;

    public HiCFixedGridAxis(int binCount, int binSize, int[] sites) {

        this.binCount = binCount;
        this.binSize = binSize;
        this.sites = sites;

        // Compute an approximate igv zoom level
        igvZoom = Math.max(0, (int) (Math.log(binCount / 700) / Globals.log2));

    }

    @Override
    public int getBinSize() {
        return binSize;
    }

    @Override
    public int getGenomicStart(double binNumber) {
        return (int) (binNumber * binSize);
    }

    @Override
    public int getGenomicEnd(double binNumber) {
        return (int) ((binNumber + 1) * binSize);
    }

    @Override
    public int getGenomicMid(double binNumber) {
        return (int) ((binNumber + 0.5) * binSize);
    }

    @Override
    public int getIGVZoom() {
        return igvZoom;
    }

    @Override
    public int getBinNumberForGenomicPosition(int genomicPosition) {
        return (int) (genomicPosition / ((double) binSize));
    }

    @Override
    public int getBinNumberForFragment(int fragment) {

        if (fragment < sites.length && fragment >= 0) {
            int genomicPosition = sites[fragment];
            return getBinNumberForGenomicPosition(genomicPosition);
        }
        throw new RuntimeException("Fragment: " + fragment + " is out of range");
    }

    @Override
    public int getBinCount() {
        return binCount;
    }

}
