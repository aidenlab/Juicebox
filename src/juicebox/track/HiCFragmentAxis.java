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

package juicebox.track;

/**
 * @author jrobinso
 *         Date: 9/14/12
 *         Time: 8:49 AM
 */
public class HiCFragmentAxis implements HiCGridAxis {

    private final int binSize;  // bin size in fragments
    private final int igvZoom;
    private final int[] sites;
    private final long chrLength;
    private final double log2 = Math.log(2.0D);
    
    
    /**
     * @param sites     ordered by start position.  Its assumed bins are contiguous, no gaps and no overlap.
     * @param chrLength
     */
    public HiCFragmentAxis(int binSize, int[] sites, long chrLength) {
        
        this.binSize = binSize;
        this.sites = sites;
        this.chrLength = chrLength;
        
        // Compute an approximate igv zoom level
        double averageBinSizeInBP = ((double) this.chrLength) / (sites.length + 1) * binSize;
        igvZoom = (int) (Math.log((this.chrLength / 700) / averageBinSizeInBP) / log2);
    }

    @Override
    public long getGenomicStart(double binNumber) {
        long fragment = (long) binNumber * binSize;
        if (fragment <= 0) return 0;
        return fragment > sites.length ? chrLength : sites[(int) fragment - 1];
    }
    
    @Override
    public long getGenomicEnd(double binNumber) {
        long siteIdx = ((long) binNumber + 1) * binSize - 1;
        if (siteIdx < 0) return 0;
        return siteIdx < sites.length ? sites[(int) siteIdx] : chrLength;
    }


//    @Override
//    public int getGenomicStart(double binNumber) {
//
//
//        if (binNumber >= sites.length) {
//            binNumber = sites.length - 1;
//        }
//
//        int bin = (int) binNumber;
//        double remainder = binNumber % bin;
//
//        double start = binNumber == 0 ? 0 : sites[bin-1];
//        double end = sites[bin];
//        double delta = end - start;
//
//        return (int) (start + remainder * delta);
//
//    }
//
//    @Override
//    public int getGenomicEnd(double binNumber) {
//
//        if (binNumber >= sites.length) {
//            return chrLength;
//        }
//
//        int bin = (int) binNumber;
//        double remainder = binNumber % bin;
//
//        double start = binNumber == 0 ? 0 : sites[bin-1];
//        double end = sites[bin];
//
//        return sites[bin];
//
//
//    }
    
    @Override
    public long getGenomicMid(double binNumber) {
        return (getGenomicStart(binNumber) + getGenomicEnd(binNumber)) / 2;
    }


    @Override
    public int getIGVZoom() {
        return igvZoom;
    }
    
    
    /**
     * Return bin that this position lies on.  Fragment 0 means position < sites[0].
     * Fragment 1 means position >= sites[0] and < sites[1].
     *
     * @param position The genome position to search for within that array
     * @return The fragment location such that position >= sites[retVal-1] and position <  sites[retVal]
     */
    @Override
    public int getBinNumberForGenomicPosition(long position) {
        return getFragmentNumberForGenomicPosition(position) / binSize;
    }


    /**
     * Return bin that this position lies on.  Fragment 0 means position < sites[0].
     * Fragment 1 means position >= sites[0] and < sites[1].
     *
     * @param position The genome position to search for within that array
     * @return The fragment location such that position >= sites[retVal-1] and position <  sites[retVal]
     */
    private int getFragmentNumberForGenomicPosition(long position) {

        int lo = 0;
        int hi = sites.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (position >= sites[mid]) {
                lo = mid + 1;
            } else {
                hi = mid;
            }

        }

        return lo;

    }

    @Override
    public int getBinNumberForFragment(int fragment) {
        if (fragment >= 0 && fragment <= sites.length) {
            return fragment / binSize;
        } else {
            throw new RuntimeException("Fragment: " + fragment + " is out of range");
        }
    }
    
    @Override
    public long getBinCount() {
        return (sites.length / binSize) + 1;
    }

    @Override
    public int getBinSize() {
        return binSize;
    }


}
