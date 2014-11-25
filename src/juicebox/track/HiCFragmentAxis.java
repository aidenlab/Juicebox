/*
 * Copyright (c) 2007-2012 The Broad Institute, Inc.
 * SOFTWARE COPYRIGHT NOTICE
 * This software and its documentation are the copyright of the Broad Institute, Inc. All rights are reserved.
 *
 * This software is supplied without any warranty or guaranteed support whatsoever. The Broad Institute is not responsible for its use, misuse, or functionality.
 *
 * This software is licensed under the terms of the GNU Lesser General Public License (LGPL),
 * Version 2.1 which is available at http://www.opensource.org/licenses/lgpl-2.1.php.
 */

package juicebox.track;

import org.broad.igv.Globals;

/**
 * @author jrobinso
 *         Date: 9/14/12
 *         Time: 8:49 AM
 */
public class HiCFragmentAxis implements HiCGridAxis {

    private final int binSize;  // bin size in fragments
    private final int igvZoom;
    private final int[] sites;
    private final int chrLength;


    /**
     * @param sites     ordered by start position.  Its assumed bins are contiguous, no gaps and no overlap.
     * @param chrLength
     */
    public HiCFragmentAxis(int binSize, int[] sites, int chrLength) {

        this.binSize = binSize;
        this.sites = sites;
        this.chrLength = chrLength;

        // Compute an approximate igv zoom level
        double averageBinSizeInBP = ((double) this.chrLength) / (sites.length + 1) * binSize;
        igvZoom = (int) (Math.log((this.chrLength / 700) / averageBinSizeInBP) / Globals.log2);
    }


    @Override
    public int getGenomicStart(double binNumber) {
        int fragNumber = (int) binNumber * binSize;
        int siteIdx = Math.min(fragNumber, sites.length - 1);

        if (binNumber >= sites.length) {
            binNumber = sites.length - 1;
        }

        return binNumber == 0 ? 0 : sites[siteIdx - 1];

    }

    @Override
    public int getGenomicEnd(double binNumber) {
        int fragNumber = (int) (binNumber + 1) * binSize - 1;
        int siteIdx = Math.min(fragNumber, sites.length - 1);
        return siteIdx < sites.length ? sites[siteIdx] : chrLength;
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
    public int getGenomicMid(double binNumber) {
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
    public int getBinNumberForGenomicPosition(int position) {
        return getFragmentNumberForGenomicPosition(position) / binSize;
    }


    /**
     * Return bin that this position lies on.  Fragment 0 means position < sites[0].
     * Fragment 1 means position >= sites[0] and < sites[1].
     *
     * @param position The genome position to search for within that array
     * @return The fragment location such that position >= sites[retVal-1] and position <  sites[retVal]
     */
    public int getFragmentNumberForGenomicPosition(int position) {

        int lo = 0;
        int hi = sites.length - 1;

        // Eliminate the extreme cases
        if (position < sites[0]) return 0;
        if (position >= sites[hi]) return sites.length;

        while (lo <= hi) {

            int mid = (lo + hi) >>> 1;
            if(position >= sites[mid-1] && position < sites[mid]) {
                return mid;
            }
            else if(position >= sites[mid]) {
                lo = mid+1;
            }
            else  {
                hi = mid;
            }

        }

        // Not found
        return -1;

    }

    @Override
    public int getBinNumberForFragment(int fragment) {
        if (fragment <= sites.length) {
            return fragment / binSize;
        } else {
            throw new RuntimeException("Fragment: " + fragment + " is out of range");
        }
    }

    @Override
    public int getBinCount() {
        return (sites.length / binSize) + 1;
    }


}
