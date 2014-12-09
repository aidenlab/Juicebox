/*
 * Copyright (C) 2011-2014 Aiden Lab - All Rights Reserved
 * SOFTWARE COPYRIGHT NOTICE
 * This software and its documentation are the copyright of the Aiden Lab All rights are reserved.
 *
 * This software is supplied without any warranty or guaranteed support whatsoever. Aiden Lab is not responsible for its use, misuse, or functionality.
 *
 * This software is licensed under the terms of the GNU Lesser General Public License (LGPL),
 * Version 2.1 which is available at http://www.opensource.org/licenses/lgpl-2.1.php.
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
