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

package juicebox.windowui;

import juicebox.HiC;

/**
 * @author jrobinso
 *         Date: 12/17/12
 *         Time: 9:16 AM
 */
public class HiCZoom {

    private final HiC.Unit unit;
    private final int binSize;

    public HiCZoom(HiC.Unit unit, int binSize) {
        this.unit = unit;
        this.binSize = binSize;
    }

    public HiC.Unit getUnit() {
        return unit;
    }

    public int getBinSize() {
        return binSize;
    }

    public String getKey() {
        return unit.toString() + "_" + binSize;
    }

    public String toString() {
        return getKey();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        HiCZoom hiCZoom = (HiCZoom) o;

        return (binSize == hiCZoom.binSize) && (unit == hiCZoom.unit);
    }

    @Override
    public int hashCode() {
        int result = unit.hashCode();
        result = 31 * result + binSize;
        return result;
    }
}
