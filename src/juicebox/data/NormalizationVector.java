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

package juicebox.data;

import juicebox.HiC;
import juicebox.windowui.NormalizationType;

/**
 * @author jrobinso
 *         Date: 2/10/13
 *         Time: 9:19 AM
 */
public class NormalizationVector {

    private final NormalizationType type;
    private final int chrIdx;
    private final HiC.Unit unit;
    private final int resolution;
    private final double[] data;

    public NormalizationVector(NormalizationType type, int chrIdx, HiC.Unit unit, int resolution, double[] data) {
        this.type = type;
        this.chrIdx = chrIdx;
        this.unit = unit;
        this.resolution = resolution;
        this.data = data;
    }

    public static String getKey(NormalizationType type, int chrIdx, String unit, int resolution) {
        return type + "_" + chrIdx + "_" + unit + "_" + resolution;
    }

    public static void main(String[] args) {
        System.out.println(getKey(NormalizationType.GW_KR, 1, "x", 0));
    }

    public String getKey() {
        return NormalizationVector.getKey(type, chrIdx, unit.toString(), resolution);
    }

    public double[] getData() {
        return data;
    }
}
