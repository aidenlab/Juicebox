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

import juicebox.windowui.NormalizationType;

import java.util.ArrayList;
import java.util.List;
//import java.util.Map;

/**
 * @author jrobinso
 *         Date: 12/26/12
 *         Time: 9:30 PM
 */
public class CombinedExpectedValueFunction implements ExpectedValueFunction {

    private final List<ExpectedValueFunction> densityFunctions;
    private double[] expectedValues = null;

    public CombinedExpectedValueFunction(ExpectedValueFunction densityFunction) {
        this.densityFunctions = new ArrayList<ExpectedValueFunction>();
        densityFunctions.add(densityFunction);
    }

    public void addDensityFunction(ExpectedValueFunction densityFunction) {
        // TODO -- verify same unit, binsize, type, denisty array size
        densityFunctions.add(densityFunction);
    }

    @Override
    public double getExpectedValue(int chrIdx, int distance) {
        double sum = 0;
        for (ExpectedValueFunction df : densityFunctions) {
            sum += df.getExpectedValue(chrIdx, distance);
        }
        return sum;
    }

    @Override
    public double[] getExpectedValues() {
        if (expectedValues != null) return expectedValues;
        int length = 0;
        for (ExpectedValueFunction df : densityFunctions) { // Removed cast to ExpectedValueFunctionImpl; change back if errors
            length = Math.max(length, df.getExpectedValues().length);
        }
        expectedValues = new double[length];
        for (ExpectedValueFunction df : densityFunctions) {

            double[] current = df.getExpectedValues();
            for (int i = 0; i < current.length; i++) {
                expectedValues[i] += current[i];
            }
        }
        return expectedValues;
    }

    @Override
    public int getLength() {
        return densityFunctions.get(0).getLength();
    }

    @Override
    public NormalizationType getNormalizationType() {
        return densityFunctions.get(0).getNormalizationType();
    }

    @Override
    public String getUnit() {
        return densityFunctions.get(0).getUnit();
    }

    @Override
    public int getBinSize() {
        return densityFunctions.get(0).getBinSize();
    }

}
