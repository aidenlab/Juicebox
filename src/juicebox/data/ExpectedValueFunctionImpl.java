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

import java.util.Map;

/**
 * Utility holder for Density calculation, for O/E maps.
 *
 * @author Jim Robinson
 * @author Neva Cherniavsky
 * @since 8/27/12
 */
public class ExpectedValueFunctionImpl implements ExpectedValueFunction {

    private final int binSize;
    private final NormalizationType type;
    private final String unit;

    private final Map<Integer, Double> normFactors;

    private final double[] expectedValues;

    public ExpectedValueFunctionImpl(NormalizationType type, String unit, int binSize, double[] expectedValues, Map<Integer, Double> normFactors) {
        this.type = type;
        this.unit = unit;
        this.binSize = binSize;
        this.normFactors = normFactors;
        this.expectedValues = expectedValues;
    }

    // This is exposed for testing, should not use directly
    public Map<Integer, Double> getNormFactors() {
        return normFactors;
    }


    /**
     * Expected value vector.  No chromosome normalization
     *
     * @return Genome-wide expected value vector
     */
    @Override
    public double[] getExpectedValues() {
        return expectedValues;
    }

    /**
     * Gets the expected value, distance and coverage normalized, chromosome-length normalized
     *
     * @param chrIdx   Chromosome index
     * @param distance Distance from diagonal in bins
     * @return Expected value, distance and coverage normalized
     */
    @Override
    public double getExpectedValue(int chrIdx, int distance) {

        double normFactor = 1.0;
        if (normFactors != null && normFactors.containsKey(chrIdx)) {
            normFactor = normFactors.get(chrIdx);
        }

        if (distance >= expectedValues.length) {

            return expectedValues[expectedValues.length - 1] / normFactor;
        } else {
            return expectedValues[distance] / normFactor;
        }
    }

    @Override
    public int getLength() {
        return expectedValues.length;
    }

    @Override
    public NormalizationType getNormalizationType() {
        return type;
    }

    @Override
    public String getUnit() {
        return unit;
    }

    @Override
    public int getBinSize() {
        return binSize;
    }

}
