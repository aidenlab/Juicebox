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

package juicebox.data;

import juicebox.HiC;
import juicebox.windowui.HiCZoom;
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
    private final HiC.Unit unit;

    private final Map<Integer, Double> normFactors;

    private final double[] expectedValues;

    public ExpectedValueFunctionImpl(NormalizationType type, HiC.Unit unit, int binSize, double[] expectedValues, Map<Integer, Double> normFactors) {
        this.type = type;
        this.unit = unit;
        this.binSize = binSize;
        this.normFactors = normFactors;
        this.expectedValues = expectedValues;
    }

    public static String getKey(HiCZoom zoom, NormalizationType normType) {
        return zoom.getKey() + "_" + normType;
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

        if (expectedValues.length > 0) {
            if (distance >= expectedValues.length) {
                return expectedValues[expectedValues.length - 1] / normFactor;
            } else {
                return expectedValues[distance] / normFactor;
            }
        } else {
            System.err.println("Expected values array is empty");
            return -1;
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
    public HiC.Unit getUnit() {
        return unit;
    }

    @Override
    public int getBinSize() {
        return binSize;
    }

}
