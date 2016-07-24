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

package juicebox.data;

import juicebox.HiC;
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
    public HiC.Unit getUnit() {
        return densityFunctions.get(0).getUnit();
    }

    @Override
    public int getBinSize() {
        return densityFunctions.get(0).getBinSize();
    }

}
