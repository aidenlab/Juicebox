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

package juicebox.tools.utils.common;

import java.util.Arrays;

/**
 * TODO merge these common helper classes
 */
public class StatPercentile {

    private final double[] statsData;

    public StatPercentile(double[] data) {
        statsData = new double[data.length];
        System.arraycopy(data, 0, statsData, 0, data.length);
        Arrays.sort(statsData);
    }

    // TODO actually could be much more optimized since same vals are queried
    public double evaluate(double val) {
        return internalEvaluate(val) * 100;
    }

    /**
     * @param val
     * @return percentile of given value as ranked relative to values in internal array
     */
    private double internalEvaluate(double val) {
        for (int i = 0; i < statsData.length; i++) {
            if (statsData[i] >= val) {
                if (statsData[i] > val) {
                    return Math.max(0.0, i / statsData.length);
                } else {
                    double percentile = 0;
                    int num = 0;
                    for (int j = i; j < statsData.length; j++) {
                        if (statsData[j] > val) {
                            break;
                        }
                        percentile += ((double) i) / statsData.length;
                        num++;
                    }
                    return percentile / num;
                }
            }
        }
        return 1.0;
    }
}
