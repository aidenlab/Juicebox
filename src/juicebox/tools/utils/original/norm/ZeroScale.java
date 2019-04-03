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

package juicebox.tools.utils.original.norm;

import juicebox.HiCGlobals;
import juicebox.data.ContactRecord;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ZeroScale {

    private final static double tolerance = 1.0e-3;
    private final static int maxIter = 200;
    private final static double del = 1.0e-2;
    private final static int maxOverallAttempts = 3;
    private final static int numTrialsWithinScalingRun = 5;

    public static double[] scale(List<ContactRecord> contactRecords, double[] targetVectorInitial, String key) {
        // if the regular call fails, loosen parameters
        double[] newVector = launchScalingWithDiffTolerances(contactRecords, targetVectorInitial, .01, 0, key);
        if (newVector == null) {
            newVector = launchScalingWithDiffTolerances(contactRecords, targetVectorInitial, .04, .01, key);
        }
        return newVector;
    }

    public static double[] launchScalingWithDiffTolerances(List<ContactRecord> contactRecords, double[] targetVectorInitial, double percentLowRowSumExcludedInitial,
                                                           double percentZValsToIgnoreInitial, String key) {

        double percentLowRowSumExcluded = percentLowRowSumExcludedInitial;
        double percentZValsToIgnore = percentZValsToIgnoreInitial;
        double[] newVector = scaleToTargetVector(contactRecords, targetVectorInitial, tolerance, percentLowRowSumExcluded, percentZValsToIgnore, maxIter, del, numTrialsWithinScalingRun);

        int count = 0;
        while (newVector == null && count++ < maxOverallAttempts) {

            percentLowRowSumExcluded = 1.5 * percentLowRowSumExcluded;
            percentZValsToIgnore = 1.5 * percentZValsToIgnore;

            if (HiCGlobals.printVerboseComments) {
                System.err.println("Did not converge for " + key);
                System.err.println("new percentLowRowSumExcluded = " + percentLowRowSumExcluded + " and new percentZValsToIgnore = " + percentZValsToIgnore);
            }
            newVector = scaleToTargetVector(contactRecords, targetVectorInitial, tolerance, percentLowRowSumExcluded, percentZValsToIgnore, maxIter, del, numTrialsWithinScalingRun);

        }

        if (newVector == null && HiCGlobals.printVerboseComments) {
            System.err.println("Scaling result still null for " + key + "; vector did not converge");
        }
        return newVector;
    }

    private static double[] scaleToTargetVector(List<ContactRecord> contactRecords, double[] targetVectorInitial, double tolerance,
                                                double percentLowRowSumExcluded, double percentZValsToIgnore,
                                                int maxIter, double del, int numTrials) {

        double high, low, err;
        int lind, hind;

        //	find the matrix dimensions
        int k = targetVectorInitial.length;
        //Math.min(zd.getXGridAxis().getBinCount() + 1, targetVectorInitial.length);

        double[] current = new double[k];
        double[] r = new double[k];
        double[] r0 = new double[k];
        int[] bad = new int[k];
        int[] bad1 = new int[k];
        double[] one = new double[k];
        double[] s = new double[k];
        double[] zz = new double[k];
        double[] targetVector = targetVectorInitial.clone();

        int l = 0;
        for (int p = 0; p < k; p++) {
            if (Double.isNaN(targetVector[p])) continue;
            if (targetVector[p] > 0) zz[l++] = targetVector[p];
        }
        Arrays.sort(zz);

        lind = (int) (((double) l) * percentZValsToIgnore + 0.5);
        hind = (int) (((double) l) * (1.0 - percentZValsToIgnore) + 0.5);
        if (lind < 0) lind = 0;
        if (hind >= l) hind = l - 1;
        low = zz[lind];
        high = zz[hind];
        zz = null;

        for (int p = 0; p < k; p++)
            if (targetVector[p] > 0 && (targetVector[p] < low || targetVector[p] > high)) targetVector[p] = Double.NaN;

        for (int p = 0; p < k; p++) one[p] = 1.0;
        for (int p = 0; p < k; p++) if (targetVector[p] == 0) one[p] = 0;

        //	find rows sums
        for (int p = 0; p < k; p++) {
            r[p] = 0;
        }

        for (ContactRecord cr : contactRecords) {
            int x = cr.getBinX();
            int y = cr.getBinY();
            final float counts = cr.getCounts();

            r[x] += counts * one[y];
            if (x < y) {
                r[y] += counts * one[x];
            }
        }

        //	find relevant percentiles
        System.arraycopy(r, 0, r0, 0, k);

        Arrays.sort(r0);

        int n = 0;
        for (int p = 0; p < k; p++) if (r0[p] == 0) n++;
        lind = n - 1 + (int) (((double) (k - n)) * percentLowRowSumExcluded + 0.5);
        hind = n - 1 + (int) (((double) (k - n)) * (1.0 - 0.01 * percentLowRowSumExcluded) + 0.5);

        if (lind < 0) lind = 0;
        if (hind >= k) hind = k - 1;
        low = r0[lind];
        high = r0[hind]; //todo ask moshe/neva if bug
        r0 = null;

        //	find the "bad" rows and exclude them
        for (int p = 0; p < k; p++) {
            if ((r[p] < low && targetVector[p] > 0) || Double.isNaN(targetVector[p])) {
                bad[p] = 1;
                targetVector[p] = 1.0;
            } else bad[p] = 0;
        }

        double[] calculatedVector = new double[k];
        double[][] errorForIteration = new double[maxIter][2];

        for (int p = 0; p < k; p++) {
            calculatedVector[p] = 1.0 - bad[p];
        }
        for (int p = 0; p < k; p++) {
            one[p] = 1.0 - bad[p];
        }
        for (int p = 0; p < k; p++) {
            if (targetVector[p] == 0) {
                one[p] = 0;
            }
        }
        for (int p = 0; p < k; p++) {
            bad1[p] = (int) Math.round(1 - one[p]);
        }

        //	start iterations
        //	r is the current rows sum; s is the correction vector to be applied to rows and columns
        err = 10.0 * (1.0 + tolerance);
        int iter = 0;
        int stuck = 0;
        double ber;
        System.arraycopy(calculatedVector, 0, current, 0, k);
        while (err > tolerance && iter++ < maxIter) {
            for (int p = 0; p < k; p++) {
                if (bad1[p] == 1) r[p] = 1.0;
            }
            for (int p = 0; p < k; p++) {
                s[p] = Math.sqrt(targetVector[p] / r[p]);
            }
            for (int p = 0; p < k; p++) {
                calculatedVector[p] *= s[p];
            }

            for (int p = 0; p < k; p++) r[p] = 0;

            for (ContactRecord cr : contactRecords) {
                int x = cr.getBinX();
                int y = cr.getBinY();
                final float counts = cr.getCounts();

                r[x] += counts * calculatedVector[y];
                if (x < y) {
                    r[y] += counts * calculatedVector[x];
                }
            }

            for (int p = 0; p < k; p++) {
                r[p] *= calculatedVector[p];
            }

            //	calculate the current relative error
            err = 0;
            ber = 0;
            for (int p = 0; p < k; p++) {
                if (bad1[p] == 1) continue;
                if (Math.abs((r[p] - targetVector[p]) / targetVector[p]) > err)
                    err = Math.abs((r[p] - targetVector[p]) / targetVector[p]);
                if (Math.abs(calculatedVector[p] - current[p]) > ber) ber = Math.abs(calculatedVector[p] - current[p]);
            }
            errorForIteration[iter - 1][0] = err;
            errorForIteration[iter - 1][1] = ber;

            System.arraycopy(calculatedVector, 0, current, 0, k);
            if (iter < numTrials + 2) continue;
            if (err > (1.0 - del) * errorForIteration[iter - 2][0]) stuck++;
            else stuck = 0;
            if (stuck >= numTrials) break;
        }
        for (int p = 0; p < k; p++) {
            if (bad[p] == 1) {
                calculatedVector[p] = Double.NaN;
            }
        }

        if (err > tolerance){
            return null;
        }
        return calculatedVector;
    }

    public static double[] normalizeVectorByScaleFactor(double[] newNormVector, List<ContactRecord> contactRecords) {

        for (int k = 0; k < newNormVector.length; k++) {
            if (newNormVector[k] <= 0 || Double.isNaN(newNormVector[k])) {
                newNormVector[k] = Double.NaN;
            } else {
                newNormVector[k] = 1 / newNormVector[k];
            }
        }

        double normalizedSumTotal = 0, sumTotal = 0;

        for (ContactRecord cr : contactRecords) {
            int x = cr.getBinX();
            int y = cr.getBinY();
            final float counts = cr.getCounts();

            if (!Double.isNaN(newNormVector[x]) && !Double.isNaN(newNormVector[y])) {
                double normalizedValue = counts / (newNormVector[x] * newNormVector[y]);
                normalizedSumTotal += normalizedValue;
                sumTotal += counts;
                if (x != y) {
                    normalizedSumTotal += normalizedValue;
                    sumTotal += counts;
                }
            }
        }

        double scaleFactor = Math.sqrt(normalizedSumTotal / sumTotal);

        for (int k = 0; k < newNormVector.length; k++) {
            if (!Double.isNaN(newNormVector[k])) {
                newNormVector[k] = scaleFactor * newNormVector[k];
            }
        }
        return newNormVector;
    }

    public static double[] mmbaScaleToVector(ArrayList<ContactRecord> contactRecords, double[] tempTargetVector) {

        double[] newNormVector = scale(contactRecords, tempTargetVector, "mmsa_scale");
        if (newNormVector != null) {
            newNormVector = normalizeVectorByScaleFactor(newNormVector, contactRecords);
        }

        return newNormVector;
    }
}
