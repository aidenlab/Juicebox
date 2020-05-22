/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab
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

package juicebox.tools.utils.norm.final2;

import juicebox.data.ContactRecord;

import java.util.Arrays;
import java.util.List;

public class FinalScale {

    private final static float tol = .0005f;
    private final static float del = .02f;
    //private final static float perc = .01f;
    private final static float dp = .005f;
    //private final static float perc1 = 0.0025f;
    private final static float dp1 = .001f;
    private final static int maxiter = 100;
    private final static boolean zerodiag = false;
    private final static int totalIterations = 200;
    private final static int threads = 1;
    private final static boolean removeZerosOnDiag = false;

    private static double[] scaleToTargetVector(List<ContactRecord> contactRecords, double[] targetVectorInitial, double tolerance,
                                                double percentLowRowSumExcluded, double percentZValsToIgnore,
                                                int maxIter, double del, int numTrials) {

        double low, zHigh, zLow, ber;
        int lind, hind;
        double localPercentLowRowSumExcluded = percentLowRowSumExcluded;
        double localPercentZValsToIgnore = percentZValsToIgnore;

        //	find the matrix dimensions
        int k = targetVectorInitial.length;

        double[] current = new double[k];
        double[] row = new double[k];
        double[] rowBackup = new double[k];
        double[] col = new double[k];
        int[] dr = new int[k];
        int[] dc = new int[k];
        double[] r0 = new double[k];
        int[] bad = new int[k];
        int[] bad1 = new int[k];
        int[] one = new int[k];
        double[] s = new double[k];
        double[] zz = new double[k];
        double[] zTargetVector = targetVectorInitial.clone();

        int l = 0;
        for (int p = 0; p < k; p++) {
            if (Double.isNaN(zTargetVector[p])) continue;
            if (zTargetVector[p] > 0) {
                zz[l++] = zTargetVector[p];
            }
        }
        Arrays.sort(zz);

        lind = (int) (l * localPercentZValsToIgnore + 0.5);
        hind = (int) (l * (1.0 - localPercentZValsToIgnore) + 0.5);
        if (lind < 0) lind = 0;
        if (hind >= l) hind = l - 1;
        zLow = zz[lind];
        zHigh = zz[hind];
        zz = null;

        for (int p = 0; p < k; p++) {
            if (zTargetVector[p] > 0 && (zTargetVector[p] < zLow || zTargetVector[p] > zHigh)) {
                zTargetVector[p] = Double.NaN;
            }
        }

        Arrays.fill(one, 1);
        for (int p = 0; p < k; p++) {
            if (zTargetVector[p] == 0) {
                one[p] = 0;
            }
        }


        if (removeZerosOnDiag) {
            Arrays.fill(bad, 1);
            for (ContactRecord cr : contactRecords) {
                int x = cr.getBinX();
                int y = cr.getBinY();
                if (x == y) {
                    bad[x] = 0;
                }
            }
        } else {
            Arrays.fill(bad, 0);
        }

        //	find rows sums
        int[] numNonZero = new int[k];
        Arrays.fill(numNonZero, 0);
        for (ContactRecord cr : contactRecords) {
            int x = cr.getBinX();
            int y = cr.getBinY();
            numNonZero[x]++;
            if (x != y) {
                numNonZero[y]++;
            }
        }


        //	find relevant percentiles
        int n0 = 0;
        for (int p = 0; p < k; p++) {
            if (numNonZero[p] > 0) {
                r0[n0++] = numNonZero[p];
            }
        }

        // need to do because sort uses whole array and the zeros at the end will cause a problem
        double[] r02 = new double[n0];
        System.arraycopy(r0, 0, r02, 0, n0);
        Arrays.sort(r02);
        r0 = r02;

        lind = (int) (n0 * localPercentLowRowSumExcluded + .5);
        if (lind < 0) lind = 0;
        low = r0[lind];


        //	find the "bad" rows and exclude them
        for (int p = 0; p < k; p++) {
            if ((numNonZero[p] < low && zTargetVector[p] > 0) || Double.isNaN(zTargetVector[p])) {
                bad[p] = 1;
                zTargetVector[p] = 1.0;
            }
        }

        utmvMul(contactRecords, one, row);
        System.arraycopy(row, 0, rowBackup, 0, k);

        for (int p = 0; p < k; p++) {
            dr[p] = 1 - bad[p];
        }
        System.arraycopy(dr, 0, dc, 0, k);
        System.arraycopy(dr, 0, one, 0, k);


        double[] calculatedVectorB = new double[k];
        double[] reportErrorForIteration = new double[maxIter];
        int[] numItersForAllIterations = new int[maxIter];

        // treat separately rows for which z[p] = 0
        for (int p = 0; p < k; p++) {
            if (zTargetVector[p] == 0) {
                one[p] = 0;
            }
        }
        for (int p = 0; p < k; p++) {
            bad1[p] = 1 - one[p];
        }

        //	start iterations
        //	row is the current rows sum; dr and dc are the current rows and columns scaling vectors
        ber = 10.0 * (1.0 + tolerance);
        double err = ber;
        int iter = 0;

        //int stuck = 0;
        System.arraycopy(dr, 0, current, 0, k);

        int fail;
        int nerr = 0;
        double[] errors = new double[10000];

        int allItersI = 0;

        while ((ber > tolerance || err > 5.0 * tolerance) && iter < maxIter) {
            iter++;
            allItersI++;
            fail = 1;

            for (int p = 0; p < k; p++) {
                if (bad1[p] == 1) row[p] = 1.0;
            }
            for (int p = 0; p < k; p++) {
                s[p] = zTargetVector[p] / row[p];
            }
            for (int p = 0; p < k; p++) {
                dr[p] *= s[p];
            }

            // find column sums and update rows scaling vector
            utmvMul(contactRecords, dr, col);
            for (int p = 0; p < k; p++) col[p] *= dc[p];
            for (int p = 0; p < k; p++) if (bad1[p] == 1) col[p] = 1.0;
            for (int p = 0; p < k; p++) s[p] = zTargetVector[p] / col[p];
            for (int p = 0; p < k; p++) dc[p] *= s[p];

            // find row sums and update columns scaling vector
            utmvMul(contactRecords, dc, row);
            for (int p = 0; p < k; p++) row[p] *= dr[p];

            // calculate current scaling vector
            for (int p = 0; p < k; p++) {
                calculatedVectorB[p] = Math.sqrt(dr[p] * dc[p]);
            }

            //	calculate the current error
            ber = 0;
            for (int p = 0; p < k; p++) {
                if (bad1[p] == 1) continue;
                double tempErr = Math.abs(calculatedVectorB[p] - current[p]);
                if (tempErr > ber) {
                    ber = tempErr;
                }
            }

            reportErrorForIteration[allItersI - 1] = ber;
            numItersForAllIterations[allItersI - 1] = iter;

            //	since calculating the error in row sums requires matrix-vector multiplication we are are doing this every 10
            //	iterations
            if (iter % 10 == 0) {
                utmvMul(contactRecords, calculatedVectorB, col);
                err = 0;
                for (int p = 0; p < k; p++) {
                    if (bad1[p] == 1) continue;
                    double tempErr = Math.abs((col[p] * calculatedVectorB[p] - zTargetVector[p]));
                    if (err < tempErr) {
                        err = tempErr;
                    }
                }
                errors[nerr++] = err;
            }

            System.arraycopy(calculatedVectorB, 0, current, 0, k);

            // check whether convergence rate is satisfactory
            // if less than 5 iterations (so less than 5 errors) and less than 2 row sums errors, there is nothing to check

            if ((ber < tolerance) && (nerr < 2 || (nerr >= 2 && errors[nerr - 1] < 0.5 * errors[nerr - 2]))) continue;

            if (iter > 5) {
                for (int p = 1; p <= 5; p++) {
                    if (reportErrorForIteration[allItersI - p] * (1.0 + del) < reportErrorForIteration[allItersI - p - 1]) {
                        fail = 0;
                    }
                }

                if (nerr >= 2 && errors[nerr - 1] > 0.75 * errors[nerr - 2]) {
                    fail = 1;
                }

                if (iter >= maxIter) {
                    fail = 1;
                }

                if (fail == 1) {
                    localPercentLowRowSumExcluded += dp;
                    localPercentZValsToIgnore += dp1;
                    nerr = 0;
                    lind = (int) (n0 * localPercentLowRowSumExcluded + 0.5);
                    low = r0[lind];
                    lind = (int) (l * localPercentZValsToIgnore + 0.5);
                    hind = (int) (l * (1.0 - localPercentZValsToIgnore) + 0.5);
                    if (lind < 0) lind = 0;
                    if (hind >= l) hind = l - 1;
                    zLow = zz[lind];
                    zHigh = zz[hind];
                    for (int p = 0; p < k; p++) {
                        if (zTargetVector[p] > 0 && (zTargetVector[p] < zLow || zTargetVector[p] > zHigh)) {
                            zTargetVector[p] = Double.NaN;
                        }
                    }
                    for (int p = 0; p < k; p++) {
                        if ((numNonZero[p] <= low && zTargetVector[p] > 0) || Double.isNaN(zTargetVector[p])) {
                            bad[p] = 1;
                            bad1[p] = 1;
                            one[p] = 0;
                            zTargetVector[p] = 1.0;
                        }
                    }


                    ber = 10.0 * (1.0 + tol);
                    err = 10.0 * (1.0 + tol);

                    //	if the current error is larger than 5 iteration ago start from scratch,
                    //	otherwise continue from the current position
                    if (reportErrorForIteration[allItersI - 1] > reportErrorForIteration[allItersI - 6]) {
                        for (int p = 0; p < k; p++) {
                            dr[p] = 1 - bad[p];
                        }
                        System.arraycopy(dr, 0, dc, 0, k);
                        System.arraycopy(dr, 0, one, 0, k);
                        System.arraycopy(dr, 0, current, 0, k);
                        System.arraycopy(rowBackup, 0, row, 0, k);
                    } else {
                        for (int p = 0; p < k; p++) {
                            dr[p] *= (1 - bad[p]);
                        }
                        for (int p = 0; p < k; p++) {
                            dc[p] *= (1 - bad[p]);
                        }
                    }
                    iter = 0;
                }

            }

            // if perc or perc1 reached upper bound or the total number of iterationbs is too high, exit
            if (localPercentLowRowSumExcluded > 0.2 || localPercentZValsToIgnore > 0.1) break;
            if (allItersI > totalIterations) break;
        }

        //	find the final error in row sums
        if (iter % 10 == 0) {
            utmvMul(contactRecords, calculatedVectorB, col);
            err = 0;
            for (int p = 0; p < k; p++) {
                if (bad1[p] == 1) continue;
                double tempErr = Math.abs(col[p] * calculatedVectorB[p] - zTargetVector[p]);
                if (err < tempErr)
                    err = tempErr;
            }
        }

        reportErrorForIteration[allItersI + 1] = ber;
        reportErrorForIteration[allItersI + 2] = err;

        for (int p = 0; p < k; p++) {
            if (bad[p] == 1) {
                calculatedVectorB[p] = Double.NaN;
            }
        }

        return calculatedVectorB;
    }


    private static void utmvMul(List<ContactRecord> contactRecords, int[] binaryVector, double[] tobeScaledVector) {
        Arrays.fill(tobeScaledVector, 0);

        for (ContactRecord cr : contactRecords) {
            int x = cr.getBinX();
            int y = cr.getBinY();
            float counts = cr.getCounts();

            tobeScaledVector[x] += counts * binaryVector[y];
            tobeScaledVector[y] += counts * binaryVector[x];
        }
    }

    private static void utmvMul(List<ContactRecord> contactRecords, double[] vector, double[] tobeScaledVector) {
        Arrays.fill(tobeScaledVector, 0);

        for (ContactRecord cr : contactRecords) {
            int x = cr.getBinX();
            int y = cr.getBinY();
            float counts = cr.getCounts();

            tobeScaledVector[x] += counts * vector[y];
            tobeScaledVector[y] += counts * vector[x];
        }
    }
}
