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

import juicebox.HiCGlobals;
import juicebox.data.ContactRecord;

import java.util.Arrays;
import java.util.List;

public class FinalScale {

    private final static float tol = .0005f;
    private final static boolean zerodiag = false;
    private final static boolean removeZerosOnDiag = false;
    private final static float percentLowRowSumExcluded = 0.0001f;
    private final static float dp = percentLowRowSumExcluded / 2;
    private final static float percentZValsToIgnore = 0;//0.0025f;
    private final static float dp1 = 0;//percentZValsToIgnore / 2;
    private final static float tolerance = .0005f;
    private final static int maxIter = 100;
    private final static int totalIterations = 3 * maxIter;
    private final static float minErrorThreshold = .02f;
    private static final float OFFSET = .5f;

    public static double[] scaleToTargetVector(List<List<ContactRecord>> contactRecordsListOfLists, double[] targetVectorInitial) {

        double low, zHigh, zLow;
        int rlind, zlind, zhind;
        float localPercentLowRowSumExcluded = percentLowRowSumExcluded;
        float localPercentZValsToIgnore = percentZValsToIgnore;

        //	find the matrix dimensions
        int k = targetVectorInitial.length;

        double[] current = new double[k];
        double[] row, col;
        double[] rowBackup = new double[k];
        double[] dr = new double[k];
        double[] dc = new double[k];
        double[] r0 = new double[k];
        int[] bad = new int[k];
        int[] bad1 = new int[k];
        double[] one = new double[k];
        double[] s = new double[k];
        double[] zz = new double[k];
        double[] zTargetVector = targetVectorInitial.clone();
        double[] calculatedVectorB = new double[k];
        double[] reportErrorForIteration = new double[totalIterations + 3];
        int[] numItersForAllIterations = new int[totalIterations + 3];

        int l = 0;
        for (int p = 0; p < k; p++) {
            if (Double.isNaN(zTargetVector[p])) continue;
            if (zTargetVector[p] > 0) {
                zz[l++] = zTargetVector[p];
            }
        }
        zz = dealWithSorting(zz, l);

        zlind = (int) Math.max(0, l * localPercentZValsToIgnore + OFFSET);
        zhind = (int) Math.min(l - 1, l * (1.0 - localPercentZValsToIgnore) + OFFSET);
        zLow = zz[zlind];
        zHigh = zz[zhind];

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
            for (List<ContactRecord> contactRecords : contactRecordsListOfLists) {
                for (ContactRecord cr : contactRecords) {
                    int x = cr.getBinX();
                    int y = cr.getBinY();
                    if (x == y) {
                        bad[x] = 0;
                    }
                }
            }
        } else {
            Arrays.fill(bad, 0);
        }

        //	find rows sums
        int[] numNonZero = new int[k];
        Arrays.fill(numNonZero, 0);
        for (List<ContactRecord> contactRecords : contactRecordsListOfLists) {
            for (ContactRecord cr : contactRecords) {
                int x = cr.getBinX();
                int y = cr.getBinY();
                numNonZero[x]++;
                if (x != y) {
                    numNonZero[y]++;
                }
            }
        }


        //	find relevant percentiles
        int n0 = 0;
        for (int p = 0; p < k; p++) {
            if (numNonZero[p] > 0) {
                r0[n0++] = numNonZero[p];
            }
        }
        r0 = dealWithSorting(r0, n0);

        rlind = (int) Math.max(0, n0 * localPercentLowRowSumExcluded + OFFSET);
        low = r0[rlind];


        //	find the "bad" rows and exclude them
        for (int p = 0; p < k; p++) {
            if ((numNonZero[p] < low && zTargetVector[p] > 0) || Double.isNaN(zTargetVector[p])) {
                bad[p] = 1;
                zTargetVector[p] = 1.0;
            }
        }

        row = sparseMultiplyGetRowSums(contactRecordsListOfLists, one, k);
        System.arraycopy(row, 0, rowBackup, 0, k);

        for (int p = 0; p < k; p++) {
            dr[p] = 1 - bad[p];
        }
        System.arraycopy(dr, 0, dc, 0, k);
        System.arraycopy(dr, 0, one, 0, k);

        // treat separately rows for which z[p] = 0
        for (int p = 0; p < k; p++) {
            if (zTargetVector[p] == 0) {
                one[p] = 0;
            }
        }
        for (int p = 0; p < k; p++) {
            bad1[p] = (int) (1 - one[p]);
        }

        System.arraycopy(dr, 0, current, 0, k);
        //	start iterations
        //	row is the current rows sum; dr and dc are the current rows and columns scaling vectors
        double ber = 10.0 * (1.0 + tolerance);
        double err = ber;
        int iter = 0;
        int fail;
        int nerr = 0;
        double[] errors = new double[10000];
        int allItersI = 0;

        // if perc or perc1 reached upper bound or the total number of iterationbs is too high, exit
        while ((ber > tolerance || err > 5.0 * tolerance) && iter < maxIter && allItersI < totalIterations
                && localPercentLowRowSumExcluded <= 0.2 && localPercentZValsToIgnore <= 0.1) {

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
            col = sparseMultiplyGetRowSums(contactRecordsListOfLists, dr, k);
            for (int p = 0; p < k; p++) col[p] *= dc[p];
            for (int p = 0; p < k; p++) if (bad1[p] == 1) col[p] = 1.0;
            for (int p = 0; p < k; p++) s[p] = zTargetVector[p] / col[p];
            for (int p = 0; p < k; p++) dc[p] *= s[p];

            // find row sums and update columns scaling vector
            row = sparseMultiplyGetRowSums(contactRecordsListOfLists, dc, k);
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
                col = sparseMultiplyGetRowSums(contactRecordsListOfLists, calculatedVectorB, k);
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
                    if (reportErrorForIteration[allItersI - p] * (1.0 + minErrorThreshold) < reportErrorForIteration[allItersI - p - 1]) {
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
                    rlind = (int) Math.max(0, n0 * localPercentLowRowSumExcluded + OFFSET);
                    low = r0[rlind];
                    zlind = (int) Math.max(0, l * localPercentZValsToIgnore + OFFSET);
                    zhind = (int) Math.min(l - 1, l * (1.0 - localPercentZValsToIgnore) + OFFSET);
                    zLow = zz[zlind];
                    zHigh = zz[zhind];
                    for (int p = 0; p < k; p++) {
                        if (zTargetVector[p] > 0 && (zTargetVector[p] < zLow || zTargetVector[p] > zHigh)) {
                            zTargetVector[p] = Double.NaN;
                        }
                    }
                    for (int p = 0; p < k; p++) {
                        if ((numNonZero[p] < low && zTargetVector[p] > 0) || Double.isNaN(zTargetVector[p])) {
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
        }

        //	find the final error in row sums
        if (iter % 10 == 0) {
            col = sparseMultiplyGetRowSums(contactRecordsListOfLists, calculatedVectorB, k);
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

        if (HiCGlobals.printVerboseComments) {
            System.out.println(allItersI);
            System.out.println(localPercentLowRowSumExcluded);
            System.out.println(localPercentZValsToIgnore);
            System.out.println(Arrays.toString(reportErrorForIteration));
        }

        return calculatedVectorB;
    }

    private static double[] dealWithSorting(double[] vector, int length) {
        double[] realVector = new double[length];
        System.arraycopy(vector, 0, realVector, 0, length);
        Arrays.sort(realVector);
        return realVector;
    }

    private static double[] sparseMultiplyGetRowSums(List<List<ContactRecord>> contactRecordsListOfLists, double[] vector, int vectorLength) {
        double[] sumVector = new double[vectorLength];

        for (List<ContactRecord> contactRecords : contactRecordsListOfLists) {
            for (ContactRecord cr : contactRecords) {
                int x = cr.getBinX();
                int y = cr.getBinY();
                float counts = cr.getCounts();
                if (x == y) {
                    counts *= .5;
                }

                sumVector[x] += counts * vector[y];
                sumVector[y] += counts * vector[x];
            }
        }

        return sumVector;
    }
}
