/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2021 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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
 *  FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.tools.utils.norm.final2;

import juicebox.HiCGlobals;
import juicebox.data.ContactRecord;
import juicebox.data.basics.ListOfFloatArrays;
import juicebox.data.basics.ListOfIntArrays;
import juicebox.data.iterator.IteratorContainer;

import java.util.Arrays;
import java.util.Iterator;

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

    public static ListOfFloatArrays scaleToTargetVector(IteratorContainer ic, ListOfFloatArrays targetVectorInitial) {

        double low, zHigh, zLow;
        int rlind, zlind, zhind;
        float localPercentLowRowSumExcluded = percentLowRowSumExcluded;
        float localPercentZValsToIgnore = percentZValsToIgnore;

        //	find the matrix dimensions
        long k = targetVectorInitial.getLength();

        ListOfFloatArrays current = new ListOfFloatArrays(k);
        ListOfFloatArrays row, col;
        ListOfFloatArrays rowBackup = new ListOfFloatArrays(k);
        ListOfFloatArrays dr = new ListOfFloatArrays(k);
        ListOfFloatArrays dc = new ListOfFloatArrays(k);
        ListOfIntArrays bad;
        ListOfIntArrays bad1 = new ListOfIntArrays(k);
        ListOfFloatArrays s = new ListOfFloatArrays(k);
        double[] zz = new double[(int) Math.min(k, Integer.MAX_VALUE - 1)];
        double[] r0 = new double[(int) Math.min(k, Integer.MAX_VALUE - 1)];
        
        ListOfFloatArrays zTargetVector = targetVectorInitial.deepClone();
        ListOfFloatArrays calculatedVectorB = new ListOfFloatArrays(k);
        ListOfFloatArrays one = new ListOfFloatArrays(k, 1);
        ListOfIntArrays numNonZero = new ListOfIntArrays(k, 0);
        
        double[] reportErrorForIteration = new double[totalIterations + 3];
        int[] numItersForAllIterations = new int[totalIterations + 3];
        
        int l = 0;
        for (long p = 0; p < k; p++) {
            if (Float.isNaN(zTargetVector.get(p))) continue;
            if (zTargetVector.get(p) > 0) {
                zz[l++] = zTargetVector.get(p);
            }
        }
        zz = dealWithSorting(zz, l);
        
        // unlikey to exceed max int for lind; hind
        // for now we will only sort one vector and hope that suffices
        zlind = (int) Math.max(0, l * localPercentZValsToIgnore + OFFSET);
        zhind = (int) Math.min(l - 1, l * (1.0 - localPercentZValsToIgnore) + OFFSET);
        zLow = zz[zlind];
        zHigh = zz[zhind];
        
        for (long p = 0; p < k; p++) {
            double valZ = zTargetVector.get(p);
            if (valZ > 0 && (valZ < zLow || valZ > zHigh)) {
                zTargetVector.set(p, Float.NaN);
            }
        }
        
        
        for (long p = 0; p < k; p++) {
            if (zTargetVector.get(p) == 0) {
                one.set(p, 0);
            }
        }
        
        
        if (removeZerosOnDiag) {
            bad = new ListOfIntArrays(k, 1);
            setBadValues(bad, ic);
        } else {
            bad = new ListOfIntArrays(k, 0);
        }

        //	find rows sums
        setRowSums(numNonZero, ic);
        
        
        //	find relevant percentiles
        int n0 = 0;
        for (long p = 0; p < k; p++) {
            int valP = numNonZero.get(p);
            if (valP > 0) {
                r0[n0++] = valP;
            }
        }
        r0 = dealWithSorting(r0, n0);
        
        rlind = (int) Math.max(0, n0 * localPercentLowRowSumExcluded + OFFSET);
        low = r0[rlind];
        
        
        //	find the "bad" rows and exclude them
        for (long p = 0; p < k; p++) {
            if ((numNonZero.get(p) < low && zTargetVector.get(p) > 0) || Float.isNaN(zTargetVector.get(p))) {
                bad.set(p, 1);
                zTargetVector.set(p, 1.0f);
            }
        }

        row = sparseMultiplyGetRowSums(ic, one, k);
        rowBackup = row.deepClone();
        
        for (long p = 0; p < k; p++) {
            dr.set(p, 1 - bad.get(p));
        }
        dc = dr.deepClone();
        one = dr.deepClone();
        
        // treat separately rows for which z[p] = 0
        for (long p = 0; p < k; p++) {
            if (zTargetVector.get(p) == 0) {
                one.set(p, 0);
            }
        }
        for (long p = 0; p < k; p++) {
            bad1.set(p, (int) (1 - one.get(p)));
        }
        
        current = dr.deepClone();
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
                if (bad1.get(p) == 1) row.set(p, 1.0f);
            }
            for (int p = 0; p < k; p++) {
                s.set(p, zTargetVector.get(p) / row.get(p));
            }
            for (long p = 0; p < k; p++) {
                dr.multiplyBy(p, s.get(p));
            }
    
            // find column sums and update rows scaling vector
            col = sparseMultiplyGetRowSums(ic, dr, k);
            for (long p = 0; p < k; p++) col.multiplyBy(p, dc.get(p));
            for (long p = 0; p < k; p++) if (bad1.get(p) == 1) col.set(p, 1.0f);
            for (long p = 0; p < k; p++) s.set(p, zTargetVector.get(p) / col.get(p));
            for (long p = 0; p < k; p++) dc.multiplyBy(p, s.get(p));
    
            // find row sums and update columns scaling vector
            row = sparseMultiplyGetRowSums(ic, dc, k);
            for (long p = 0; p < k; p++) row.multiplyBy(p, dr.get(p));
    
            // calculate current scaling vector
            for (long p = 0; p < k; p++) {
                calculatedVectorB.set(p, (float) Math.sqrt(dr.get(p) * dc.get(p)));
            }
    
            //	calculate the current error
            ber = 0;
            for (long p = 0; p < k; p++) {
                if (bad1.get(p) == 1) continue;
                double tempErr = Math.abs(calculatedVectorB.get(p) - current.get(p));
                if (tempErr > ber) {
                    ber = tempErr;
                }
            }
    
            reportErrorForIteration[allItersI - 1] = ber;
            numItersForAllIterations[allItersI - 1] = iter;
    
            //	since calculating the error in row sums requires matrix-vector multiplication we are are doing this every 10
            //	iterations
            if (iter % 10 == 0) {
                col = sparseMultiplyGetRowSums(ic, calculatedVectorB, k);
                err = 0;
                for (long p = 0; p < k; p++) {
                    if (bad1.get(p) == 1) continue;
                    double tempErr = Math.abs((col.get(p) * calculatedVectorB.get(p) - zTargetVector.get(p)));
                    if (err < tempErr) {
                        err = tempErr;
                    }
                }
                errors[nerr++] = err;
            }
    
            current = calculatedVectorB.deepClone();

            // check whether convergence rate is satisfactory
            // if less than 5 iterations (so less than 5 errors) and less than 2 row sums errors, there is nothing to check

            if ((ber < tolerance) && (nerr < 2 || (nerr >= 2 && errors[nerr - 1] < 0.5 * errors[nerr - 2]))) continue;

            if (iter > 5) {
                for (int q = 1; q <= 5; q++) {
                    if (reportErrorForIteration[allItersI - q] * (1.0 + minErrorThreshold) < reportErrorForIteration[allItersI - q - 1]) {
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
                    for (long p = 0; p < k; p++) {
                        if (zTargetVector.get(p) > 0 && (zTargetVector.get(p) < zLow || zTargetVector.get(p) > zHigh)) {
                            zTargetVector.set(p, Float.NaN);
                        }
                    }
                    for (long p = 0; p < k; p++) {
                        if ((numNonZero.get(p) < low && zTargetVector.get(p) > 0) || Float.isNaN(zTargetVector.get(p))) {
                            bad.set(p, 1);
                            bad1.set(p, 1);
                            one.set(p, 0);
                            zTargetVector.set(p, 1.0f);
                        }
                    }
    
    
                    ber = 10.0 * (1.0 + tol);
                    err = 10.0 * (1.0 + tol);
    
                    //	if the current error is larger than 5 iteration ago start from scratch,
                    //	otherwise continue from the current position
                    if (reportErrorForIteration[allItersI - 1] > reportErrorForIteration[allItersI - 6]) {
                        for (long p = 0; p < k; p++) {
                            dr.set(p, 1 - bad.get(p));
                        }
                        dc = dr.deepClone();
                        one = dr.deepClone();
                        current = dr.deepClone();
                        row = rowBackup.deepClone();
                    } else {
                        for (long p = 0; p < k; p++) {
                            dr.multiplyBy(p, (1 - bad.get(p)));
                        }
                        for (long p = 0; p < k; p++) {
                            dc.multiplyBy(p, (1 - bad.get(p)));
                        }
                    }
                    iter = 0;
                }
            }
        }

        //	find the final error in row sums
        if (iter % 10 == 0) {
            col = sparseMultiplyGetRowSums(ic, calculatedVectorB, k);
            err = 0;
            for (int p = 0; p < k; p++) {
                if (bad1.get(p) == 1) continue;
                double tempErr = Math.abs(col.get(p) * calculatedVectorB.get(p) - zTargetVector.get(p));
                if (err < tempErr)
                    err = tempErr;
            }
        }
        
        reportErrorForIteration[allItersI + 1] = ber;
        reportErrorForIteration[allItersI + 2] = err;
        
        for (long p = 0; p < k; p++) {
            if (bad.get(p) == 1) {
                calculatedVectorB.set(p, Float.NaN);
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

    private static void setRowSums(ListOfIntArrays numNonZero, IteratorContainer ic) {
        Iterator<ContactRecord> iterator = ic.getNewContactRecordIterator();
        while (iterator.hasNext()) {
            ContactRecord cr = iterator.next();
            int x = cr.getBinX();
            int y = cr.getBinY();
            numNonZero.addTo(x, 1);
            if (x != y) {
                numNonZero.addTo(y, 1);
            }
        }
    }

    private static void setBadValues(ListOfIntArrays bad, IteratorContainer ic) {
        Iterator<ContactRecord> iterator = ic.getNewContactRecordIterator();
        while (iterator.hasNext()) {
            ContactRecord cr = iterator.next();
            int x = cr.getBinX();
            int y = cr.getBinY();
            if (x == y) {
                bad.set(x, 0);
            }
        }
    }

    private static double[] dealWithSorting(double[] vector, int length) {
        double[] realVector = new double[length];
        System.arraycopy(vector, 0, realVector, 0, length);
        Arrays.sort(realVector);
        return realVector;
    }

    private static ListOfFloatArrays sparseMultiplyGetRowSums(IteratorContainer ic,
                                                              ListOfFloatArrays vector, long vectorLength) {
        return ic.sparseMultiply(vector, vectorLength);
    }
}
