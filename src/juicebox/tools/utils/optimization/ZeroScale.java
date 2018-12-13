/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2018 Broad Institute, Aiden Lab
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

package juicebox.tools.utils.optimization;

import java.util.Arrays;

public class ZeroScale {

    /********************************************************************************************************************
     *
     *	This function allows more that 2^31 - 1 nonzero entries. It acceptc a list of c arrays where array i contains m[i] elements
     *
     *	numArrays is the number of arrays
     *	numElementsForArrays is array containing the number of elements of the c arrays
     *	i and j are lists of c 0-based arrays each containing the row and column indices of the nonzero bins
     *	x is a list of c arrays containing the nonzero matrix entries
     *
     *	indxI, indxJ, and matrixVals define the upper triangle of the (squarre symmetric) matrix
     *
     *	targetVector is the "target" vector, i.e. we want rows (and columns) sums to be equal to z
     *	on exit b will hold the scaling vector, i.e. by multiplying the rows and columns of the original matrix
     by b we get the scaled matrix;
     *	on exit report contains the relative error after each iteration
     *
     *	below are arguments having default values
     *
     *	verb indicates whether the function needs to output the progress; 1 means report, 0 means run silent
     *	tolerance is the desired relative error
     *	perc is the percentage of low rows sums to be excluded (i.e. 0.01, 0.02, etc.)
     *	perc1 is the percentage of low and high values of z to ignore
     *	maxiter is the maximum number of iterations allowed
     *	del and trial are for determining that the convergence is too slow and early termination (before maxiter iteration): if
     the relative error decreased by less than del for trials consecuitive iterations the call is terminated and -iter is
     returned (where iter is the number of iterations); calling function can check the return value and know whether convergence
     was reached
     *
     *	Note that making any optional argument negative causes the default value to be used
     *
     *
     *
     * defaults
     * verb = 0
     * tolerance = 1e-3
     * perc 1e-2
     * perc1 0.25e-2
     * maxiter =200
     * del=1.0e-2
     * trials=5
     *
     *
     ***********************************************************************************************************************/

    private static boolean verbose = false;
    private static double tolerance = 1.0e-3;
    private static double percentLowRowSumExcluded = 1.0e-2;
    private static double percentZValsToIgnore = 0.25e-2;
    private static int maxIter = 200;
    private static double del = 1.0e-2;
    private static int trials = 5;


    public static double[] scale(int numArrays, int[] numElementsForArrays, int[][] indxI, int[][] indxJ, double[][] matrixVals, double[] targetVector) {

        double high, low, err;
        int lind, hind;

        //	find the matrix dimensions
        int k = 0;
        for (int ic = 0; ic < numArrays; ic++) {
            for (int p = 0; p < numElementsForArrays[ic]; p++) {
                if (indxJ[ic][p] > k) k = indxJ[ic][p];
            }
        }
        k++;

        double[] current = new double[k];
        double[] r = new double[k];
        double[] r0 = new double[k];
        int[] bad = new int[k];
        int[] bad1 = new int[k];
        double[] one = new double[k];
        double[] s = new double[k];
        double[] zz = new double[k];

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
        for (int ic = 0; ic < numArrays; ic++) {
            for (int p = 0; p < numElementsForArrays[ic]; p++) {
                r[indxI[ic][p]] += matrixVals[ic][p] * one[indxJ[ic][p]];
                if (indxI[ic][p] < indxJ[ic][p]) {
                    r[indxJ[ic][p]] += matrixVals[ic][p] * one[indxI[ic][p]];
                }
            }
        }

        //	find relevant percentiles
        for (int p = 0; p < k; p++) r0[p] = r[p];

        Arrays.sort(r0);

        int n = 0;
        for (int p = 0; p < k; p++) if (r0[p] == 0) n++;
        lind = n - 1 + (int) (((double) (k - n)) * percentLowRowSumExcluded + 0.5);
        hind = n - 1 + (int) (((double) (k - n)) * (1.0 - 0.01 * percentLowRowSumExcluded) + 0.5);

        if (lind < 0) lind = 0;
        if (hind >= k) hind = k - 1;
        low = r0[lind];
        high = r0[hind];
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
        for (int p = 0; p < k; p++) current[p] = calculatedVector[p];
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
            for (int ic = 0; ic < numArrays; ic++) {
                for (int p = 0; p < numElementsForArrays[ic]; p++) {
                    r[indxI[ic][p]] += matrixVals[ic][p] * calculatedVector[indxJ[ic][p]];
                    if (indxI[ic][p] < indxJ[ic][p]) {
                        r[indxJ[ic][p]] += matrixVals[ic][p] * calculatedVector[indxI[ic][p]];
                    }
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
            if (verbose) System.out.printf("%d: %30.15lf %30.15lf\n", iter, err, ber);
            for (int p = 0; p < k; p++) {
                current[p] = calculatedVector[p];
            }
            if (iter < trials + 2) continue;
            if (err > (1.0 - del) * errorForIteration[iter - 2][0]) stuck++;
            else stuck = 0;
            if (stuck >= trials) break;
        }
        for (int p = 0; p < k; p++) {
            if (bad[p] == 1) {
                calculatedVector[p] = Double.NaN;
            }
        }

        /*
        if (err > tolerance){
            return new ScaledVectorData(calculatedVector, errorForIteration, -iter);
        }
        else{
            return new ScaledVectorData(calculatedVector, errorForIteration, iter);
        }
        */
        return calculatedVector;
    }


    static class ScaledVectorData {
        double[] b;
        double[][] report;
        int iter;

        public ScaledVectorData(double[] b, double[][] report, int iter) {
            this.b = b;
            this.report = report;
            this.iter = iter;
        }
    }
}
