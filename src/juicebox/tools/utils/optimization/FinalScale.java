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

public class FinalScale {

    public static int scale0(int c, int[] m, int[][] i, int[][] j, double[][] x, double[] z, double[] b, double[] report,
                             boolean verbose, double tolerance, double percent, double percent2, int maxIter, double del, int numTrials) {
        if (tolerance < 0) tolerance = 1.0e-3;
        if (percent < 0) percent = 1.0e-2;
        if (percent2 < 0) percent = 0.25e-2;
        if (maxIter < 0) maxIter = 200;
        if (del < 0) del = 1.0e-2;
        if (numTrials < 0) numTrials = 5;

        int k, n;
        double high, low, err;
        int lind, hind;

        //	find the matrix dimensions
        k = 0;
        for (int ic = 0; ic < c; ic++) {
            for (int p = 0; p < m[ic]; p++) {
                if (j[ic][p] > k) k = j[ic][p];
            }
        }
        k++;

        double[] r = new double[k];
        double[] r0 = new double[k];
        double[] s = new double[k];
        double[] one = new double[k];
        int[] bad = new int[k];
        double[] zz = new double[k];

        int l = 0;
        for (int p = 0; p < k; p++) {
            if (!Double.isNaN(z[p])) {
                zz[l++] = z[p];
            }
        }

        Arrays.sort(zz);

        lind = (int) (((double) l) * percent2 + 0.5);
        hind = (int) (((double) l) * (1.0 - percent2) + 0.5);
        if (lind < 0) lind = 0;
        if (hind >= l) hind = l - 1;

        low = zz[lind];
        high = zz[hind];
        zz = null;

        for (int p = 0; p < k; p++) {
            if (z[p] < low || z[p] > high) {
                z[p] = Double.NaN;
            }
        }


        //	find rows sums
        for (int p = 0; p < k; p++) {
            r[p] = 0;
        }

        for (int ic = 0; ic < c; ic++) {
            for (int p = 0; p < m[ic]; p++) {
                r[i[ic][p]] += x[ic][p];
                if (i[ic][p] < j[ic][p]) {
                    r[j[ic][p]] += x[ic][p];
                }
            }
        }

        //	find relevant percentiles
        for (int p = 0; p < k; p++) {
            r0[p] = r[p];
        }

        Arrays.sort(r0);

        n = 0;
        for (int p = 0; p < k; p++) {
            if (r0[p] == 0) {
                n++;
            }
        }
        lind = n - 1 + (int) (((double) (k - n)) * percent + 0.5);
        hind = n - 1 + (int) (((double) (k - n)) * (1.0 - 0.01 * percent) + 0.5);

        if (lind < 0) lind = 0;
        if (hind >= k) hind = k - 1;
        low = r0[lind];
        high = r0[hind];
        r0 = null;

        //	find the "bad" rows and exclude them
        for (int p = 0; p < k; p++) {
            //		if (r[p] < low || r[p] > high || isnan(z[p])) {
            if (r[p] < low || Double.isNaN(z[p])) {
                bad[p] = 1;
                z[p] = 1.0;
            } else bad[p] = 0;
        }


        for (int p = 0; p < k; p++) {
            b[p] = 1.0 - bad[p];
        }
        for (int p = 0; p < k; p++) {
            one[p] = 1.0 - bad[p];
        }


        //	start iterations
        //	r is the current rows sum; s is the correction vector to be applied to rows and columns
        err = 10.0 * (1.0 + tolerance);
        int iter = 0;
        int stuck = 0;
        while (err > tolerance && iter++ < maxIter) {
            for (int p = 0; p < k; p++) {
                if (bad[p] == 1) r[p] = 1.0;
            }
            for (int p = 0; p < k; p++) {
                s[p] = Math.sqrt(z[p] / r[p]);
            }
            for (int p = 0; p < k; p++) {
                b[p] *= s[p];
            }

            for (int p = 0; p < k; p++) {
                r[p] = 0;
            }
            for (int ic = 0; ic < c; ic++) {
                for (int p = 0; p < m[ic]; p++) {
                    r[i[ic][p]] += x[ic][p] * b[j[ic][p]];
                    if (i[ic][p] < j[ic][p]) {
                        r[j[ic][p]] += x[ic][p] * b[i[ic][p]];
                    }
                }
            }
            for (int p = 0; p < k; p++) {
                r[p] *= b[p];
            }

            //	calculate the current relative error
            err = 0;
            for (int p = 0; p < k; p++) {
                if (bad[p] == 1) continue;
                if (Math.abs((r[p] - z[p]) / z[p]) > err) {
                    err = Math.abs((r[p] - z[p]) / z[p]);
                }
            }
            report[iter - 1] = err;
            if (verbose) {
                System.out.printf("%d: %30.15lf\n", iter, err);
            }
            if (iter < numTrials + 2) {
                continue;
            }
            if (err > (1.0 - del) * report[iter - 2]) {
                stuck++;
            } else stuck = 0;
            if (stuck >= numTrials) {
                break;
            }
        }
        for (int p = 0; p < k; p++) {
            if (bad[p] == 1) {
                b[p] = Double.NaN;
            }
        }

        if (err > tolerance) {
            return (-iter);
        } else return (iter);
    }

    /********************************************************************************************************************
     *
     *	This function allows more that 2^31 - 1 nonzero entries. It acceptc a list of c arrays where array i contains m[i] elements
     *
     *	c is the number of arrays
     *	m is array containing the number of elements of the c arrays
     *	i and j are lists of c 0-based arrays each containing the row and column indices of the nonzero bins
     *	x is a list of c arrays containing the nonzero matrix entries
     *
     *	i, j, and x define the upper triangle of the (squarre symmetric) matrix
     *
     *	z is the "target" vector, i.e. we want rows (and columns) sums to be equal to z
     *	on exit b will hold the scaling vector, i.e. by multiplying the rows and columns of the original matrix
     by b we get the scaled matrix;
     *	on exit report contains the relative error after each iteration
     *
     *	below are arguments having default values
     *
     *	verb indicates whether the function needs to output the progress; 1 means report, 0 means run silent
     *	tol is the desired relative error
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
     * tol = 1e-3
     * perc 1e-2
     * perc1 0.25e-2
     * maxiter =200
     * del=1.0e-2
     * trials=5
     *
     *
     ***********************************************************************************************************************/


    int scale(int numArrays, int[] numElementsForArrays, int[][] i, int[][] j, double[][] x, double[] z, double[] b, double[][] report,
              boolean verbose, double tol, double percentLowRowSumExcluded, double percentZValsToIgnore, int maxIter, double del, int trials) {
        if (tol < 0) tol = 1.0e-3;
        if (percentLowRowSumExcluded < 0) percentLowRowSumExcluded = 1.0e-2;
        if (percentZValsToIgnore < 0) percentLowRowSumExcluded = 0.25e-2;
        if (maxIter < 0) maxIter = 200;
        if (del < 0) del = 1.0e-2;
        if (trials < 0) trials = 5;


        double high, low, err;
        //	will be allocated so need to be freed
        int lind, hind;

        //	find the matrix dimensions
        int k = 0;
        for (int ic = 0; ic < numArrays; ic++) {
            for (int p = 0; p < numElementsForArrays[ic]; p++) {
                if (j[ic][p] > k) k = j[ic][p];
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
            if (Double.isNaN(z[p])) continue;
            if (z[p] > 0) zz[l++] = z[p];
        }
        Arrays.sort(zz);

        lind = (int) (((double) l) * percentZValsToIgnore + 0.5);
        hind = (int) (((double) l) * (1.0 - percentZValsToIgnore) + 0.5);
        if (lind < 0) lind = 0;
        if (hind >= l) hind = l - 1;
        low = zz[lind];
        high = zz[hind];
        zz = null;

        for (int p = 0; p < k; p++) if (z[p] > 0 && (z[p] < low || z[p] > high)) z[p] = Double.NaN;

        for (int p = 0; p < k; p++) one[p] = 1.0;
        for (int p = 0; p < k; p++) if (z[p] == 0) one[p] = 0;

        //	find rows sums
        for (int p = 0; p < k; p++) {
            r[p] = 0;
        }
        for (int ic = 0; ic < numArrays; ic++) {
            for (int p = 0; p < numElementsForArrays[ic]; p++) {
                r[i[ic][p]] += x[ic][p] * one[j[ic][p]];
                if (i[ic][p] < j[ic][p]) {
                    r[j[ic][p]] += x[ic][p] * one[i[ic][p]];
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
            if ((r[p] < low && z[p] > 0) || Double.isNaN(z[p])) {
                bad[p] = 1;
                z[p] = 1.0;
            } else bad[p] = 0;
        }

        for (int p = 0; p < k; p++) b[p] = 1.0 - bad[p];
        for (int p = 0; p < k; p++) one[p] = 1.0 - bad[p];
        for (int p = 0; p < k; p++) if (z[p] == 0) one[p] = 0;
        for (int p = 0; p < k; p++)
            bad1[p] = (int) Math.round(1 - one[p])
                    ;

        //	start iterations
        //	r is the current rows sum; s is the correction vector to be applied to rows and columns
        err = 10.0 * (1.0 + tol);
        int iter = 0;
        int stuck = 0;
        double ber;
        for (int p = 0; p < k; p++) current[p] = b[p];
        while (err > tol && iter++ < maxIter) {
            for (int p = 0; p < k; p++) {
                if (bad1[p] == 1) r[p] = 1.0;
            }
            for (int p = 0; p < k; p++) {
                s[p] = Math.sqrt(z[p] / r[p]);
            }
            for (int p = 0; p < k; p++) {
                b[p] *= s[p];
            }

            for (int p = 0; p < k; p++) r[p] = 0;
            for (int ic = 0; ic < numArrays; ic++) {
                for (int p = 0; p < numElementsForArrays[ic]; p++) {
                    r[i[ic][p]] += x[ic][p] * b[j[ic][p]];
                    if (i[ic][p] < j[ic][p]) {
                        r[j[ic][p]] += x[ic][p] * b[i[ic][p]];
                    }
                }
            }
            for (int p = 0; p < k; p++) {
                r[p] *= b[p];
            }

            //	calculate the current relative error
            err = 0;
            ber = 0;
            for (int p = 0; p < k; p++) {
                if (bad1[p] == 1) continue;
                if (Math.abs((r[p] - z[p]) / z[p]) > err) err = Math.abs((r[p] - z[p]) / z[p]);
                if (Math.abs(b[p] - current[p]) > ber) ber = Math.abs(b[p] - current[p]);
            }
            report[iter - 1][0] = err;
            report[iter - 1][1] = ber;
            if (verbose) System.out.printf("%d: %30.15lf %30.15lf\n", iter, err, ber);
            for (int p = 0; p < k; p++) {
                current[p] = b[p];
            }
            if (iter < trials + 2) continue;
            if (err > (1.0 - del) * report[iter - 2][0]) stuck++;
            else stuck = 0;
            if (stuck >= trials) break;
        }
        for (int p = 0; p < k; p++) {
            if (bad[p] == 1) {
                b[p] = Double.NaN;
            }
        }

        if (err > tol) return (-iter);
        else return (iter);
    }


}
