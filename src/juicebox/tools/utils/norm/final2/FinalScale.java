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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class FinalScale {

    float tol = .0005f;
    float del = .02f;
    float perc = .01f;
    float dp = .005f;
    float perc1 = 0.0025f;
    float dp1 = .001f;
    boolean verb = true;
    int maxiter = 100;
    boolean zerodiag = false;
    int All_iter = 200;
    int threads = 1;

    private static double[] scaleToTargetVector(List<ContactRecord> contactRecords, double[] targetVectorInitial, double tolerance,
                                                double percentLowRowSumExcluded, double percentZValsToIgnore,
                                                int maxIter, double del, int numTrials) {

        double high, low, ber;
        int lind, hind;

        //	find the matrix dimensions
        int k = targetVectorInitial.length;

        double[] current = new double[k];
        double[] row = new double[k];
        double[] col = new double[k];
        double[] dr = new double[k];
        double[] dc = new double[k];
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

        Arrays.fill(one, 1);
        Arrays.fill(bad, 1);
        for (int p = 0; p < k; p++) if (targetVector[p] == 0) one[p] = 0;


        for (ContactRecord cr : contactRecords) {
            int x = cr.getBinX();
            int y = cr.getBinY();
            if (x == y) {
                bad[x] = 0;
            }
        }


        //	find rows sums
        utmvMul(contactRecords, one, row);


        //	find relevant percentiles
        System.arraycopy(row, 0, r0, 0, k);

        Arrays.sort(r0);

        int n = 0;
        for (int p = 0; p < k; p++) {
            if (r0[p] == 0) {
                n++;
            }
        }

        lind = n - 1 + (int) (((double) (k - n)) * percentLowRowSumExcluded + 0.5);
        hind = n - 1 + (int) (((double) (k - n)) * (1.0 - 0.1 * percentLowRowSumExcluded) + 0.5);

        if (lind < 0) {
            lind = 0;
        }
        if (hind >= k) {
            hind = k - 1;
        }
        low = r0[lind];
        high = r0[hind];
        r0 = null;

        //	find the "bad" rows and exclude them
        for (int p = 0; p < k; p++) {
            if (((row[p] < low || row[p] > high) && targetVector[p] > 0) || Double.isNaN(targetVector[p])) {
                bad[p] = 1;
                targetVector[p] = 1.0;
            }
        }

        double[] calculatedVector = new double[k];
        double[] errorForIteration = new double[maxIter];

        for (int p = 0; p < k; p++) {
            dr[p] = 1.0 - bad[p];
        }
        for (int p = 0; p < k; p++) {
            dc[p] = dr[p];
        }
        for (int p = 0; p < k; p++) {
            one[p] = dr[p];
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
        //	row is the current rows sum; s is the correction vector to be applied to rows and columns
        ber = 10.0 * (1.0 + tolerance);
        int iter = 0;
        int stuck = 0;

        for (int p = 0; p < k; p++) {
            current[p] = dr[p];
        }

        double err = 0;
        while ((ber > tolerance || err > 5.0 * tolerance) && iter++ < maxIter) {
            for (int p = 0; p < k; p++) {
                if (bad1[p] == 1) row[p] = 1.0;
            }
            for (int p = 0; p < k; p++) {
                s[p] = targetVector[p] / row[p];
            }
            for (int p = 0; p < k; p++) {
                dr[p] *= s[p];
            }

            utmvMul(contactRecords, dr, col);

            for (int p = 0; p < k; p++) col[p] *= dc[p];
            for (int p = 0; p < k; p++) if (bad1[p] == 1) col[p] = 1.0;
            for (int p = 0; p < k; p++) s[p] = targetVector[p] / col[p];
            for (int p = 0; p < k; p++) dc[p] *= s[p];

            utmvMul(contactRecords, dc, row);
            for (int p = 0; p < k; p++) row[p] *= dr[p];

            for (int p = 0; p < k; p++) {
                calculatedVector[p] = Math.sqrt(dr[p] * dc[p]);
            }

            //	calculate the current relative error
            ber = 0;
            for (int p = 0; p < k; p++) {
                if (bad1[p] == 1) continue;
                if (Math.abs(calculatedVector[p] - current[p]) > ber) ber = Math.abs(calculatedVector[p] - current[p]);
            }

            errorForIteration[iter - 1] = ber;
            System.arraycopy(calculatedVector, 0, current, 0, k);
            if (iter < numTrials + 2) continue;
            if (ber > (1.0 - del) * errorForIteration[iter - 2]) stuck++;
            else stuck = 0;
            if (stuck >= numTrials) break;
        }

        utmvMul(contactRecords, calculatedVector, col);
        err = 0;
        for (int p = 0; p < k; p++) {
            if (bad1[p] == 1) continue;
            if (err < Math.abs(col[p] * calculatedVector[p] - targetVector[p]))
                err = Math.abs(col[p] * calculatedVector[p] - targetVector[p]);
        }
        for (int p = 0; p < k; p++) {
            if (bad[p] == 1) {
                calculatedVector[p] = Double.NaN;
            }
        }

        if (ber > tolerance) {
            return null;
        }
        return calculatedVector;
    }


    public void utmvMul(int[][] iIndexes2D, int[][] jIndexes2D, float[][] xValues2D, float[] vVector, float[] result, float[][] rs, int maxIndex) {

        int numThreads = xValues2D.length;

        ScaleThreadObject[] objects = new ScaleThreadObject[numThreads];
        for (int t = 0; t < numThreads; t++) {
//		rs[t] = (float *) malloc(k*sizeof(float));
            Arrays.fill(rs[t], 0);
            objects[t] = new ScaleThreadObject(iIndexes2D[t], jIndexes2D[t], xValues2D[t], vVector, rs[t]);
        }

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        for (int t = 0; t < numThreads; t++) {
            final int t2 = t;
            Runnable worker = new Runnable() {
                @Override
                public void run() {
                    objects[t2].execute();
                }
            };
            executor.execute(worker);
        }
        executor.shutdown();

        // Wait until all threads finish
        while (!executor.isTerminated()) {
        }

        for (int p = 0; p < maxIndex; p++) {
            result[p] = 0;
            for (int t = 0; t < numThreads; t++) {
                result[p] += rs[t][p];
            }
        }
    }
}
