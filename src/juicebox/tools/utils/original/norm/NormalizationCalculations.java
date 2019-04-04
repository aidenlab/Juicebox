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

import juicebox.data.ContactRecord;
import juicebox.data.MatrixZoomData;
import juicebox.windowui.NormalizationHandler;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.stat.StatUtils;
import org.broad.igv.Globals;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;


/**
 * Class for computing VC ("Vanilla Coverage") and KR normalization vector.
 * <p/>
 * Note: currently these are valid for square matrices only.
 *
 * @author jrobinso
 *         Date: 1/25/13
 *         Time: 4:03 PM
 */
public class NormalizationCalculations {

    private ArrayList<ContactRecord> contactRecords;
    private int totSize;
    private boolean isEnoughMemory = false;

    NormalizationCalculations(MatrixZoomData zd) {

        if (zd.getChr1Idx() != zd.getChr2Idx()) {
            throw new RuntimeException("Norm cannot be calculated for inter-chr matrices.");
        }

        Iterator<ContactRecord> iter1 = zd.getNewContactRecordIterator();
        int count = 0;
        while (iter1.hasNext()) {
            iter1.next();
            count++;
        }
        if (count * 1000 < Runtime.getRuntime().maxMemory()) {
            isEnoughMemory = true;

            this.contactRecords = new ArrayList<>();
            Iterator<ContactRecord> iter = zd.getNewContactRecordIterator();
            while (iter.hasNext()) {
                ContactRecord cr = iter.next();
                contactRecords.add(cr);
            }
            this.totSize = zd.getXGridAxis().getBinCount();
        }
    }

    public NormalizationCalculations(ArrayList<ContactRecord> list, int totSize) {
        this.contactRecords = list;
        this.totSize = totSize;
    }

    public static void calcKR(String path) throws IOException {

        BufferedReader reader = org.broad.igv.util.ParsingUtils.openBufferedReader(path);

        String nextLine;
        int lineCount = 0;
        int maxBin = 0;
        ArrayList<ContactRecord> readList = new ArrayList<>();
        while ((nextLine = reader.readLine()) != null) {
            lineCount++;
            String[] tokens = Globals.singleTabMultiSpacePattern.split(nextLine);
            int nTokens = tokens.length;
            if (nTokens != 3) {
                System.err.println("Number of columns incorrect at line" + lineCount + ": " + nextLine);
                System.exit(62);
            }
            int binX = Integer.parseInt(tokens[0]);
            int binY = Integer.parseInt(tokens[1]);
            int count = Integer.parseInt(tokens[2]);
            ContactRecord record = new ContactRecord(binX, binY, count);
            readList.add(record);
            if (binX > maxBin) maxBin = binX;
            if (binY > maxBin) maxBin = binY;
        }
        NormalizationCalculations nc = new NormalizationCalculations(readList, maxBin + 1);
        double[] norm = nc.getNorm(NormalizationHandler.KR);
        for (double d : norm) {
            System.out.println(d);
        }

    }

    /*
    function [x,res] = bnewt(A,tol,x0,delta,fl)
          % BNEWT A balancing algorithm for symmetric matrices
    %
    % X = BNEWT(A) attempts to find a vector X such that
    % diag(X)*A*diag(X) is close to doubly stochastic. A must
    % be symmetric and nonnegative.
    %
    % X0: initial guess. TOL: error tolerance.
    % DEL: how close balancing vectors can get to the edge of the
          % positive cone. We use a relative measure on the size of elements.
    % FL: intermediate convergence statistics on/off.
          % RES: residual error, measured by norm(diag(x)*A*x - e).
          % Initialise
          [n,n]=size(A); e = ones(n,1); res=[];
           if nargin < 5, fl = 0; end
        if nargin < 4, delta = 0.1; end
        if nargin < 3, x0 = e; end
        if nargin < 2, tol = 1e-6; end
    */
    private static double[] computeKRNormVector(int[] offset, ArrayList<ContactRecord> list, double tol, double[] x0, double delta) {

        int n = x0.length;
        double[] e = new double[n];
        for (int i = 0; i < e.length; i++) e[i] = 1;

        double g = 0.9;
        double etamax = 0.1;
        double eta = etamax;

        double rt = Math.pow(tol, 2);

        double[] v = sparseMultiplyFromContactRecords(offset, list, x0);
        double[] rk = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            v[i] = v[i] * x0[i];
            rk[i] = 1 - v[i];
        }
        double rho_km1 = 0;
        for (double aRk : rk) {
            rho_km1 += aRk * aRk;
        }
        double rout = rho_km1;
        double rold = rout;
        int MVP = 0;  // We'll count matrix vector products.

        int not_changing = 0;
        while (rout > rt && not_changing < 100) {    // Outer iteration
            int k = 0;
            double[] y = new double[e.length];
            double[] ynew = new double[e.length];
            double[] Z = new double[e.length];
            double[] p = new double[e.length];
            double[] w = new double[e.length];
            double alpha;
            double beta;
            double gamma;
            double rho_km2 = rho_km1;
            System.arraycopy(e, 0, y, 0, y.length);

            double innertol = Math.max(Math.pow(eta, 2) * rout, rt);
            while (rho_km1 > innertol) {   // Inner iteration by CG
                k++;

                if (k == 1) {
                    rho_km1 = 0;
                    for (int i = 0; i < Z.length; i++) {
                        Z[i] = rk[i] / v[i];
                        p[i] = Z[i];
                        rho_km1 += rk[i] * Z[i];
                    }
                } else {
                    beta = rho_km1 / rho_km2;
                    for (int i = 0; i < p.length; i++) {
                        p[i] = Z[i] + beta * p[i];
                    }
                }
                double[] tmp = new double[e.length];
                for (int i = 0; i < tmp.length; i++) {
                    tmp[i] = x0[i] * p[i];
                }
                tmp = sparseMultiplyFromContactRecords(offset, list, tmp);
                alpha = 0;
                // Update search direction efficiently.
                for (int i = 0; i < tmp.length; i++) {
                    w[i] = x0[i] * tmp[i] + v[i] * p[i];
                    alpha += p[i] * w[i];
                }
                alpha = rho_km1 / alpha;
                double minynew = Double.MAX_VALUE;
                // Test distance to boundary of cone.
                for (int i = 0; i < p.length; i++) {
                    ynew[i] = y[i] + alpha * p[i];
                    if (ynew[i] < minynew) minynew = ynew[i];
                }
                if (minynew <= delta) {
                    if (delta == 0) break;     // break out of inner loop?
                    gamma = Double.MAX_VALUE;
                    for (int i = 0; i < ynew.length; i++) {
                        if (alpha * p[i] < 0) {
                            if ((delta - y[i]) / (alpha * p[i]) < gamma) {
                                gamma = (delta - y[i]) / (alpha * p[i]);
                            }
                        }
                    }
                    for (int i = 0; i < y.length; i++)
                        y[i] = y[i] + gamma * alpha * p[i];
                    break;   // break out of inner loop?
                }
                rho_km2 = rho_km1;
                rho_km1 = 0;
                for (int i = 0; i < y.length; i++) {
                    y[i] = ynew[i];
                    rk[i] = rk[i] - alpha * w[i];
                    Z[i] = rk[i] / v[i];
                    rho_km1 += rk[i] * Z[i];
                }

            } // end inner loop
            for (int i = 0; i < x0.length; i++) {
                x0[i] = x0[i] * y[i];
            }
            v = sparseMultiplyFromContactRecords(offset, list, x0);
            rho_km1 = 0;
            for (int i = 0; i < v.length; i++) {
                v[i] = v[i] * x0[i];
                rk[i] = 1 - v[i];
                rho_km1 += rk[i] * rk[i];
            }
            if (Math.abs(rho_km1 - rout) < 0.000001 || Double.isInfinite(rho_km1)) {
                not_changing++;
            }
            rout = rho_km1;
            MVP = MVP + k + 1;
            //  Update inner iteration stopping criterion.
            double rat = rout / rold;
            rold = rout;
            double r_norm = Math.sqrt(rout);
            double eta_o = eta;
            eta = g * rat;
            if (g * Math.pow(eta_o, 2) > 0.1) {
                eta = Math.max(eta, g * Math.pow(eta_o, 2));
            }
            eta = Math.max(Math.min(eta, etamax), 0.5 * tol / r_norm);
        }
        if (not_changing >= 100) {
            return null;
        }
        return x0;
    }

    private static double[] sparseMultiplyFromContactRecords(int[] offset, ArrayList<ContactRecord> list, double[] vector) {
        double[] result = new double[vector.length];

        for (ContactRecord cr : list) {
            int row = cr.getBinX();
            int col = cr.getBinY();
            float value = cr.getCounts();

            row = offset[row];
            col = offset[col];

            if (row != -1 && col != -1) {
                result[row] += vector[col] * value;
                if (row != col) {
                    result[col] += vector[row] * value;
                }
            }
        }

        return result;
    }

    boolean isEnoughMemory() {
        return isEnoughMemory;
    }

    public double[] getNorm(NormalizationType normOption) {
        double[] norm;
        switch (normOption.getLabel().toUpperCase()) {
            case NormalizationHandler.strKR:
            case NormalizationHandler.strGW_KR:
            case NormalizationHandler.strINTER_KR:
                norm = computeKR();
                break;
            case NormalizationHandler.strVC:
            case NormalizationHandler.strVC_SQRT:
            case NormalizationHandler.strGW_VC:
            case NormalizationHandler.strINTER_VC:
                norm = computeVC();
                break;
            case NormalizationHandler.strSCALE:
            case NormalizationHandler.strGW_SCALE:
            case NormalizationHandler.strINTER_SCALE:
                norm = computeMMBA();
                break;
            case NormalizationHandler.strNONE:
                norm = new double[totSize];
                Arrays.fill(norm, 1);
                return norm;
            default:
                System.err.println("Not supported for normalization " + normOption);
                return null;
        }

        if (norm != null) {
            double factor = getSumFactor(norm);
            for (int i = 0; i < norm.length; i++) {
                norm[i] = norm[i] * factor;
            }
        }
        return norm;
    }

    /**
     * Compute vanilla coverage norm, just the sum of the rows
     *
     * @return Normalization vector
     */
    double[] computeVC() {
        double[] rowsums = new double[totSize];

        for (int i = 0; i < rowsums.length; i++) rowsums[i] = 0;

        for (ContactRecord cr : contactRecords) {
            int x = cr.getBinX();
            int y = cr.getBinY();
            float value = cr.getCounts();
            rowsums[x] += value;
            if (x != y) {
                rowsums[y] += value;
            }
        }

        return rowsums;

    }

    /**
     * Get the sum of the normalized matrix
     *
     * @param norm Normalization vector
     * @return Square root of ratio of original to normalized vector
     */
    double getSumFactor(double[] norm) {
        double matrix_sum = 0;
        double norm_sum = 0;
        for (ContactRecord cr : contactRecords) {
            int x = cr.getBinX();
            int y = cr.getBinY();
            float value = cr.getCounts();
            if (!Double.isNaN(norm[x]) && !Double.isNaN(norm[y]) && norm[x] > 0 && norm[y] > 0) {
                // want total sum of matrix, not just upper triangle
                if (x == y) {
                    norm_sum += value / (norm[x] * norm[y]);
                    matrix_sum += value;
                } else {
                    norm_sum += 2 * value / (norm[x] * norm[y]);
                    matrix_sum += 2 * value;
                }

            }
        }
        return Math.sqrt(norm_sum / matrix_sum);
    }

    double[] computeKR() {

        boolean recalculate = true;
        int[] offset = getOffset(0);
        double[] kr = null;
        int iteration = 1;

        while (recalculate && iteration <= 6) {
            // create new matrix indices upon every iteration, because we've thrown out rows
            // newSize is size of new sparse matrix (non-sparse rows)
            int newSize = 0;
            for (int offset1 : offset) {
                if (offset1 != -1) newSize++;
            }

            // initialize x0 for call the compute KR norm
            double[] x0 = new double[newSize];
            for (int i = 0; i < x0.length; i++) x0[i] = 1;

            x0 = computeKRNormVector(offset, contactRecords, 0.000001, x0, 0.1);

            // assume all went well and we don't need to recalculate
            recalculate = false;
            int rowsTossed = 0;

            if (x0 == null || iteration == 5) {
                // if x0 is no good, throw out some percentage of rows and reset the offset array that gives those rows
                recalculate = true;
                if (iteration < 5) {
                    offset = getOffset(iteration);
                } else {
                    offset = getOffset(10);
                }
                //   System.out.print(" " + iteration + "%");
            } else {
                // otherwise, check to be sure there are no tiny KR values
                // create true KR vector
                kr = new double[totSize];
                int krIndex = 0;
                for (int offset1 : offset) {
                    if (offset1 == -1) {
                        kr[krIndex++] = Double.NaN;
                    } else {
                        kr[krIndex++] = (1.0 / x0[offset1]);
                    }
                }
                // find scaling factor
                double mySum = getSumFactor(kr);

                // if any values are too small, recalculate.  set those rows to be thrown out and reset the offset
                // note that if no rows are thrown out, the offset should not change
                int index = 0;
                for (int i = 0; i < kr.length; i++) {
                    if (kr[i] * mySum < 0.01) {
                        offset[i] = -1;
                        rowsTossed++;
                        recalculate = true;
                    } else {
                        if (offset[i] != -1) offset[i] = index++;
                    }
                }
                // if (recalculate) System.out.print(" " + rowsTossed);
            }
            iteration++;
            System.gc();
        }
        if (iteration > 6 && recalculate) {
            kr = new double[totSize];
            for (int i = 0; i < totSize; i++) {
                kr[i] = Double.NaN;
            }
        }

        return kr;

    }

    private int[] getOffset(double percent) {
        double[] rowSums = new double[totSize];

        for (int i = 0; i < rowSums.length; i++) rowSums[i] = 0;

        for (ContactRecord cr : contactRecords) {
            int x = cr.getBinX();
            int y = cr.getBinY();
            float value = cr.getCounts();
            rowSums[x] += value;
            if (x != y) {
                rowSums[y] += value;
            }
        }

        double thresh = 0;
        if (percent > 0) {
            // Get percent threshold from positive row sums (nonzero)
            int j = 0;
            for (double sum : rowSums) if (sum != 0) j++;
            double[] posRowSums = new double[j];
            j = 0;
            for (double sum : rowSums) if (sum != 0) posRowSums[j++] = sum;
            thresh = StatUtils.percentile(posRowSums, percent);
        }
        int[] offset = new int[rowSums.length];
        int index = 0;
        for (int i = 0; i < rowSums.length; i++) {
            if (rowSums[i] <= thresh) {
                offset[i] = -1;
            } else {
                offset[i] = index++;
            }
        }

        return offset;

    }

    public double[] computeMMBA() {

        double[] tempTargetVector = new double[totSize];
        for (int k = 0; k < totSize; k++) {
            tempTargetVector[k] = 1;
        }

        return ZeroScale.mmbaScaleToVector(contactRecords, tempTargetVector);
    }
}
