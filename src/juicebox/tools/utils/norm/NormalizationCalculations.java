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

package juicebox.tools.utils.norm;

import juicebox.data.ContactRecord;
import juicebox.data.MatrixZoomData;
import juicebox.data.basics.ListOfDoubleArrays;
import juicebox.data.basics.ListOfFloatArrays;
import juicebox.data.basics.ListOfIntArrays;
import juicebox.windowui.NormalizationHandler;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.stat.StatUtils;
import org.broad.igv.Globals;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


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
    
    private List<List<ContactRecord>> contactRecords;
    private long totSize;
    private boolean isEnoughMemory = false;

    /**
     * Initializing from a single MatrixZoomData object
     *
     * @param zd
     */
    public NormalizationCalculations(MatrixZoomData zd) {

        if (zd.getChr1Idx() != zd.getChr2Idx()) {
            throw new RuntimeException("Norm cannot be calculated for inter-chr matrices.");
        }

        long count = zd.getNumberOfContactRecords();
        if (count * 1000 < Runtime.getRuntime().maxMemory()) {
            isEnoughMemory = true;

            this.contactRecords = zd.getContactRecordList();
            this.totSize = zd.getXGridAxis().getBinCount();
        }
    }

    /**
     * Initialize from genomewide data or direct dump/calcK CLT
     *
     * @param list
     * @param totSize
     */
    public NormalizationCalculations(List<List<ContactRecord>> list, int totSize) {
        this.contactRecords = list;
        this.totSize = totSize;
    }

    public static void calcKR(String path) throws IOException {

        BufferedReader reader = org.broad.igv.util.ParsingUtils.openBufferedReader(path);

        String nextLine;
        int lineCount = 0;
        int maxBin = 0;
        List<ContactRecord> readList = new ArrayList<>();
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
        List<List<ContactRecord>> listOfLists = new ArrayList<>();
        listOfLists.add(readList);
        NormalizationCalculations nc = new NormalizationCalculations(listOfLists, maxBin + 1);
        for (float[] array : nc.getNorm(NormalizationHandler.KR).getValues()) {
            for (double d : array) {
                System.out.println(d);
            }
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
    private static ListOfFloatArrays computeKRNormVector(ListOfIntArrays offset, List<List<ContactRecord>> listOfLists, double tol, ListOfFloatArrays x0, double delta) {
    
        long n = x0.getLength();
        ListOfFloatArrays e = new ListOfFloatArrays(n, 1);
    
        double g = 0.9;
        double etamax = 0.1;
        double eta = etamax;
    
        double rt = Math.pow(tol, 2);
    
        ListOfFloatArrays v = sparseMultiplyFromContactRecords(offset, listOfLists, x0);
        ListOfFloatArrays rk = new ListOfFloatArrays(v.getLength());
        for (long i = 0; i < v.getLength(); i++) {
            v.multiplyBy(i, x0.get(i));
            rk.set(i, 1 - v.get(i));
        }
        float rho_km1 = 0;
        for (float[] aRkArray : rk.getValues()) {
            for (float aRk : aRkArray) {
                rho_km1 += aRk * aRk;
            }
        }
        double rout = rho_km1;
        double rold = rout;
        int MVP = 0;  // We'll count matrix vector products.
    
        int not_changing = 0;
        while (rout > rt && not_changing < 100) {    // Outer iteration
            int k = 0;
            ListOfFloatArrays y = e.deepClone();
            ListOfFloatArrays ynew = new ListOfFloatArrays(e.getLength());
            ListOfFloatArrays Z = new ListOfFloatArrays(e.getLength());
            ListOfFloatArrays p = new ListOfFloatArrays(e.getLength());
            ListOfFloatArrays w = new ListOfFloatArrays(e.getLength());
            float alpha;
            double beta;
            float gamma;
            double rho_km2 = rho_km1;
        
        
            double innertol = Math.max(Math.pow(eta, 2) * rout, rt);
            while (rho_km1 > innertol) {   // Inner iteration by CG
                k++;

                if (k == 1) {
                    rho_km1 = 0;
                    for (long i = 0; i < Z.getLength(); i++) {
                        float rkVal = rk.get(i);
                        float zVal = rkVal / v.get(i);
                        Z.set(i, zVal);
                        rho_km1 += rkVal * zVal;
                    }
                    p = Z.deepClone();
    
                } else {
                    beta = rho_km1 / rho_km2;
                    p.multiplyEverythingBy(beta);
                    for (long i = 0; i < p.getLength(); i++) {
                        p.addTo(i, Z.get(i));
                    }
                }
                ListOfFloatArrays tmp = new ListOfFloatArrays(e.getLength());
                for (long i = 0; i < tmp.getLength(); i++) {
                    tmp.set(i, x0.get(i) * p.get(i));
                }
                tmp = sparseMultiplyFromContactRecords(offset, listOfLists, tmp);
                alpha = 0;
                // Update search direction efficiently.
                for (long i = 0; i < tmp.getLength(); i++) {
                    double pVal = p.get(i);
                    float wVal = (float) (x0.get(i) * tmp.get(i) + v.get(i) * pVal);
                    w.set(i, wVal);
                    alpha += pVal * wVal;
                }
                alpha = rho_km1 / alpha;
                double minynew = Double.MAX_VALUE;
                // Test distance to boundary of cone.
                for (long i = 0; i < p.getLength(); i++) {
                    float yVal = y.get(i) + alpha * p.get(i);
                    ynew.set(i, yVal);
                    if (yVal < minynew) {
                        minynew = yVal;
                    }
                }
                if (minynew <= delta) {
                    if (delta == 0) break;     // break out of inner loop?
                    gamma = Float.MAX_VALUE;
                    for (int i = 0; i < ynew.getLength(); i++) {
                        double pVal = p.get(i);
                        if (alpha * pVal < 0) {
                            double yVal = y.get(i);
                            if ((delta - yVal) / (alpha * pVal) < gamma) {
                                gamma = (float) ((delta - yVal) / (alpha * pVal));
                            }
                        }
                    }
                    for (int i = 0; i < y.getLength(); i++)
                        y.addTo(i, gamma * alpha * p.get(i));
                    break;   // break out of inner loop?
                }
                rho_km2 = rho_km1;
                rho_km1 = 0;
                for (long i = 0; i < y.getLength(); i++) {
                    y.set(i, ynew.get(i));
                    rk.addTo(i, -alpha * w.get(i));
                    float rkVal = rk.get(i);
                    Z.set(i, rkVal / v.get(i));
                    rho_km1 += rkVal * Z.get(i);
                }
            
            } // end inner loop
            for (long i = 0; i < x0.getLength(); i++) {
                x0.multiplyBy(i, y.get(i));
            }
            v = sparseMultiplyFromContactRecords(offset, listOfLists, x0);
            rho_km1 = 0;
            for (long i = 0; i < v.getLength(); i++) {
                v.multiplyBy(i, x0.get(i));
                float rkVal = 1 - v.get(i);
                rk.set(i, rkVal);
            
                rho_km1 += rkVal * rkVal;
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
    
    private static ListOfFloatArrays sparseMultiplyFromContactRecords(ListOfIntArrays offset, List<List<ContactRecord>> listOfLists, ListOfFloatArrays vector) {
        ListOfFloatArrays result = new ListOfFloatArrays(vector.getLength());
        
        for (List<ContactRecord> localList : listOfLists) {
            for (ContactRecord cr : localList) {
                int row = cr.getBinX();
                int col = cr.getBinY();
                float value = cr.getCounts();
                
                row = offset.get(row);
                col = offset.get(col);
                
                if (row != -1 && col != -1) {
                    result.addTo(row, vector.get(col) * value);
                    if (row != col) {
                        result.addTo(col, vector.get(row) * value);
                    }
                }
            }
        }
        
        return result;
    }
    
    boolean isEnoughMemory() {
        return isEnoughMemory;
    }
    
    public ListOfFloatArrays getNorm(NormalizationType normOption) {
        ListOfFloatArrays norm;
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
                return new ListOfFloatArrays(totSize, 1);
            default:
                System.err.println("Not supported for normalization " + normOption);
                return null;
        }
        
        if (norm != null && norm.getLength() > 0) {
            double factor = getSumFactor(norm);
            System.out.println();
            norm.multiplyEverythingBy(factor);
        }
        return norm;
    }
    
    /**
     * Compute vanilla coverage norm, just the sum of the rows
     *
     * @return Normalization vector
     */
    ListOfFloatArrays computeVC() {
        ListOfFloatArrays rowsums = new ListOfFloatArrays(totSize, 0);
        
        for (List<ContactRecord> localList : contactRecords) {
            for (ContactRecord cr : localList) {
                int x = cr.getBinX();
                int y = cr.getBinY();
                float value = cr.getCounts();
                rowsums.addTo(x, value);
                if (x != y) {
                    rowsums.addTo(y, value);
                }
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
    public double getSumFactor(ListOfFloatArrays norm) {
        Double[] normMatrixSums = getNormMatrixSumFactor(norm);
        return Math.sqrt(normMatrixSums[0] / normMatrixSums[1]);
    }
    
    public Double[] getNormMatrixSumFactor(ListOfFloatArrays norm) {
        double matrix_sum = 0;
        double norm_sum = 0;
        for (List<ContactRecord> localList : contactRecords) {
            for (ContactRecord cr : localList) {
                int x = cr.getBinX();
                int y = cr.getBinY();
                float value = cr.getCounts();
                double valX = norm.get(x);
                double valY = norm.get(y);
                if (!Double.isNaN(valX) && !Double.isNaN(valY) && valX > 0 && valY > 0) {
                    // want total sum of matrix, not just upper triangle
                    if (x == y) {
                        norm_sum += value / (valX * valY);
                        matrix_sum += value;
                    } else {
                        norm_sum += 2 * value / (valX * valY);
                        matrix_sum += 2 * value;
                    }
                }
            }
        }
        return new Double[]{norm_sum, matrix_sum};
    }
    
    public Double[] getNormMatrixSumFactor(double[] norm) {
        double matrix_sum = 0;
        double norm_sum = 0;
        for (List<ContactRecord> localList : contactRecords) {
            for (ContactRecord cr : localList) {
                int x = cr.getBinX();
                int y = cr.getBinY();
                float value = cr.getCounts();
                double valX = norm[x];
                double valY = norm[y];
                if (!Double.isNaN(valX) && !Double.isNaN(valY) && valX > 0 && valY > 0) {
                    // want total sum of matrix, not just upper triangle
                    if (x == y) {
                        norm_sum += value / (valX * valY);
                        matrix_sum += value;
                    } else {
                        norm_sum += 2 * value / (valX * valY);
                        matrix_sum += 2 * value;
                    }
                }
            }
        }
        return new Double[]{norm_sum, matrix_sum};
    }
    
    
    public int getNumberOfValidEntriesInVector(double[] norm) {
        int counter = 0;
        for (double val : norm) {
            if (!Double.isNaN(val) && val > 0) {
                counter++;
            }
        }
        return counter;
    }
    
    
    ListOfFloatArrays computeKR() {
        
        boolean recalculate = true;
        ListOfIntArrays offset = getOffset(0);
        ListOfFloatArrays kr = null;
        int iteration = 1;
        
        while (recalculate && iteration <= 6) {
            // create new matrix indices upon every iteration, because we've thrown out rows
            // newSize is size of new sparse matrix (non-sparse rows)
            long newSize = 0;
            for (int[] array : offset.getValues()) {
                for (int offset1 : array) {
                    if (offset1 != -1) newSize++;
                }
            }
            
            // initialize x0 for call the compute KR norm
            ListOfFloatArrays x0 = new ListOfFloatArrays(newSize, 1);
            
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
                kr = new ListOfFloatArrays(totSize);
                int krIndex = 0;
                for (int[] offsetArray : offset.getValues()) {
                    for (int offset1 : offsetArray) {
                        if (offset1 == -1) {
                            kr.set(krIndex++, Float.NaN);
                        } else {
                            kr.set(krIndex++, (1.0f / x0.get(offset1)));
                        }
                    }
                }
                // find scaling factor
                double mySum = getSumFactor(kr);
    
                // if any values are too small, recalculate.  set those rows to be thrown out and reset the offset
                // note that if no rows are thrown out, the offset should not change
                int index = 0;
                for (long i = 0; i < kr.getLength(); i++) {
                    if (kr.get(i) * mySum < 0.01) {
                        offset.set(i, -1);
                        rowsTossed++;
                        recalculate = true;
                    } else {
                        if (offset.get(i) != -1) {
                            offset.set(i, index++);
                        }
                    }
                }
                // if (recalculate) System.out.print(" " + rowsTossed);
            }
            iteration++;
            System.gc();
        }
        if (iteration > 6 && recalculate) {
            kr = new ListOfFloatArrays(totSize, Float.NaN);
        }
        
        return kr;
    }
    
    private ListOfIntArrays getOffset(double percent) {
        ListOfDoubleArrays rowSums = new ListOfDoubleArrays(totSize, 0);
        
        for (List<ContactRecord> localList : contactRecords) {
            for (ContactRecord cr : localList) {
                int x = cr.getBinX();
                int y = cr.getBinY();
                float value = cr.getCounts();
                rowSums.addTo(x, value);
                if (x != y) {
                    rowSums.addTo(y, value);
                }
            }
        }

        double thresh = 0;
        if (percent > 0) {
            // Get percent threshold from positive row sums (nonzero)
            int j = 0;
            for (double[] array : rowSums.getValues()) {
                for (double sum : array) {
                    if (sum != 0) {
                        j++;
                    }
                }
            }
            double[] posRowSums = new double[j];
            j = 0;
            for (double[] array : rowSums.getValues()) {
                for (double sum : array) {
                    if (sum != 0) {
                        posRowSums[j++] = sum;
                    }
                }
            }
            thresh = StatUtils.percentile(posRowSums, percent);
        }
        
        ListOfIntArrays offset = new ListOfIntArrays(rowSums.getLength());
        int index = 0;
        for (long i = 0; i < rowSums.getLength(); i++) {
            if (rowSums.get(i) <= thresh) {
                offset.set(i, -1);
            } else {
                offset.set(i, index++);
            }
        }
        
        return offset;
        
    }
    
    public ListOfFloatArrays computeMMBA() {
        
        ListOfFloatArrays tempTargetVector = new ListOfFloatArrays(totSize, 1);
        
        return ZeroScale.mmbaScaleToVector(contactRecords, tempTargetVector);
    }
}
