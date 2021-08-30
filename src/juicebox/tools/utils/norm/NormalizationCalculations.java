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

package juicebox.tools.utils.norm;

import juicebox.data.ContactRecord;
import juicebox.data.basics.ListOfDoubleArrays;
import juicebox.data.basics.ListOfFloatArrays;
import juicebox.data.basics.ListOfIntArrays;
import juicebox.data.iterator.IteratorContainer;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Random;


/**
 * Class for computing VC ("Vanilla Coverage") and KR normalization vector.
 * <p/>
 * Note: currently these are valid for square matrices only.
 *
 * @author jrobinso
 * Date: 1/25/13
 * Time: 4:03 PM
 */
public class NormalizationCalculations {

    private final long matrixSize; // x and y symmetric
    private boolean isEnoughMemory = false;
    private final IteratorContainer ic;

    public NormalizationCalculations(IteratorContainer ic) {
        this.ic = ic;
        this.matrixSize = ic.getMatrixSize();
        isEnoughMemory = ic.getIsThereEnoughMemoryForNormCalculation();
    }

    private static ListOfDoubleArrays sparseMultiplyFromContactRecords(ListOfIntArrays offset,
                                                                       Iterator<ContactRecord> iterator, ListOfDoubleArrays vector) {
        ListOfDoubleArrays result = new ListOfDoubleArrays(vector.getLength());

        while (iterator.hasNext()) {
            ContactRecord cr = iterator.next();
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

        return result;
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
    private ListOfDoubleArrays computeKRNormVector(ListOfIntArrays offset, double tol, ListOfDoubleArrays x0, double delta) {

        long n = x0.getLength();
        ListOfDoubleArrays e = new ListOfDoubleArrays(n, 1);

        double g = 0.9;
        double etamax = 0.1;
        double eta = etamax;

        double rt = Math.pow(tol, 2);

        ListOfDoubleArrays v = sparseMultiplyFromContactRecords(offset, getIterator(), x0);
        ListOfDoubleArrays rk = new ListOfDoubleArrays(v.getLength());
        for (long i = 0; i < v.getLength(); i++) {
            v.multiplyBy(i, x0.get(i));
            rk.set(i, 1 - v.get(i));
        }
        double rho_km1 = 0;
        for (double[] aRkArray : rk.getValues()) {
            for (double aRk : aRkArray) {
                rho_km1 += aRk * aRk;
            }
        }
        double rout = rho_km1;
        double rold = rout;
        int MVP = 0;  // We'll count matrix vector products.

        int not_changing = 0;
        while (rout > rt && not_changing < 100) {    // Outer iteration
            int k = 0;
            ListOfDoubleArrays y = e.deepClone();
            ListOfDoubleArrays ynew = new ListOfDoubleArrays(e.getLength());
            ListOfDoubleArrays Z = new ListOfDoubleArrays(e.getLength());
            ListOfDoubleArrays p = new ListOfDoubleArrays(e.getLength());
            ListOfDoubleArrays w = new ListOfDoubleArrays(e.getLength());
            double alpha;
            double beta;
            double gamma;
            double rho_km2 = rho_km1;


            double innertol = Math.max(Math.pow(eta, 2) * rout, rt);
            while (rho_km1 > innertol) {   // Inner iteration by CG
                k++;

                if (k == 1) {
                    rho_km1 = 0;
                    for (long i = 0; i < Z.getLength(); i++) {
                        double rkVal = rk.get(i);
                        double zVal = rkVal / v.get(i);
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
                ListOfDoubleArrays tmp = new ListOfDoubleArrays(e.getLength());
                for (long i = 0; i < tmp.getLength(); i++) {
                    tmp.set(i, x0.get(i) * p.get(i));
                }
                tmp = sparseMultiplyFromContactRecords(offset, getIterator(), tmp);
                alpha = 0;
                // Update search direction efficiently.
                for (long i = 0; i < tmp.getLength(); i++) {
                    double pVal = p.get(i);
                    double wVal = (x0.get(i) * tmp.get(i) + v.get(i) * pVal);
                    w.set(i, wVal);
                    alpha += pVal * wVal;
                }
                alpha = rho_km1 / alpha;
                double minynew = Double.MAX_VALUE;
                // Test distance to boundary of cone.
                for (long i = 0; i < p.getLength(); i++) {
                    double yVal = y.get(i) + alpha * p.get(i);
                    ynew.set(i, yVal);
                    if (yVal < minynew) {
                        minynew = yVal;
                    }
                }
                if (minynew <= delta) {
                    if (delta == 0) break;     // break out of inner loop?
                    gamma = Double.MAX_VALUE;
                    for (long i = 0; i < ynew.getLength(); i++) {
                        double pVal = p.get(i);
                        if (alpha * pVal < 0) {
                            double yVal = y.get(i);
                            if ((delta - yVal) / (alpha * pVal) < gamma) {
                                gamma = ((delta - yVal) / (alpha * pVal));
                            }
                        }
                    }
                    for (long i = 0; i < y.getLength(); i++) {
                        y.addTo(i, gamma * alpha * p.get(i));
                    }
                    break;   // break out of inner loop?
                }
                rho_km2 = rho_km1;
                rho_km1 = 0;
                y = ynew.deepClone();
                for (long i = 0; i < y.getLength(); i++) {
                    rk.addTo(i, -alpha * w.get(i));
                    double rkVal = rk.get(i);
                    Z.set(i, rkVal / v.get(i));
                    rho_km1 += rkVal * Z.get(i);
                }

            } // end inner loop
            for (long i = 0; i < x0.getLength(); i++) {
                x0.multiplyBy(i, y.get(i));
            }
            v = sparseMultiplyFromContactRecords(offset, getIterator(), x0);
            rho_km1 = 0;
            for (long i = 0; i < v.getLength(); i++) {
                v.multiplyBy(i, x0.get(i));
                double rkVal = 1 - v.get(i);
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

    private Iterator<ContactRecord> getIterator() {
        return ic.getNewContactRecordIterator();
    }

    boolean isEnoughMemory() {
        return isEnoughMemory;
    }

    public ListOfFloatArrays getNorm(NormalizationType normOption) {
        ListOfFloatArrays norm;
        if (normOption.usesKR()) {
            norm = computeKR();
        } else if (normOption.usesVC()) {
            norm = computeVC();
        } else if (normOption.usesSCALE()) {
            norm = computeMMBA();
        } else if (normOption.isNONE()) {
            return new ListOfFloatArrays(matrixSize, 1);
        } else {
            System.err.println("Not supported for normalization " + normOption);
            return null;
        }

        if (norm != null && norm.getLength() > 0) {
            double factor = getSumFactor(norm);
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
        ListOfFloatArrays rowsums = new ListOfFloatArrays(matrixSize, 0);

        Iterator<ContactRecord> iterator = getIterator();
        while (iterator.hasNext()) {
            ContactRecord cr = iterator.next();
            int x = cr.getBinX();
            int y = cr.getBinY();
            float value = cr.getCounts();
            rowsums.addTo(x, value);
            if (x != y) {
                rowsums.addTo(y, value);
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
        double[] normMatrixSums = getNormMatrixSumFactor(norm);
        return Math.sqrt(normMatrixSums[0] / normMatrixSums[1]);
    }
    
    public double[] getNormMatrixSumFactor(ListOfFloatArrays norm) {
        double matrix_sum = 0;
        double norm_sum = 0;

        Iterator<ContactRecord> iterator = getIterator();
        while (iterator.hasNext()) {
            ContactRecord cr = iterator.next();
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
        return new double[]{norm_sum, matrix_sum};
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
            ListOfDoubleArrays x0 = new ListOfDoubleArrays(newSize, 1);
            
            x0 = computeKRNormVector(offset, 0.000001, x0, 0.1);

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
                kr = new ListOfFloatArrays(matrixSize);
                int krIndex = 0;
                for (int[] offsetArray : offset.getValues()) {
                    for (int offset1 : offsetArray) {
                        if (offset1 == -1) {
                            kr.set(krIndex++, Float.NaN);
                        } else {
                            kr.set(krIndex++, (float) (1.0f / x0.get(offset1)));
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
            kr = new ListOfFloatArrays(matrixSize, Float.NaN);
        }

        return kr;
    }
    
    private ListOfIntArrays getOffset(double percent) {
        ListOfDoubleArrays rowSums = new ListOfDoubleArrays(matrixSize, 0);

        Iterator<ContactRecord> iterator = getIterator();
        while (iterator.hasNext()) {
            ContactRecord cr = iterator.next();
            int x = cr.getBinX();
            int y = cr.getBinY();
            float value = cr.getCounts();
            rowSums.addTo(x, value);
            if (x != y) {
                rowSums.addTo(y, value);
            }
        }

        double thresh = 0;
        if (percent > 0) {
            // Get percent threshold from positive row sums (nonzero)
            DescriptiveStatistics stats = new DescriptiveStatistics();
            rowSums.getValues().forEach(sum -> Arrays.stream(sum).filter(i-> i != 0).forEach(stats::addValue));
            thresh = stats.getPercentile( percent);
            /*
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
             */
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
        
        ListOfFloatArrays tempTargetVector = new ListOfFloatArrays(matrixSize, 1);

        return ZeroScale.mmbaScaleToVector(ic, tempTargetVector);
    }

    /*public BigContactRecordList booleanBalancing() {
        ListOfFloatArrays rowsums = new ListOfFloatArrays(totSize, 0);
        Map<Float,Map<Long,Integer>> rowsumIndices = new HashMap<>();
        Map<Long, List<LinkedContactRecord>> rows = new HashMap<>();
        //Map<Long, RandomizedCollection> remainingContacts = new HashMap<>();
        List<Double> thresholds = new ArrayList<>(Arrays.asList(1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,3.0,4.0,5.0));

        for (List<ContactRecord> localList : contactRecords) {
            for (ContactRecord cr : localList) {
                int x = cr.getBinX();
                int y = cr.getBinY();
                float value = cr.getCounts();
                rowsums.addTo(x, value);
                if (x != y) {
                    rowsums.addTo(y, value);

                }
                List<LinkedContactRecord> row1 = rows.get((long) x);
                List<LinkedContactRecord> row2 = rows.get((long) y);
                if (row1 == null) {
                    row1 = new ArrayList<>();
                    rows.put((long) x, row1);
                    //remainingContacts.put((long) x, new RandomizedCollection());
                }
                if (row2 == null) {
                    row2 = new ArrayList<>();
                    rows.put((long) y, row2);
                    //remainingContacts.put((long) y, new RandomizedCollection());
                }
                int xCurrentSize = row1.size();
                int yCurrentSize = row2.size();
                for (int i = 0; i < value; i++) {
                    row1.add(new LinkedContactRecord(cr, yCurrentSize+i));
                    //remainingContacts.get((long) x).insert(xCurrentSize);
                    if (x != y) {
                        row2.add(new LinkedContactRecord(cr, xCurrentSize+i));
                        //remainingContacts.get((long) y).insert(yCurrentSize);
                    }
                }


            }
        }
        System.out.println("loaded contacts for matrix: " + chr1 + "-" + chr2);

        int rowSumThreshold = (int) (1.0d / getSumFactor(rowsums));
        List<Float> sortedRowSums = new ArrayList<>();
        for (long i = 0; i < rowsums.getLength(); i++) {
            //Instant E = Instant.now();
            Map<Long,Integer> sumIndexList = rowsumIndices.get(rowsums.get(i));
            if (sumIndexList == null) {
                rowsumIndices.put(rowsums.get(i), new HashMap<>());
                sumIndexList = rowsumIndices.get(rowsums.get(i));
            }
            sumIndexList.put(i,1);
            sortedRowSums.add(rowsums.get(i));
            //Instant F = Instant.now();
            //System.err.println(Duration.between(E,F).toNanos());
        }

        System.out.println("mapped row sums to rows indices for matrix: " + chr1 + "-" + chr2);


        Collections.sort(sortedRowSums);

        Map<Float,Integer> SumMap = new HashMap<>();
        List<Float> SumKeys = new ArrayList<>(rowsumIndices.keySet());
        Collections.sort(SumKeys);
        int keyCounter = 0;
        for (float key : SumKeys) {
            //Collections.sort(rowsumIndices.get(key));
            while (sortedRowSums.get(keyCounter)!=key) {
                keyCounter++;
            }
            SumMap.put(key,keyCounter);
        }

        System.out.println("sorted row sums for matrix: " + chr1 + "-" + chr2);

        Instant A = Instant.now();
        Map<Long,Integer> currentRows = rowsumIndices.get(sortedRowSums.get(sortedRowSums.size()-1));
        long currentRow = currentRows.keySet().iterator().next();
        //System.out.println(currentRows.keySet().size() + " " + sortedRowSums.get(sortedRowSums.size()-1) + " " + currentRow + " " + rowsums.get(currentRow));
        Instant B = Instant.now();
        //System.err.println(Duration.between(A,B).toNanos());
        //System.err.println(rowSumThreshold + " " + getSumFactor(rowsums) + " " + rowsums.get(rowsums.getMaxRow()));
        Random randomNumberGenerator = new Random(0);
        while (rowsums.get(currentRow) > rowSumThreshold) {
            float currentRowSum = rowsums.get(currentRow);
            //Instant C = Instant.now();
            //List<Integer> removedContactList = new ArrayList<>(removedContacts.get(currentRow));
            //Collections.sort(removedContactList);
            //int randomContact = getRandomWithExclusion(randomNumberGenerator, rows.get(currentRow).size(), removedContactList);
            //removedContacts.get(currentRow).add(randomContact);
            //int randomContact = remainingContacts.get(currentRow).getRandom();
            int randomContact = (int) (rows.get(currentRow).size() * randomNumberGenerator.nextDouble());
            //remainingContacts.get(currentRow).remove(randomContact);
            long firstRow, secondRow;
            if (rows.get(currentRow).get(randomContact).getContactRecord().getBinX() == currentRow) {
                firstRow = (long) rows.get(currentRow).get(randomContact).getContactRecord().getBinX();
                secondRow = (long) rows.get(currentRow).get(randomContact).getContactRecord().getBinY();
            } else {
                firstRow = (long) rows.get(currentRow).get(randomContact).getContactRecord().getBinY();
                secondRow = (long) rows.get(currentRow).get(randomContact).getContactRecord().getBinX();
            }
            //System.out.println(currentRow + " " + firstRow + " " + secondRow);
            float firstRowSum = rowsums.get(firstRow);
            int symmetricRandomContactPosition = rows.get(firstRow).get(randomContact).getLink();
            long lastRow = rows.get(firstRow).get(rows.get(firstRow).size()-1).getContactRecord().getBinX() == firstRow? rows.get(firstRow).get(rows.get(firstRow).size()-1).getContactRecord().getBinY() : rows.get(firstRow).get(rows.get(firstRow).size()-1).getContactRecord().getBinX();
            int lastLink = rows.get(firstRow).get(rows.get(firstRow).size()-1).getLink();
            //System.out.println("initial values " + currentRow + " " + rows.get(firstRow).get(randomContact).getContactRecord().getBinX() + " " + rows.get(firstRow).get(randomContact).getContactRecord().getBinY() + " " + rows.get(firstRow).get(randomContact).getLink() +  " " + rows.get(secondRow).get(symmetricRandomContactPosition).getContactRecord().getBinX() + " " + rows.get(secondRow).get(symmetricRandomContactPosition).getContactRecord().getBinY() + " " + rows.get(secondRow).get(symmetricRandomContactPosition).getLink() + " " + rows.get(firstRow).get(rows.get(firstRow).size()-1).getContactRecord().getBinX() + " " + rows.get(firstRow).get(rows.get(firstRow).size()-1).getContactRecord().getBinY() + " " + rows.get(firstRow).get(rows.get(firstRow).size()-1).getLink() + " " + rows.get(lastRow).get(lastLink).getContactRecord().getBinX() + " " + rows.get(lastRow).get(lastLink).getContactRecord().getBinY() + " " + rows.get(lastRow).get(lastLink).getLink());
            rows.get(firstRow).get(randomContact).getContactRecord().incrementCount(-1);
            rows.get(firstRow).set(randomContact, rows.get(firstRow).get(rows.get(firstRow).size()-1));
            //System.out.println("first row swapped " + currentRow + " " + rows.get(firstRow).get(randomContact).getContactRecord().getBinX() + " " + rows.get(firstRow).get(randomContact).getContactRecord().getBinY() + " " + rows.get(firstRow).get(randomContact).getLink() +  " " + rows.get(secondRow).get(symmetricRandomContactPosition).getContactRecord().getBinX() + " " + rows.get(secondRow).get(symmetricRandomContactPosition).getContactRecord().getBinY() + " " + rows.get(secondRow).get(symmetricRandomContactPosition).getLink() + " " + rows.get(firstRow).get(rows.get(firstRow).size()-1).getContactRecord().getBinX() + " " + rows.get(firstRow).get(rows.get(firstRow).size()-1).getContactRecord().getBinY() + " " + rows.get(firstRow).get(rows.get(firstRow).size()-1).getLink() + " " + rows.get(lastRow).get(lastLink).getContactRecord().getBinX() + " " + rows.get(lastRow).get(lastLink).getContactRecord().getBinY() + " " + rows.get(lastRow).get(lastLink).getLink());

            long partnerRow;
            int partnerLink = rows.get(firstRow).get(randomContact).getLink();
            if (rows.get(firstRow).get(randomContact).getContactRecord().getBinX() == firstRow) {
                partnerRow = (long) rows.get(firstRow).get(randomContact).getContactRecord().getBinY();
            } else {
                partnerRow = (long) rows.get(firstRow).get(randomContact).getContactRecord().getBinX();
            }
            //System.out.println(firstRow + " " + rows.get(firstRow).get(randomContact).getContactRecord().getBinY() + " " + rows.get(firstRow).get(randomContact).getContactRecord().getBinX() + " " + partnerRow + " " + partnerLink + " " + symmetricRandomContactPosition);
            rows.get(partnerRow).get(partnerLink).setLink(randomContact);
            //System.out.println("partner link updated " + currentRow + " " + rows.get(firstRow).get(randomContact).getContactRecord().getBinX() + " " + rows.get(firstRow).get(randomContact).getContactRecord().getBinY() + " " + rows.get(firstRow).get(randomContact).getLink() +  " " + rows.get(secondRow).get(symmetricRandomContactPosition).getContactRecord().getBinX() + " " + rows.get(secondRow).get(symmetricRandomContactPosition).getContactRecord().getBinY() + " " + rows.get(secondRow).get(symmetricRandomContactPosition).getLink() + " " + rows.get(firstRow).get(rows.get(firstRow).size()-1).getContactRecord().getBinX() + " " + rows.get(firstRow).get(rows.get(firstRow).size()-1).getContactRecord().getBinY() + " " + rows.get(firstRow).get(rows.get(firstRow).size()-1).getLink() + " " + rows.get(lastRow).get(lastLink).getContactRecord().getBinX() + " " + rows.get(lastRow).get(lastLink).getContactRecord().getBinY() + " " + rows.get(lastRow).get(lastLink).getLink());
            rows.get(firstRow).remove(rows.get(firstRow).size()-1);
            //System.out.println("first row removed " + currentRow + " " + rows.get(firstRow).get(Math.min(randomContact,rows.get(firstRow).size()-1)).getContactRecord().getBinX() + " " + rows.get(firstRow).get(Math.min(randomContact,rows.get(firstRow).size()-1)).getContactRecord().getBinY() + " " + rows.get(firstRow).get(Math.min(randomContact,rows.get(firstRow).size()-1)).getLink() +  " " + rows.get(secondRow).get(symmetricRandomContactPosition).getContactRecord().getBinX() + " " + rows.get(secondRow).get(symmetricRandomContactPosition).getContactRecord().getBinY() + " " + rows.get(secondRow).get(symmetricRandomContactPosition).getLink() + " " + rows.get(firstRow).get(rows.get(firstRow).size()-1).getContactRecord().getBinX() + " " + rows.get(firstRow).get(rows.get(firstRow).size()-1).getContactRecord().getBinY() + " " + rows.get(firstRow).get(rows.get(firstRow).size()-1).getLink() + " " + rows.get(lastRow).get(lastLink).getContactRecord().getBinX() + " " + rows.get(lastRow).get(lastLink).getContactRecord().getBinY() + " " + rows.get(lastRow).get(lastLink).getLink());

            rowsums.addTo(firstRow, -1);
            float newFirstRowSum = rowsums.get(firstRow);
            //System.out.println(currentRowSum + " " + rowsums.get(currentRow) + " " + sortedRowSums.get(sortedRowSums.size()-1) );
            rowsumIndices.get(currentRowSum).remove(currentRow);
            Map<Long,Integer> newRow1 = rowsumIndices.get(newFirstRowSum);
            if (newRow1 == null) {
                rowsumIndices.put(newFirstRowSum, new HashMap<>());
                newRow1 = rowsumIndices.get(newFirstRowSum);
                newRow1.put(firstRow,1);
            } else {
                newRow1.put(firstRow,1);
            }
            int switchPlace1 = SumMap.get(firstRowSum);
            float potentialSwitchSum1 = sortedRowSums.get(switchPlace1);
            if (switchPlace1 == sortedRowSums.size()-1) {
                sortedRowSums.set(switchPlace1, newFirstRowSum);
                if (SumMap.get(newFirstRowSum) == null) {
                    SumMap.put(newFirstRowSum, switchPlace1);
                }
            } else {
                sortedRowSums.set(sortedRowSums.size()-1, potentialSwitchSum1);
                sortedRowSums.set(switchPlace1, newFirstRowSum);
                SumMap.put(firstRowSum, switchPlace1+1);
                if (SumMap.get(newFirstRowSum) == null) {
                    SumMap.put(newFirstRowSum, switchPlace1);
                }
            }
            //System.out.println(currentRowSum + " " + rowsums.get(currentRow) + " " + sortedRowSums.get(sortedRowSums.size()-1) );

            if (firstRow != secondRow) {
                float secondRowSum = rowsums.get(secondRow);
                //removedContacts.get(secondRow).add(symmetricRandomContactPosition);
                //remainingContacts.get(secondRow).remove(symmetricRandomContactPosition);
                lastRow = rows.get(secondRow).get(rows.get(secondRow).size()-1).getContactRecord().getBinX() == secondRow? rows.get(secondRow).get(rows.get(secondRow).size()-1).getContactRecord().getBinY() : rows.get(secondRow).get(rows.get(secondRow).size()-1).getContactRecord().getBinX();
                lastLink = rows.get(secondRow).get(rows.get(secondRow).size()-1).getLink();
                //System.out.println("second row last check " + currentRow + " " + rows.get(secondRow).get(symmetricRandomContactPosition).getContactRecord().getBinX() + " " + rows.get(secondRow).get(symmetricRandomContactPosition).getContactRecord().getBinY() + " " + rows.get(secondRow).get(symmetricRandomContactPosition).getLink() + " " + rows.get(secondRow).get(rows.get(secondRow).size()-1).getContactRecord().getBinX() + " " + rows.get(secondRow).get(rows.get(secondRow).size()-1).getContactRecord().getBinY() + " " + rows.get(secondRow).get(rows.get(secondRow).size()-1).getLink() + " " + rows.get(lastRow).get(lastLink).getContactRecord().getBinX() + " " + rows.get(lastRow).get(lastLink).getContactRecord().getBinY() + " " + rows.get(lastRow).get(lastLink).getLink());

                rows.get(secondRow).set(symmetricRandomContactPosition, rows.get(secondRow).get(rows.get(secondRow).size()-1));
                //System.out.println("second row swapped " + currentRow + " " + rows.get(secondRow).get(symmetricRandomContactPosition).getContactRecord().getBinX() + " " + rows.get(secondRow).get(symmetricRandomContactPosition).getContactRecord().getBinY() + " " + rows.get(secondRow).get(symmetricRandomContactPosition).getLink() + " " + rows.get(secondRow).get(rows.get(secondRow).size()-1).getContactRecord().getBinX() + " " + rows.get(secondRow).get(rows.get(secondRow).size()-1).getContactRecord().getBinY() + " " + rows.get(secondRow).get(rows.get(secondRow).size()-1).getLink() + " " + rows.get(lastRow).get(lastLink).getContactRecord().getBinX() + " " + rows.get(lastRow).get(lastLink).getContactRecord().getBinY() + " " + rows.get(lastRow).get(lastLink).getLink());

                partnerLink = rows.get(secondRow).get(symmetricRandomContactPosition).getLink();
                if (rows.get(secondRow).get(symmetricRandomContactPosition).getContactRecord().getBinX() == secondRow) {
                    partnerRow = (long) rows.get(secondRow).get(symmetricRandomContactPosition).getContactRecord().getBinY();
                } else {
                    partnerRow = (long) rows.get(secondRow).get(symmetricRandomContactPosition).getContactRecord().getBinX();
                }
                if (symmetricRandomContactPosition!=rows.get(secondRow).size()-1) {
                    rows.get(partnerRow).get(partnerLink).setLink(symmetricRandomContactPosition);
                }
                //System.out.println("partner link updated " + currentRow + " " + rows.get(secondRow).get(symmetricRandomContactPosition).getContactRecord().getBinX() + " " + rows.get(secondRow).get(symmetricRandomContactPosition).getContactRecord().getBinY() + " " + rows.get(secondRow).get(symmetricRandomContactPosition).getLink() + " " + rows.get(secondRow).get(rows.get(secondRow).size()-1).getContactRecord().getBinX() + " " + rows.get(secondRow).get(rows.get(secondRow).size()-1).getContactRecord().getBinY() + " " + rows.get(secondRow).get(rows.get(secondRow).size()-1).getLink() + " " + rows.get(lastRow).get(lastLink).getContactRecord().getBinX() + " " + rows.get(lastRow).get(lastLink).getContactRecord().getBinY() + " " + rows.get(lastRow).get(lastLink).getLink());

                rows.get(secondRow).remove(rows.get(secondRow).size()-1);
                if (rows.get(secondRow).size()>0) {
                    //System.out.println("second row removed " + currentRow + " " + rows.get(secondRow).get(Math.min(symmetricRandomContactPosition, rows.get(secondRow).size() - 1)).getContactRecord().getBinX() + " " + rows.get(secondRow).get(Math.min(symmetricRandomContactPosition, rows.get(secondRow).size() - 1)).getContactRecord().getBinY() + " " + rows.get(secondRow).get(Math.min(symmetricRandomContactPosition, rows.get(secondRow).size() - 1)).getLink() + " " + rows.get(secondRow).get(rows.get(secondRow).size() - 1).getContactRecord().getBinX() + " " + rows.get(secondRow).get(rows.get(secondRow).size() - 1).getContactRecord().getBinY() + " " + rows.get(secondRow).get(rows.get(secondRow).size() - 1).getLink() + " " + rows.get(lastRow).get(lastLink).getContactRecord().getBinX() + " " + rows.get(lastRow).get(lastLink).getContactRecord().getBinY() + " " + rows.get(lastRow).get(lastLink).getLink());
                }
                rowsums.addTo(secondRow, -1);
                //System.out.println(currentRowSum + " " + rowsums.get(currentRow) + " " + sortedRowSums.get(sortedRowSums.size()-1) + " " + secondRowSum );
                float newSecondRowSum = rowsums.get(secondRow);
                //int removeIndex2 = Collections.binarySearch(rowsumIndices.get(secondRowSum),secondRow);
                rowsumIndices.get(secondRowSum).remove(secondRow);
                Map<Long,Integer> newRow2 = rowsumIndices.get(newSecondRowSum);
                if (newRow2 == null) {
                    rowsumIndices.put(newSecondRowSum, new HashMap<>());
                    newRow2 = rowsumIndices.get(newSecondRowSum);
                    newRow2.put(secondRow,1);
                } else {
                    newRow2.put(secondRow,1);
                }
                int switchPlace2 = SumMap.get(secondRowSum);
                if (switchPlace2 == 0) {
                    sortedRowSums.set(0,newSecondRowSum);
                    if (SumMap.get(newSecondRowSum) == null) {
                        SumMap.put(newSecondRowSum,0);
                    }
                    if (sortedRowSums.get(1)==secondRowSum) {
                        SumMap.put(secondRowSum,1);
                    } else {
                        SumMap.remove(secondRowSum);
                    }
                } else {
                    sortedRowSums.set(switchPlace2, newSecondRowSum);
                    if (SumMap.get(newSecondRowSum) == null) {
                        SumMap.put(newSecondRowSum,switchPlace2);
                    }
                    if (sortedRowSums.size() == switchPlace2+1) {
                        SumMap.remove(secondRowSum);
                    } else if (sortedRowSums.get(switchPlace2+1) == secondRowSum) {
                        SumMap.put(secondRowSum, switchPlace2+1);
                    } else {
                        SumMap.remove(secondRowSum);
                    }
                }
            }
            //System.out.println(currentRowSum + " " + rowsums.get(currentRow) + " " + sortedRowSums.get(sortedRowSums.size()-1) );
            currentRows = rowsumIndices.get(sortedRowSums.get(sortedRowSums.size()-1));
            currentRow = currentRows.keySet().iterator().next();
            double maxRatio = (rowsums.get(currentRow)*1.0d)/rowSumThreshold;
            if (thresholds.size()>0 && maxRatio < thresholds.get(thresholds.size()-1)) {
                double passedThreshold = thresholds.get(thresholds.size()-1);
                while (thresholds.size()>0 && maxRatio < thresholds.get(thresholds.size()-1)) {
                    passedThreshold = thresholds.remove(thresholds.size()-1);
                }
                System.out.println("passed threshold: " + passedThreshold + " for matrix: " + chr1 + "-" + chr2 + "(current max sum: " + rowsums.get(currentRow) + " , sum threshold: " + rowSumThreshold + ")");
            }
            //Instant D = Instant.now();
            //System.err.println(Duration.between(C,D).toMillis());
        }

        return contactRecords;

    }*/

    public int getRandomWithExclusion(Random rnd, int end, List<Integer> exclude) {
        int random = 0;
        try {
            random = rnd.nextInt(end - exclude.size());
        } catch (Exception e) {
            System.err.println(end + " " + exclude.size());
            e.printStackTrace();
        }
        for (int ex : exclude) {
            if (random < ex) {
                break;
            }
            random++;
        }
        return random;
    }
}