/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2023 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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

package juicebox.tools.utils.juicer.localizer;

import juicebox.HiCGlobals;
import juicebox.data.Dataset;
import juicebox.data.HiCFileTools;
import juicebox.data.MatrixZoomData;
import juicebox.data.NormalizationVector;
import juicebox.data.basics.ListOfDoubleArrays;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.track.feature.Feature2D;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.distribution.PoissonDistribution;
import org.apache.commons.math.distribution.PoissonDistributionImpl;
import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.RealMatrix;

import java.io.File;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class LocalizerUtils {

    /**
     * Size filtering of loops
     *
     * @param features
     * @param minPeakDist
     * @param maxPeakDist
     * @return
     */
    public static ArrayList<Feature2D> filterFeaturesBySize(List<Feature2D> features,
                                                            double minPeakDist, double maxPeakDist, int resolution) {
        ArrayList<Feature2D> sizeFilteredFeatures = new ArrayList<>();

        for (Feature2D feature : features) {
            double xMidPt = feature.getMidPt1();
            double yMidPt = feature.getMidPt2();
            int dist = (int) Math.round(Math.abs(xMidPt - yMidPt) / resolution);

            if (dist >= minPeakDist) {
                if (dist <= maxPeakDist) {
                    sizeFilteredFeatures.add(feature);
                }
            }
        }
        return new ArrayList<>(sizeFilteredFeatures);
    }

    public static RealMatrix extractLocalizedData(MatrixZoomData zd, Feature2D loop,
                                                  int L, int resolution, int window, NormalizationType norm) throws IOException {
        long loopX = loop.getMidPt1() / resolution;
        long loopY = loop.getMidPt2() / resolution;
        long binXStart = loopX - window;
        long binXEnd = loopX + (window + 1);
        long binYStart = loopY - window;
        long binYEnd = loopY + (window + 1);

        return HiCFileTools.extractLocalBoundedRegion(zd, binXStart, binXEnd, binYStart, binYEnd, L, L, norm, false);
    }

    public static List<List<Double>> extractNormParts(Dataset ds, MatrixZoomData zd, Feature2D loop, int resolution, int window, NormalizationType norm) throws IOException {
        long loopX = loop.getMidPt1() / resolution;
        long loopY = loop.getMidPt2() / resolution;
        long binXStart = loopX - window;
        long binXEnd = loopX + (window + 1);
        long binYStart = loopY - window;
        long binYEnd = loopY + (window + 1);
        List<Double> normH = new ArrayList<>();
        List<Double> normV = new ArrayList<>();
        NormalizationVector nv1 = ds.getNormalizationVector(zd.getChr1Idx(), zd.getZoom(), norm);
        NormalizationVector nv2 = ds.getNormalizationVector(zd.getChr2Idx(), zd.getZoom(), norm);
        ListOfDoubleArrays nv1Data = nv1.getData();
        ListOfDoubleArrays nv2Data = nv2.getData();
        for (long i = binXStart; i < binXEnd; i++) {
            normH.add(nv1Data.get(i));
        }
        for (long i = binYStart; i < binYEnd; i++) {
            normV.add(nv2Data.get(i));
        }
        List<List<Double>> normParts = new ArrayList<>();
        normParts.add(normH);
        normParts.add(normV);
        return normParts;
    }

    public static List<List<Double>> localMax(RealMatrix rawSource, List<Double> normH, List<Double> normV, int window, int numPeaks, double maxPval, File outputDirectory) {

        // calculate total number of contacts in local window
        double rawSourceSum = matrixSum(rawSource, 0, rawSource.getRowDimension(), 0, rawSource.getColumnDimension());
        //MatrixTools.saveMatrixText(new File(outputDirectory, "inputData.txt").getPath(), rawSource);

        // create copy of raw data and box blur over requested window (rawSummedData)
        // multiply box blurred matrix by window size to bring back into contact space (rawSummedData)
        Instant A = Instant.now();
        RealMatrix rawSourceCopy = rawSource.copy();
        RealMatrix rawSummedData = new Array2DRowRealMatrix(rawSource.getRowDimension(), rawSource.getColumnDimension());
        boxBlur(rawSourceCopy, rawSummedData, window);
        multiplyInPlaceRound(rawSummedData, (window + window + 1) * (window + window + 1));
        Instant B = Instant.now();
        if (HiCGlobals.printVerboseComments) {
            System.out.println("time to blur raw matrix: " + Duration.between(A, B).toMillis());
        }
        // calculate normed local matrix using input local norm vectors (normSource)
        // calculate gaussian blurred normed local matrix over requested window (normBlurred), multiply in place by ratio of box blurred raw matrix to normed matrix in order to make sure same number of contacts
        RealMatrix normSource = calculateNormMatrix(rawSource, normH, normV);
        Instant C = Instant.now();
        if (HiCGlobals.printVerboseComments) {
            System.out.println("time to calculate norm matrix: " + Duration.between(B, C).toMillis());
        }
        //MatrixTools.saveMatrixText(new File(outputDirectory, "normInputData.txt").getPath(), normSource);
        RealMatrix normBlurred = gaussBlur(normSource, window); //new Array2DRowRealMatrix(rawSource.getRowDimension(), rawSource.getColumnDimension());
        double rawSummedSum = matrixSum(rawSummedData, 0, rawSource.getRowDimension(), 0, rawSource.getColumnDimension());
        double normBlurredSum = matrixSum(normBlurred, 0, rawSource.getRowDimension(), 0, rawSource.getColumnDimension());
        multiplyInPlace(normBlurred, rawSummedSum / normBlurredSum);
        Instant D = Instant.now();
        if (HiCGlobals.printVerboseComments) {
            System.out.println("time to blur norm matrix: " + Duration.between(C, D).toMillis());
        }

        //MatrixTools.saveMatrixText(new File(outputDirectory, "summedData.txt").getPath(), rawSummedData);
        //MatrixTools.saveMatrixText(new File(outputDirectory, "normSummedData.txt").getPath(), normBlurred);

        // set a non-uniform expected based on ratio of raw data to normed data (normExpected)
        // multiply in place to make sure it has the same number of contacts as the boxBlurred raw matrix
        RealMatrix normExpected = setNormExpected(rawSummedData, normBlurred, normH, normV);
        double normExpectedSum = matrixSum(normExpected, 0, rawSource.getRowDimension(), 0, rawSource.getColumnDimension());
        multiplyInPlace(normExpected, rawSummedSum / normExpectedSum);
        Instant E = Instant.now();
        if (HiCGlobals.printVerboseComments) {
            System.out.println("time to set expected: " + Duration.between(D, E).toMillis());
        }
        //MatrixTools.saveMatrixText(new File(outputDirectory, "normExpectedData.txt").getPath(), normExpected);

        //do local nonmaximum suppression using the Poisson Z-score [2*(sqrt(X)-sqrt(lambda))]
        RealMatrix localMaxData = nonmaxsuppressPoissonZ(rawSummedData, normExpected, window);
        Instant F = Instant.now();
        if (HiCGlobals.printVerboseComments) {
            System.out.println("time to suppress non-max: " + Duration.between(E, F).toMillis());
        }
        //MatrixTools.saveMatrixText(new File(outputDirectory, "localMaxData.txt").getPath(), localMaxData);

        // count non-zero entries for bonferroni correction
        int pValAdj = nonZeroEntries(localMaxData, window, rawSource.getRowDimension() - window, window, rawSource.getColumnDimension() - window);

        // rank localized peaks
        List<List<Double>> orderedPeaks = orderPeaksPoissonZ(localMaxData, normExpected, window, maxPval, pValAdj);
        Instant G = Instant.now();
        if (HiCGlobals.printVerboseComments) {
            System.out.println("time to identify peaks: " + Duration.between(F, G).toMillis());
        }

        // return localized peaks
        List<Double> finalPeaksR = new ArrayList<>();
        List<Double> finalPeaksC = new ArrayList<>();
        List<Double> finalPeaksP = new ArrayList<>();
        List<Double> finalPeaksO = new ArrayList<>();
        List<Double> finalPeaksZ = new ArrayList<>();
        numPeaks = numPeaks > orderedPeaks.get(0).size() ? orderedPeaks.get(0).size() : numPeaks;
        for (int i = 0; i < numPeaks; i++) {
            double peakRow = orderedPeaks.get(0).get(i);
            double peakCol = orderedPeaks.get(1).get(i);
            double peakPVal = orderedPeaks.get(2).get(i);
            double peakObserved = orderedPeaks.get(3).get(i);
            double peakZscore = orderedPeaks.get(4).get(i);
            //System.out.println(peakRow + " " + peakCol + " " + peakEntry);
            finalPeaksR.add(peakRow);
            finalPeaksC.add(peakCol);
            finalPeaksP.add(peakPVal);
            finalPeaksO.add(peakObserved);
            finalPeaksZ.add(peakZscore);
        }
        List<List<Double>> finalPeaks = new ArrayList<>();
        finalPeaks.add(finalPeaksR);
        finalPeaks.add(finalPeaksC);
        finalPeaks.add(finalPeaksP);
        finalPeaks.add(finalPeaksO);
        finalPeaks.add(finalPeaksZ);
        return finalPeaks;

    }

    public static RealMatrix calculateNormMatrix(RealMatrix inputData, List<Double> normH, List<Double> normV) {
        int numRows = inputData.getRowDimension();
        int numCols = inputData.getColumnDimension();
        Instant A = Instant.now();
        RealMatrix normSource = new Array2DRowRealMatrix(numRows, numCols);
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                double valI = normH.get(i);
                double valJ = normV.get(j);
                // todo == 0 probably not the best thing to do here
                if (valI != 0 && valJ != 0 && !Double.isNaN(valI) && !Double.isNaN(valJ)) {
                    normSource.setEntry(i,j,inputData.getEntry(i,j) / (valI * valJ));
                } else {
                    normSource.setEntry(i,j,0);
                }
            }
        }
        return normSource;
    }

    public static RealMatrix setNormExpected(RealMatrix rawSummedData, RealMatrix normBlurred, List<Double> normH, List<Double> normV) {
        int numRows = rawSummedData.getRowDimension();
        int numCols = rawSummedData.getColumnDimension();
        Instant A = Instant.now();
        RealMatrix normExpected = new Array2DRowRealMatrix(numRows, numCols);
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                if (normBlurred.getEntry(i,j)<=0) {
                    normBlurred.setEntry(i,j,0);
                    rawSummedData.setEntry(i,j,0);
                }
                double valI = normH.get(i);
                double valJ = normV.get(j);
                // todo == 0 probably not the best thing to do here
                if (normBlurred.getEntry(i,j)!=0) {
                    normExpected.setEntry(i,j,(rawSummedData.getEntry(i,j)/normBlurred.getEntry(i,j)));
                    //System.out.println(rawSummedData.getEntry(i,j) + " " + normBlurred.getEntry(i,j) + " " + normExpected.getEntry(i,j));
                } else {
                    normExpected.setEntry(i,j,0);
                }
            }
        }
        return normExpected;
    }

    public static double matrixSum(RealMatrix inputData, int rowBound1, int rowBound2, int colBound1, int colBound2) {
        double inputDataSum = 0;
        for (int i = rowBound1; i < rowBound2; i++) {
            for (int j = colBound1; j < colBound2; j++) {
                inputDataSum += inputData.getEntry(i,j);
            }
        }
        return inputDataSum;
    }

    public static int nonZeroEntries(RealMatrix inputData, int rowBound1, int rowBound2, int colBound1, int colBound2) {
        int nonZeroEntries = 0;
        for (int i = rowBound1; i < rowBound2; i++) {
            for (int j = colBound1; j < colBound2; j++) {
                if (inputData.getEntry(i,j)>0) {
                    nonZeroEntries++;
                }
            }
        }
        return nonZeroEntries;
    }

    public static void multiplyInPlace(RealMatrix inputData, double scaleRatio) {
        int numRows = inputData.getRowDimension();
        int numCols = inputData.getColumnDimension();
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                inputData.multiplyEntry(i,j,scaleRatio);
            }
        }
    }

    public static void multiplyInPlaceRound(RealMatrix inputData, double scaleRatio) {
        int numRows = inputData.getRowDimension();
        int numCols = inputData.getColumnDimension();
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                double val = inputData.getEntry(i,j);
                double multVal = (double) Math.round(val*scaleRatio);
                inputData.setEntry(i,j,multVal);
            }
        }
    }

    public static RealMatrix gaussBlur(RealMatrix inputData, int window) {
        //int numRows = inputData.getRowDimension() - window;
        //int numCols = inputData.getColumnDimension() - window;
        //RealMatrix outputData = MatrixTools.cleanArray2DMatrix(numRows, numCols);
        //RealMatrix outputData = MatrixTools.deepCopy(inputData);
        int numRows = inputData.getRowDimension();
        int numCols = inputData.getColumnDimension();
        RealMatrix outputData = new Array2DRowRealMatrix(numRows, numCols);
        int[] boxSizes = boxesForGauss(window, 3);
        boxBlur(inputData, outputData, (boxSizes[0]-1)/2);
        boxBlur(outputData, inputData, (boxSizes[1]-1)/2);
        boxBlur(inputData, outputData, (boxSizes[2]-1)/2);
        //double outputDataSum = 0;
        //for (int i = window/2; i < (numRows-window/2); i++) {
        //    for (int j = window/2; j < (numCols-window/2); j++) {
        //        double sumVal = 0;
        //        for (int rowOffset = 0; rowOffset < window; rowOffset++) {
        //            for (int colOffset = 0; colOffset < window; colOffset++) {
        //                sumVal = sumVal + (inputData.getEntry(i - (int) window/2 + rowOffset, j - (int) window/2 + colOffset) *
        //Math.exp(-1*(Math.pow((0-window/2+rowOffset),2)+Math.pow((0-window/2+colOffset),2))/(2*Math.pow(window/3,2)))*(1.0d/(2*Math.PI*Math.pow(window/3,2))));
        //            }
        //        }
        //        outputData.setEntry(i-(window/2),j-(window/2), sumVal);
        //        outputDataSum += sumVal;
        //    }
        //}


        return outputData;
    }

    public static int[] boxesForGauss(int sigma, int numBoxes) {
        int wl = (int) Math.floor(Math.sqrt((12*sigma*sigma/numBoxes)+1));
        if (wl % 2 == 0) {
            wl--;
        }
        int wu = wl+2;
        int m = Math.round(((12*sigma*sigma) - (numBoxes*wl*wl) - (4*numBoxes*wl) - (3*numBoxes))/((-4*wl) - 4));
        int[] boxSizes = new int[numBoxes];
        for (int i = 0; i < numBoxes; i++ ) {
            boxSizes[i] = i<m? wl: wu;
        }
        return boxSizes;

    }

    public static void boxBlur(RealMatrix inputData, RealMatrix outputData, int radius) {
        int numRows = inputData.getRowDimension();
        int numCols = inputData.getColumnDimension();
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                outputData.setEntry(i,j,inputData.getEntry(i,j));
            }
        }
        boxBlurH(outputData, inputData, radius);
        boxBlurT(inputData, outputData, radius);
    }

    public static void boxBlurH(RealMatrix source, RealMatrix target, int radius) {
        double iarr = 1.0 / (radius+radius+1);
        int numRows = source.getRowDimension();
        int numCols = source.getColumnDimension();
        for (int i = 0; i < numRows; i++) {
            int ti = 0;
            int li = 0;
            int ri = ti + radius;
            double fv = source.getEntry(i,0);
            double lv = source.getEntry(i,numCols-1);
            double val = (radius+1)*fv;
            for (int j = 0; j < radius; j++) {
                val += source.getEntry(i,j);
            }
            for (int j = 0; j <= radius; j++) {
                val += source.getEntry(i, ri) - fv;
                ri++;
                target.setEntry(i, ti, val * iarr);
                ti++;
            }
            for (int j = radius + 1; j < numCols - radius; j++) {
                val += source.getEntry(i,ri) - source.getEntry(i, li);
                ri++; li++;
                target.setEntry(i, ti, val * iarr);
                ti++;
            }
            for (int j = numCols - radius; j < numCols; j++) {
                val += lv - source.getEntry(i, li);
                li++;
                target.setEntry(i, ti, val * iarr);
                ti++;
            }
        }
    }

    public static void boxBlurT(RealMatrix source, RealMatrix target, int radius) {
        double iarr = 1.0 / (radius+radius+1);
        int numRows = source.getRowDimension();
        int numCols = source.getColumnDimension();
        for (int i = 0; i < numCols; i++) {
            int ti = 0;
            int li = 0;
            int ri = ti + radius;
            double fv = source.getEntry(0,i);
            double lv = source.getEntry(numRows-1,i);
            double val = (radius+1)*fv;
            for (int j = 0; j < radius; j++) {
                val += source.getEntry(j,i);
            }
            for (int j = 0; j <= radius; j++) {
                val += source.getEntry(ri, i) - fv;
                ri++;
                target.setEntry(ti, i, val * iarr);
                ti++;
            }
            for (int j = radius + 1; j < numCols - radius; j++) {
                val += source.getEntry(ri,i) - source.getEntry(li, i);
                ri++; li++;
                target.setEntry(ti, i, val * iarr);
                ti++;
            }
            for (int j = numCols - radius; j < numCols; j++) {
                val += lv - source.getEntry(li, i);
                li++;
                target.setEntry(ti, i, val * iarr);
                ti++;
            }
        }
    }

    public static RealMatrix localGradient(RealMatrix inputData, int window) {
        int numRows = inputData.getRowDimension() - (2*window);
        int numCols = inputData.getColumnDimension() - (2*window);
        RealMatrix outputData = MatrixTools.cleanArray2DMatrix(numRows, numCols);
        for (int i = window; i < numRows-window; i++) {
            for (int j = window; j < numCols-window; j++) {
                double outsideAverage = (inputData.getEntry(i-window, j) + inputData.getEntry(i-window,j-window) +
                        inputData.getEntry(i, j-window) + inputData.getEntry(i+window, j-window) + inputData.getEntry(i+window, j) +
                        inputData.getEntry(i+window, j+window) + inputData.getEntry(i, j+window) + inputData.getEntry(i-window, j+window))/8.0;
                double gradientVal = (inputData.getEntry(i,j) - outsideAverage)/Math.sqrt(inputData.getEntry(i,j));
                if (gradientVal > 0) {
                    outputData.setEntry(i-window,j-window,gradientVal);
                }
            }

        }
        return outputData;
    }

    public static RealMatrix nonmaxsuppress(RealMatrix inputData, int window) {
        int numRows = inputData.getRowDimension();
        int numCols = inputData.getColumnDimension();
        RealMatrix outputData = MatrixTools.cleanArray2DMatrix(numRows, numCols);
        Random randomGenerator = new Random(0);
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                if (inputData.getEntry(i,j) > 0 && outputData.getEntry(i,j) >= 0) {
                    int minRow = Math.max(0, i - window);
                    int minCol = Math.max(0, j - window);
                    int maxRow = Math.min(i + window, numRows - 1);
                    int maxCol = Math.min(j + window, numCols - 1);
                    int maxValR = i;
                    int maxValC = j;
                    double maxVal = inputData.getEntry(i, j);
                    outputData.setEntry(maxValR, maxValC, inputData.getEntry(i, j));
                    for (int nmsRow = minRow; nmsRow <= maxRow; nmsRow++) {
                        for (int nmsCol = minCol; nmsCol <= maxCol; nmsCol++) {
                            if (nmsRow != i || nmsCol != j) {
                                if (inputData.getEntry(nmsRow, nmsCol) > maxVal) {
                                    outputData.setEntry(maxValR, maxValC, -1);
                                } else if (inputData.getEntry(nmsRow, nmsCol) == maxVal) {
                                    float coinToss = randomGenerator.nextFloat();
                                    if (coinToss < 0.5) {
                                        outputData.setEntry(maxValR, maxValC, -1);
                                    } else {
                                        outputData.setEntry(nmsRow, nmsCol, -1);
                                    }
                                } else {
                                    outputData.setEntry(nmsRow, nmsCol, -1);
                                }
                            }
                        }
                    }
                }
            }
        }
        return outputData;
    }

    public static RealMatrix nonmaxsuppressPoissonZ(RealMatrix observedData, RealMatrix expectedData, int window) {
        int numRows = observedData.getRowDimension();
        int numCols = observedData.getColumnDimension();
        RealMatrix outputData = MatrixTools.cleanArray2DMatrix(numRows, numCols);
        Random randomGenerator = new Random(0);
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                if (observedData.getEntry(i,j) > 0 && outputData.getEntry(i,j) >= 0) {
                    int minRow = Math.max(0, i - window);
                    int minCol = Math.max(0, j - window);
                    int maxRow = Math.min(i + window, numRows - 1);
                    int maxCol = Math.min(j + window, numCols - 1);
                    int maxValR = i;
                    int maxValC = j;
                    double maxVal = 2 * (Math.sqrt(observedData.getEntry(i, j)) - Math.sqrt(expectedData.getEntry(i, j)));
                    outputData.setEntry(maxValR, maxValC, observedData.getEntry(i, j));
                    for (int nmsRow = minRow; nmsRow <= maxRow; nmsRow++) {
                        for (int nmsCol = minCol; nmsCol <= maxCol; nmsCol++) {
                            if (nmsRow != i || nmsCol != j) {
                                double testVal = 2 * (Math.sqrt(observedData.getEntry(nmsRow, nmsCol)) - Math.sqrt(expectedData.getEntry(nmsRow, nmsCol)));
                                if (testVal > maxVal) {
                                    outputData.setEntry(maxValR, maxValC, -1);
                                } else if (testVal == maxVal) {
                                    float coinToss = randomGenerator.nextFloat();
                                    if (coinToss < 0.5) {
                                        outputData.setEntry(maxValR, maxValC, -1);
                                    } else {
                                        outputData.setEntry(nmsRow, nmsCol, -1);
                                    }
                                } else {
                                    outputData.setEntry(nmsRow, nmsCol, -1);
                                }
                            }
                        }
                    }
                }
            }
        }
        return outputData;
    }

    public static List<List<Double>> orderPeaks(RealMatrix inputData, double sourceSum, List<Double> normH, List<Double> normV, int window, double maxPval, int pValAdj) {
        List<List<Double>> peaks = new ArrayList<>();
        peaks.add(new ArrayList<>());
        peaks.add(new ArrayList<>());
        peaks.add(new ArrayList<>());
        peaks.add(new ArrayList<>());
        int numRows = inputData.getRowDimension();
        int numCols = inputData.getColumnDimension();
        Instant A = Instant.now();
        RealMatrix normExpected = new Array2DRowRealMatrix(numRows, numCols);
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                normExpected.setEntry(i,j, sourceSum*normH.get(i)*normV.get(j)/(numRows*numCols));
            }
        }
        Instant B = Instant.now();
        double normSum = matrixSum(normExpected, 0, numRows, 0, numCols);
        Instant C = Instant.now();
        multiplyInPlace(normExpected, sourceSum/normSum);
        Instant D = Instant.now();
        RealMatrix normSummed = new Array2DRowRealMatrix(numRows, numCols);
        boxBlur(normExpected, normSummed, window);
        Instant E = Instant.now();
        multiplyInPlace(normSummed, (window+window+1)*(window+window+1));
        Instant F = Instant.now();
        for (int i = window/2; i < inputData.getRowDimension()-window/2; i++) {
            for (int j = window/2; j < inputData.getColumnDimension()-window/2; j++) {
                if (inputData.getEntry(i,j) > 0) {
                    Instant H = Instant.now();
                    PoissonDistribution dist = new PoissonDistributionImpl(normSummed.getEntry(i,j));
                    double pVal = 0;

                    try {
                        pVal = 1-dist.cumulativeProbability((int) inputData.getEntry(i,j)-1);
                    } catch (Exception e) {
                        e.printStackTrace();
                        System.err.println("error calculating p value");
                    }
                    pVal = pVal * pValAdj;
                    Instant I = Instant.now();
                    if (pVal <= maxPval) {
                        //System.out.println(i + " " + j + " " + inputData.getEntry(i,j) + " " + normSummed.getEntry(i,j) + " " + pVal);
                        int insertionPoint = Collections.binarySearch(peaks.get(2), pVal);
                        if (insertionPoint < 0) {
                            insertionPoint = -1 * (insertionPoint + 1);
                        } else if (insertionPoint >= 0) {
                            if (inputData.getEntry(i,j) <= peaks.get(3).get(insertionPoint)) {
                                insertionPoint = insertionPoint+1;
                            }
                        }
                        peaks.get(0).add(insertionPoint, (double) i);
                        peaks.get(1).add(insertionPoint, (double) j);
                        peaks.get(2).add(insertionPoint, pVal);
                        peaks.get(3).add(insertionPoint, inputData.getEntry(i,j));
                    }
                    Instant J = Instant.now();
                    //System.out.println("individual pval: " + Duration.between(H,I).toNanos());
                    //System.out.println("individual sort: " + Duration.between(I,J).toNanos());
                }
            }
        }
        Instant G = Instant.now();
        //System.out.println("norm set: " + Duration.between(A,B).toMillis());
        //System.out.println("norm sum: " + Duration.between(B,C).toMillis());
        //System.out.println("norm multiply: " + Duration.between(C,D).toMillis());
        //System.out.println("norm sliding: " + Duration.between(D,E).toMillis());
        //System.out.println("norm multiply 2: " + Duration.between(E,F).toMillis());
        //System.out.println("norm pval: " + Duration.between(F,G).toMillis());
        return peaks;
    }

    public static List<List<Double>> orderPeaksPoissonZ(RealMatrix inputData, RealMatrix expectedData, int window, double maxPval, int pValAdj) {
        List<List<Double>> peaks = new ArrayList<>();
        peaks.add(new ArrayList<>());
        peaks.add(new ArrayList<>());
        peaks.add(new ArrayList<>());
        peaks.add(new ArrayList<>());
        peaks.add(new ArrayList<>());
        for (int i = window; i < inputData.getRowDimension()-window; i++) {
            for (int j = window; j < inputData.getColumnDimension()-window; j++) {
                if (inputData.getEntry(i,j) > 0) {
                    Instant H = Instant.now();
                    PoissonDistribution dist = new PoissonDistributionImpl(expectedData.getEntry(i,j));
                    double pVal = 0;

                    try {
                        pVal = 1-dist.cumulativeProbability((int) inputData.getEntry(i,j)-1);
                    } catch (Exception e) {
                        e.printStackTrace();
                        System.err.println("error calculating p value");
                    }
                    pVal = pVal * pValAdj;
                    Instant I = Instant.now();
                    if (pVal <= maxPval) {
                        //System.out.println(i + " " + j + " " + inputData.getEntry(i,j) + " " + normSummed.getEntry(i,j) + " " + pVal);
                        double zscore = 2*(Math.sqrt(inputData.getEntry(i, j)) - Math.sqrt(expectedData.getEntry(i,j)));
                        int insertionPoint = Collections.binarySearch(peaks.get(4), zscore, Collections.reverseOrder());
                        if (insertionPoint < 0) {
                            insertionPoint = -1 * (insertionPoint + 1);
                        } //else if (insertionPoint >= 0) {
                        //    if (2*(Math.sqrt(inputData.getEntry(i, j)) - Math.sqrt(expectedData.getEntry(i,j))) <= peaks.get(4).get(insertionPoint)) {
                        //        insertionPoint = insertionPoint+1;
                        //    }
                        //}

                        peaks.get(0).add(insertionPoint, (double) i);
                        peaks.get(1).add(insertionPoint, (double) j);
                        peaks.get(2).add(insertionPoint, pVal);
                        peaks.get(3).add(insertionPoint, inputData.getEntry(i,j));
                        peaks.get(4).add(insertionPoint, zscore);
                    }
                    Instant J = Instant.now();
                    //System.out.println("individual pval: " + Duration.between(H,I).toNanos());
                    //System.out.println("individual sort: " + Duration.between(I,J).toNanos());
                }
            }
        }
        Instant G = Instant.now();
        //System.out.println("norm set: " + Duration.between(A,B).toMillis());
        //System.out.println("norm sum: " + Duration.between(B,C).toMillis());
        //System.out.println("norm multiply: " + Duration.between(C,D).toMillis());
        //System.out.println("norm sliding: " + Duration.between(D,E).toMillis());
        //System.out.println("norm multiply 2: " + Duration.between(E,F).toMillis());
        //System.out.println("norm pval: " + Duration.between(F,G).toMillis());
        return peaks;
    }
}
