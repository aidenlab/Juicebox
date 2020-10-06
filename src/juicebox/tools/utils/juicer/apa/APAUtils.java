/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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

package juicebox.tools.utils.juicer.apa;

import juicebox.data.HiCFileTools;
import juicebox.data.MatrixZoomData;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.tools.utils.common.StatPercentile;
import juicebox.track.feature.Feature2D;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.RealMatrix;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Muhammad Shamim on 1/21/15.
 */
public class APAUtils {

    /**
     * @param filename
     * @param matrix
     */
    public static void saveMeasures(String filename, RealMatrix matrix, int currentRegionWidth) {

        Writer writer = null;

        APARegionStatistics apaStats = new APARegionStatistics(matrix, currentRegionWidth);

        try {
            writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), StandardCharsets.UTF_8));
            writer.write("P2M" + '\t' + apaStats.getPeak2mean() + '\n');
            writer.write("P2UL" + '\t' + apaStats.getPeak2UL() + '\n');
            writer.write("P2UR" + '\t' + apaStats.getPeak2UR() + '\n');
            writer.write("P2LL" + '\t' + apaStats.getPeak2LL() + '\n');
            writer.write("P2LR" + '\t' + apaStats.getPeak2LR() + '\n');
            writer.write("ZscoreLL" + '\t' + apaStats.getZscoreLL());
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            try {
                if (writer != null)
                    writer.close();
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    }

    public static void saveListText(String filename, List<Double> array) {
        Writer writer = null;
        try {
            writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), StandardCharsets.UTF_8));
            for (double val : array) {
                writer.write(val + " ");
            }
            writer.write("\n");
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            try {
                if (writer != null)
                    writer.close();
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    }

    public static RealMatrix standardNormalization(RealMatrix matrix) {
        return matrix.copy().scalarMultiply(1. / Math.max(1., MatrixTools.mean(matrix)));
    }


    public static RealMatrix centerNormalization(RealMatrix matrix) {

        int center = matrix.getRowDimension() / 2;
        double centerVal = matrix.getEntry(center, center);

        if (centerVal == 0) {
            centerVal = MatrixTools.minimumPositive(matrix);
            if (centerVal == 0)
                centerVal = 1;
        }

        return matrix.copy().scalarMultiply(1. / centerVal);
    }

    public static double peakEnhancement(RealMatrix matrix) {
        int rows = matrix.getRowDimension();
        int center = rows / 2;
        double centerVal = matrix.getEntry(center, center);
        double remainingSum = MatrixTools.sum(matrix.getData()) - centerVal;
        double remainingAverage = remainingSum / (rows * rows - 1);
        return centerVal / remainingAverage;
    }


    /**
     * @param data
     * @return
     */
    public static RealMatrix rankPercentile(RealMatrix data) {
        int n = data.getColumnDimension();
        StatPercentile percentile = new StatPercentile(MatrixTools.flattenedRowMajorOrderMatrix(data));

        RealMatrix matrix = new Array2DRowRealMatrix(n, n);
        for (int r = 0; r < n; r++) {
            for (int c = 0; c < n; c++) {
                double currValue = data.getEntry(r, c);
                if (currValue == 0) {
                    matrix.setEntry(r, c, 0);
                } else {
                    matrix.setEntry(r, c, percentile.evaluate(currValue));
                }
                //matrix.setEntry(r, c, percentile.evaluate());
            }
        }
        return matrix;
    }

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

    public static RealMatrix extractLocalizedDataForAFA(MatrixZoomData zd, Feature2D loop,
                                                        int resolution, int window, NormalizationType norm) throws IOException {
		long loopX = loop.getMidPt1() / resolution;
		long loopY = loop.getMidPt2() / resolution;
		long binXStart = loopY;
		long binXEnd = loopX + (window + 1);
		long binYStart = loopY - window;
		long binYEnd = loopX + 1;
		int L = (int) (binXStart - binXEnd);
		int dis = zd.getBinSize();
        /*int loopX = loop.getMidPt1() / resolution;
        int loopY = loop.getMidPt2() / resolution;
        int binXStart = loopX - window;
        int binXEnd = loopY+1;
        int binYStart = loopX;
        int binYEnd = loopY + window;
        L = binXStart - binXEnd;*/
	
		return HiCFileTools.extractLocalBoundedRegion(zd, binXStart, binXEnd, binYStart, binYEnd, L, L, norm, false);
    }

    public static RealMatrix linearInterpolation (RealMatrix original, int targetNumRows, int targetNumCols){
        int r, c, i = 0, j = 0; //i is index of  newRPos, j is index of newCPos
        int[] newRPos = new int [original.getRowDimension()];
        int[] newCPos = new int [original.getColumnDimension()];
        int rowFolds, colFolds, cSpan, rSpan; //cSpan is distance of 2 nearby original entries of the same row in resized matrix
                                                //rSpan is distance of 2 nearby original entries of the same column in resized matrix
        double value;
        //Resizing (magnifying) matrix using linear interpolation
        // initialize final matrix
        RealMatrix resizedMatrix = MatrixTools.cleanArray2DMatrix(targetNumRows, targetNumCols);
        //For every entries in the original matrix add it to the resized matrix
        rowFolds = targetNumRows / original.getRowDimension();
        colFolds = targetNumCols / original.getColumnDimension();
        for (r = 0; r < original.getRowDimension(); r++){
            newRPos [r] = r*rowFolds;
            for (c = 0; c < original.getColumnDimension(); c++) {
                newCPos[c] = c * colFolds;
                resizedMatrix.addToEntry(newRPos[r], newCPos[c], original.getEntry(r, c));
            }
        }
        cSpan = newCPos [1] - newCPos[0];
        rSpan = newRPos [1] - newRPos[0];
        //For every entry in the row that contains the entries from the original matrix, calculate and add its value using linear interpolation
        double currentOgEntry = 0;
        double nextOgEntry = 0;
        double fraction = 0;
        for (i = 0; i < newRPos.length; i ++){
            r = newRPos[i];
            for (c = 0; c < resizedMatrix.getColumnDimension(); c++) {
                if (c >= newCPos[newCPos.length - 1]) {
                    resizedMatrix.addToEntry(r, c, resizedMatrix.getEntry(r, newCPos[newCPos.length - 1]));
                } else {
                    currentOgEntry = resizedMatrix.getEntry(r, c - c%cSpan);
                    nextOgEntry = resizedMatrix.getEntry(r, c + cSpan - c%cSpan);
                    fraction = (((double)cSpan) - (double)(c%cSpan)) / (double)cSpan;
                    value = currentOgEntry * fraction + nextOgEntry * (1.0 - fraction);
                    resizedMatrix.addToEntry(r, c, value);
                }
            }
        }
        //For every other entry in the resized matrix, calculate its value using linear interpolation
        i = 0; j = 0;
        for (r = 0; r < resizedMatrix.getRowDimension(); r++){
            if (r != newRPos[i] && r < newRPos[newRPos.length-1]) {
                for (c = 0; c < resizedMatrix.getColumnDimension(); c++) {
                    currentOgEntry = resizedMatrix.getEntry(r - r%rSpan, c);
                    nextOgEntry = resizedMatrix.getEntry(r + rSpan - r%rSpan, c);
                    fraction = (double)(r%rSpan) / (double)rSpan;
                    value = currentOgEntry * (1.0 - fraction) + nextOgEntry * fraction;
                    resizedMatrix.addToEntry(r, c, value);
                }
            }
            else {
                for (c = 0; c < resizedMatrix.getColumnDimension(); c++) {
                    resizedMatrix.addToEntry(r, c, resizedMatrix.getEntry(newRPos[newRPos.length-1], c));
                }
            }
        }
        return resizedMatrix;
    }
    public static RealMatrix expandWithZeros (RealMatrix original, int targetNumRows, int targetNumCols){
        int r, c;
        RealMatrix newMatrix = MatrixTools.cleanArray2DMatrix(targetNumRows, targetNumCols);
        for (r = 0; r < original.getRowDimension(); r++){
            for (c = 0; c < original.getColumnDimension(); c++){
                newMatrix.addToEntry(r, c, original.getEntry(r, c));
            }
        }
        return newMatrix;
    }
    public static RealMatrix boxSampling (RealMatrix original, int targetNumRows, int targetNumCols){
        //This method scale down a matrix using box sampling
        RealMatrix resizedMatrix = MatrixTools.cleanArray2DMatrix(targetNumRows, targetNumCols);
        int rowScale = original.getRowDimension() / resizedMatrix.getRowDimension();
        int colScale = original.getColumnDimension() / resizedMatrix.getColumnDimension();
        RealMatrix tile = MatrixTools.cleanArray2DMatrix(rowScale, colScale);
        int r, c, i, j;
        double value;
        for (r = 0; r < resizedMatrix.getRowDimension(); r += rowScale) {
            for (c = 0; c < resizedMatrix.getColumnDimension(); c += colScale){
                //add values to entries in tile matrix
                for (i = 0; i < rowScale; i ++){
                    for (j = 0; j < colScale; j++){
                        tile.addToEntry(i, j, original.getEntry(r+i, c+j));
                    }
                }
                value = maxInMatrix(tile);
                resizedMatrix.addToEntry(r/rowScale, c/colScale, value);
            }
        }
        return resizedMatrix;
    }
    public static double maxInMatrix (RealMatrix matrix){
        double result = matrix.getEntry(0,0);
        int r, c;
        double entry;
        for (r = 0; r < matrix.getRowDimension(); r++){
            for (c = 0; c < matrix.getColumnDimension(); c++){
                entry = matrix.getEntry(r, c);
                if (result < entry) result = entry;
            }
        }
        return result;
    }
    static int lcm (int a, int b){
        return (a*b)/gcd (a, b);
    }
    static int gcd (int a, int b){
        if (a == 0 || b == 0) return 0;
        if (a == b) return a;
        if (a > b) return gcd (a-b, b);
        return gcd (b-a, a);
    }
    public static RealMatrix matrixScaling (RealMatrix original, int targetNumRows, int targetNumCols){
        int intermediateRowDimention = lcm (original.getRowDimension(), targetNumRows);
        int intermediateColDimention = lcm (original.getColumnDimension(), targetNumCols);
        RealMatrix intermediateMatrix = linearInterpolation(original, intermediateRowDimention, intermediateColDimention);
        // scale the intermediate down with box sampling
        RealMatrix resizedMatrix = boxSampling(intermediateMatrix, targetNumRows, targetNumCols);
        return resizedMatrix;
    }
}
