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

package juicebox.tools.utils.juicer.curse;

import juicebox.data.*;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.RealMatrix;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ExtractingOEDataUtils {

    public static RealMatrix extractLocalThresholdedLogOEBoundedRegion(MatrixZoomData zd, int binXStart, int binXEnd,
                                                                       int binYStart, int binYEnd, int numRows, int numCols,
                                                                       NormalizationType normalizationType, boolean isIntra,
                                                                       ExpectedValueFunction df, int chrIndex, double threshold) throws IOException {
        if (isIntra && df == null) {
            System.err.println("DF is null");
            return null;
        }

        // numRows/numCols is just to ensure a set size in case bounds are approximate
        // left upper corner is reference for 0,0
        List<Block> blocks = HiCFileTools.getAllRegionBlocks(zd, binXStart, binXEnd, binYStart, binYEnd, normalizationType);

        RealMatrix data = MatrixTools.cleanArray2DMatrix(numRows, numCols);

        double averageCount = zd.getAverageCount();

        if (blocks.size() > 0) {
            for (Block b : blocks) {
                if (b != null) {
                    for (ContactRecord rec : b.getContactRecords()) {

                        int x = rec.getBinX();
                        int y = rec.getBinY();

                        double expected;
                        if (isIntra) {
                            int dist = Math.abs(x - y);
                            expected = df.getExpectedValue(chrIndex, dist);
                        } else {
                            expected = (averageCount > 0 ? averageCount : 1);
                        }

                        double oeVal = Math.log(rec.getCounts() / expected);
                        oeVal = Math.min(Math.max(-threshold, oeVal), threshold);

                        // place oe value in relative position
                        int relativeX = rec.getBinX() - binXStart;
                        int relativeY = rec.getBinY() - binYStart;
                        if (relativeX >= 0 && relativeX < numRows) {
                            if (relativeY >= 0 && relativeY < numCols) {
                                data.addToEntry(relativeX, relativeY, oeVal);
                            }
                        }

                        if (isIntra) {
                            // check if the other half of matrix should also be displayed/passed in
                            relativeX = rec.getBinY() - binXStart;
                            relativeY = rec.getBinX() - binYStart;
                            if (relativeX >= 0 && relativeX < numRows) {
                                if (relativeY >= 0 && relativeY < numCols) {
                                    data.addToEntry(relativeX, relativeY, oeVal);
                                }
                            }
                        }
                    }
                }
            }
        }
        // ~force cleanup
        blocks = null;

        return data;
    }


    public static double extractAveragedOEBoundedRegion(MatrixZoomData zd, int binXStart, int binXEnd,
                                                        int binYStart, int binYEnd, int numRows, int numCols,
                                                        NormalizationType normalizationType, boolean isIntra,
                                                        ExpectedValueFunction df, int chrIndex) throws IOException {
        if (isIntra && df == null) {
            System.err.println("DF is null");
            return 0.0;
        }

        // numRows/numCols is just to ensure a set size in case bounds are approximate
        // left upper corner is reference for 0,0
        List<Block> blocks = HiCFileTools.getAllRegionBlocks(zd, binXStart, binXEnd, binYStart, binYEnd, normalizationType);

        RealMatrix data = MatrixTools.cleanArray2DMatrix(numRows, numCols);

        if (blocks.size() > 0) {
            for (Block b : blocks) {
                if (b != null) {
                    for (ContactRecord rec : b.getContactRecords()) {

                        int x = rec.getBinX();
                        int y = rec.getBinY();

                        double expected = 1;
                        if (isIntra) {
                            int dist = Math.abs(x - y);
                            expected = df.getExpectedValue(chrIndex, dist);
                        }


                        double oeVal = rec.getCounts();
                        if (isIntra) {
                            oeVal = Math.log(rec.getCounts() / expected);
                            //oeVal = Math.min(Math.max(-threshold, oeVal), threshold);
                        }

                        // place oe value in relative position
                        int relativeX = rec.getBinX() - binXStart;
                        int relativeY = rec.getBinY() - binYStart;
                        if (relativeX >= 0 && relativeX < numRows) {
                            if (relativeY >= 0 && relativeY < numCols) {
                                data.addToEntry(relativeX, relativeY, oeVal);
                            }
                        }

                        if (isIntra) {
                            // check if the other half of matrix should also be displayed/passed in
                            relativeX = rec.getBinY() - binXStart;
                            relativeY = rec.getBinX() - binYStart;
                            if (relativeX >= 0 && relativeX < numRows) {
                                if (relativeY >= 0 && relativeY < numCols) {
                                    data.addToEntry(relativeX, relativeY, oeVal);
                                }
                            }
                        }
                    }
                }
            }
        }
        // ~force cleanup
        blocks = null;

        return MatrixTools.getAverage(data);
    }

    public static double[][] extractLocalOEBoundedRegion(MatrixZoomData zd, int binXStart, int binXEnd,
                                                         int binYStart, int binYEnd, int numRows, int numCols,
                                                         NormalizationType normalizationType, boolean isIntra,
                                                         ExpectedValueFunction df, int chrIndex, double threshold) throws IOException {

        if (isIntra && df == null) {
            System.err.println("DF is null");
            System.exit(87);
            return null;
        }

        // numRows/numCols is just to ensure a set size in case bounds are approximate
        // left upper corner is reference for 0,0
        List<Block> blocks = HiCFileTools.getAllRegionBlocks(zd, binXStart, binXEnd, binYStart, binYEnd, normalizationType);

        RealMatrix data = MatrixTools.cleanArray2DMatrix(numRows, numCols);

        if (blocks.size() > 0) {
            for (Block b : blocks) {
                if (b != null) {
                    for (ContactRecord rec : b.getContactRecords()) {

                        int x = rec.getBinX();
                        int y = rec.getBinY();

                        double expected = 1;
                        if (isIntra) {
                            int dist = Math.abs(x - y);
                            expected = df.getExpectedValue(chrIndex, dist);
                        }


                        double oeVal = rec.getCounts();
                        if (isIntra) {
                            oeVal = Math.log(rec.getCounts() / expected);
                            oeVal = Math.min(Math.max(-threshold, oeVal), threshold);
                        }

                        if (Double.isNaN(oeVal)) oeVal = 0;

                        // place oe value in relative position
                        int relativeX = rec.getBinX() - binXStart;
                        int relativeY = rec.getBinY() - binYStart;
                        if (relativeX >= 0 && relativeX < numRows) {
                            if (relativeY >= 0 && relativeY < numCols) {
                                data.addToEntry(relativeX, relativeY, oeVal);
                            }
                        }

                        if (isIntra) {
                            // check if the other half of matrix should also be displayed/passed in
                            relativeX = rec.getBinY() - binXStart;
                            relativeY = rec.getBinX() - binYStart;
                            if (relativeX >= 0 && relativeX < numRows) {
                                if (relativeY >= 0 && relativeY < numCols) {
                                    data.addToEntry(relativeX, relativeY, oeVal);
                                }
                            }
                        }
                    }
                }
            }
        }
        // ~force cleanup
        blocks = null;

        return data.getData();
    }

    public static double extractAveragedOEFromRegion(double[][] allDataForRegion, int binXStart, int binXEnd,
                                                     int binYStart, int binYEnd, boolean isIntra, double threshold,
                                                     double averageForRegion) {

        List<Double> values = new ArrayList<>();
        int totalNumInclZero = (binXEnd - binXStart) * (binYEnd - binYStart);
        int numNotZero = 0;
        double accum = 0;
        for (int i = binXStart; i < binXEnd; i++) {
            for (int j = binYStart; j < binYEnd; j++) {
                if (!Double.isNaN(allDataForRegion[i][j])) {
                    accum += allDataForRegion[i][j];
                    values.add(allDataForRegion[i][j]);
                    numNotZero++;
                }
            }
        }

        if (numNotZero == 0) numNotZero = 1;
        if (Double.isNaN(accum)) accum = 0;

        double average;
        if (isIntra) {
// todo           average = Math.log(accum/totalNumInclZero)*2;
            //average = accum/numNotZero;//*2
            //average = Math.log(accum/numNotZero);//*2
            average = accum / totalNumInclZero;//*2
            //Collections.sort(values);
            //average = Math.log(values.get(values.size()/2));

        } else {
            average = Math.log(accum / totalNumInclZero); //  div / averageForRegion
            //average = accum/totalNumInclZero;
        }

        average = Math.max(Math.min(average, threshold), -threshold);

        if (Double.isNaN(average)) average = 0;

        return average;
    }
}
