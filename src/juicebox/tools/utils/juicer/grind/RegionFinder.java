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

package juicebox.tools.utils.juicer.grind;

import juicebox.data.*;
import juicebox.mapcolorui.Feature2DHandler;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.tools.utils.dev.drink.ExtractingOEDataUtils;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.RealMatrix;
import org.broad.igv.feature.Chromosome;

import java.io.IOException;
import java.io.Writer;
import java.util.List;
import java.util.Random;
import java.util.Set;

abstract public class RegionFinder {

    protected static final Random generator = new Random(0);
    protected Integer x;
    protected Integer y;
    protected Integer z;
    protected Integer stride;
    protected Dataset ds;
    protected String originalPath;
    protected NormalizationType norm;
    protected Set<Integer> resolutions;
    protected ChromosomeHandler chromosomeHandler;
    protected boolean onlyMakePositiveExamples;
    protected boolean useDenseLabelsNotBinary;
    protected boolean useAmorphicPixelLabeling;
    protected boolean useObservedOverExpected;
    protected boolean useTxtInsteadOfNPY;
    protected boolean generateImages;
    protected boolean useDiagonal;
    protected boolean featureDirectionOrientationIsImportant;
    protected String imgFileType;
    protected Feature2DList inputFeature2DList;
    protected int offsetOfCornerFromDiagonal;


    public RegionFinder(ParameterConfigurationContainer container) {
        this.x = container.x;
        this.y = container.y;
        this.z = container.z;
        this.stride = container.stride;
        this.ds = container.ds;
        this.originalPath = container.outputDirectory.getPath();
        this.norm = container.norm;
        this.resolutions = container.resolutions;
        this.chromosomeHandler = container.ds.getChromosomeHandler();
        this.onlyMakePositiveExamples = container.onlyMakePositiveExamples;
        this.useDenseLabelsNotBinary = container.useDenseLabelsNotBinary;
        this.useObservedOverExpected = container.useObservedOverExpected;
        this.useAmorphicPixelLabeling = container.useAmorphicPixelLabeling;
        this.useTxtInsteadOfNPY = container.useTxtInsteadOfNPY;
        this.useDiagonal = container.useDiagonal;
        this.featureDirectionOrientationIsImportant = container.featureDirectionOrientationIsImportant;
        this.imgFileType = container.imgFileType;
        generateImages = imgFileType != null && imgFileType.length() > 0;
        inputFeature2DList = container.feature2DList;
        this.offsetOfCornerFromDiagonal = container.offsetOfCornerFromDiagonal;

    }

    private static boolean stripeIsCorrectOrientation(int rowLength, int colLength, boolean isVerticalFeature2D) {
        if (isVerticalFeature2D) {
            return rowLength > colLength;
        } else {
            return colLength > rowLength;
        }
    }

    public abstract void makeExamples();

    protected void getTrainingDataAndSaveToFile(MatrixZoomData zd, Chromosome chrom, int rowIndex, int colIndex, int resolution,
                                                Feature2DHandler feature2DHandler, Integer numInputRows, Integer numInputCols, String posPath, String negPath,
                                                Writer posWriter, Writer posLabelWriter, Writer negWriter, boolean isVerticalFeature2D) throws IOException {
        int rectULX = rowIndex;
        int rectULY = colIndex;
        int rectLRX = rowIndex + numInputRows;
        int rectLRY = colIndex + numInputCols;
        int numRows = numInputRows;
        int numCols = numInputCols;

        if (featureDirectionOrientationIsImportant && isVerticalFeature2D) {
            rectULX = rowIndex - numInputCols;
            rectULY = colIndex - numInputRows;
            rectLRX = rowIndex;
            rectLRY = colIndex;
            numRows = numInputCols;
            numCols = numInputRows;
        }

        int[][] labelsMatrix = new int[numRows][numCols];
        int[][] experimentalAmorphicLabelsMatrix = new int[numRows][numCols];

        RealMatrix localizedRegionData;
        if (useObservedOverExpected) {
            ExpectedValueFunction df = ds.getExpectedValues(zd.getZoom(), norm);
            if (df == null) {
                System.err.println("O/E data not available at " + zd.getZoom() + " " + norm);
                return;
            }
            localizedRegionData = ExtractingOEDataUtils.extractObsOverExpBoundedRegion(zd,
                    rectULX, rectLRX, rectULY, rectLRY,
                    numRows, numCols, norm, true, df,
                    chrom.getIndex(), 4, true, ExtractingOEDataUtils.ThresholdType.LOG_OE_BOUNDED_SCALED_BTWN_ZERO_ONE);
        } else {
            localizedRegionData = HiCFileTools.extractLocalBoundedRegion(zd,
                    rectULX, rectLRX, rectULY, rectLRY,
                    numRows, numCols, norm, true);
        }

        List<Feature2D> inputListFoundFeatures = feature2DHandler.getContainedFeatures(chrom, rectULX, rectULY, rectLRX, rectLRY, resolution);


        boolean annotationFoundInRegion = false;
        for (Feature2D feature2D : inputListFoundFeatures) {
            int featureRowLength = Math.max(feature2D.getWidth1() / resolution, 1);
            int featureColLength = Math.max(feature2D.getWidth2() / resolution, 1);

            if (!featureDirectionOrientationIsImportant || stripeIsCorrectOrientation(featureRowLength, featureColLength, isVerticalFeature2D)) {

                int relativeStartRowFromOrigin = feature2D.getStart1() / resolution - rectULX;
                int relativeStartColFromOrigin = feature2D.getStart2() / resolution - rectULY;
                MatrixTools.labelRegionWithOnes(labelsMatrix, featureRowLength, numRows, featureColLength, numCols, relativeStartRowFromOrigin, relativeStartColFromOrigin);

                if (useAmorphicPixelLabeling) {
                    MatrixTools.labelEnrichedRegionWithOnes(experimentalAmorphicLabelsMatrix, localizedRegionData.getData(), featureRowLength, numRows, featureColLength, numCols, relativeStartRowFromOrigin, relativeStartColFromOrigin);
                }
                annotationFoundInRegion = true;
            }
        }

        float[][] finalData = MatrixTools.convertToFloatMatrix(localizedRegionData.getData());
        int[][] finalLabels = labelsMatrix;
        int[][] finalExpLabels = experimentalAmorphicLabelsMatrix;


        String orientationType = "";
        if (featureDirectionOrientationIsImportant) {
            orientationType = "_Horzntl";
            if (isVerticalFeature2D) {
                finalData = GrindUtils.appropriatelyTransformVerticalStripes(finalData);
                finalLabels = GrindUtils.appropriatelyTransformVerticalStripes(finalLabels);
                finalExpLabels = GrindUtils.appropriatelyTransformVerticalStripes(finalExpLabels);
                orientationType = "_Vertcl";
            }
        }
        String filePrefix = chrom.getName() + "_" + rowIndex + "_" + colIndex + orientationType;

        if (annotationFoundInRegion) {
            GrindUtils.saveGrindMatrixDataToFile(filePrefix + "_matrix", posPath, finalData, posWriter, useTxtInsteadOfNPY);
            GrindUtils.saveGrindMatrixDataToFile(filePrefix + "_matrix.label", posPath, finalLabels, posLabelWriter, useTxtInsteadOfNPY);
            if (useAmorphicPixelLabeling) {
                GrindUtils.saveGrindMatrixDataToFile(filePrefix + "_matrix.label.exp", posPath, finalExpLabels, posLabelWriter, useTxtInsteadOfNPY);
            }
        } else if (!onlyMakePositiveExamples) {
            GrindUtils.saveGrindMatrixDataToFile(filePrefix + "_matrix", negPath, finalData, negWriter, useTxtInsteadOfNPY);
        }
    }
}
