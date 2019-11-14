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

import juicebox.data.HiCFileTools;
import juicebox.data.MatrixZoomData;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.tools.utils.common.UNIXTools;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.RealMatrix;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.util.Pair;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;

public class DistortionFinder extends RegionFinder {

    private Integer imgSliceWidth, imgHalfSliceWidth;
    private Integer numManipulations;
    private final int specificResolution;
    private Writer posDataWriter, posLabelWriter, negDataWriter, negLabelWriter, posImgWriter, posImgLabelWriter, negImgWriter, negImgLabelWriter;
    private String negPath, posPath, negImgPath, posImgPath;
    private int maxNumberOfExamplesForRegion = Integer.MAX_VALUE;
    private boolean ignoreNumberOfExamplesForRegion = false;
    private int counter = 0;
    private static final String prefixString = System.currentTimeMillis() + "_";

    // grind -k KR -r 5000,10000,25000,100000 --stride 3 -c 1,2,3 --dense-labels --distort <hic file> null <128,4,1000> <directory>

    public DistortionFinder(int resolution, ParameterConfigurationContainer container) {
        super(container);

        this.imgSliceWidth = x;
        imgHalfSliceWidth = x / 2;
        this.numManipulations = y;
        if (z > 0) {
            this.maxNumberOfExamplesForRegion = z;
        } else {
            ignoreNumberOfExamplesForRegion = true;
        }
        this.specificResolution = resolution;
    }

    public static void getTrainingDataAndSaveToFile(MatrixZoomData zd1, MatrixZoomData zd2, MatrixZoomData zd12,
                                                    int box1XIndex, int box2XIndex, String chrom1Name, String chrom2Name,
                                                    boolean isContinuousRegion, int imgHalfSliceWidth, NormalizationType norm,
                                                    boolean generateImages, int numManipulations, String imgFileType,
                                                    String posPath, String negPath, String posImgPath, String negImgPath,
                                                    Writer posDataWriter, Writer negDataWriter, Writer posLabelWriter, Writer negLabelWriter,
                                                    Writer posImgWriter, Writer negImgWriter, Writer posImgLabelWriter, Writer negImgLabelWriter,
                                                    boolean includeLabels) {

        int box1RectUL = box1XIndex;
        int box1RectLR = box1XIndex + imgHalfSliceWidth;

        int box2RectUL = box2XIndex;
        int box2RectLR = box2XIndex + imgHalfSliceWidth;

        try {
            float[][] compositeMatrix = generateCompositeMatrixWithNansCleanedFromZDS(zd1, zd2, zd12,
                    box1RectUL, box1RectLR, box2RectUL, box2RectLR, imgHalfSliceWidth, norm);

            float[][] labelsMatrix = GrindUtils.generateDefaultDistortionLabelsFile(compositeMatrix.length, 4, isContinuousRegion);
            //GrindUtils.cleanUpLabelsMatrixBasedOnData(labelsMatrix, compositeMatrix);

            String filePrefix = prefixString + "orig_" + chrom1Name + "_" + box1XIndex + "_" + chrom2Name + "_" + box2XIndex + "_matrix";
            GrindUtils.saveGrindMatrixDataToFile(filePrefix, negPath, compositeMatrix, negDataWriter, false);
            if (includeLabels) {
                GrindUtils.saveGrindMatrixDataToFile(filePrefix + "_labels", negPath, labelsMatrix, negLabelWriter, false);
            }

            if (generateImages) {
                String imagePrefix = filePrefix + "." + imgFileType;
                GrindUtils.saveGrindMatrixDataToImage(imagePrefix, negImgPath, compositeMatrix, negImgWriter, false);
                if (includeLabels) {
                    GrindUtils.saveGrindMatrixDataToImage(imagePrefix + "_labels." + imgFileType, negImgPath, labelsMatrix, negImgLabelWriter, true);
                }
            }

            if (includeLabels) {
                for (int k = 0; k < numManipulations; k++) {
                    Pair<float[][], float[][]> alteredMatrices = GrindUtils.randomlyManipulateMatrix(compositeMatrix, labelsMatrix, generator);
                    compositeMatrix = alteredMatrices.getFirst();
                    labelsMatrix = alteredMatrices.getSecond();

                    if (k == 0 || k == (numManipulations - 1) || generator.nextBoolean()) {
                        filePrefix = prefixString + "dstrt_" + chrom1Name + "_" + box1XIndex + "_" + chrom2Name + "_" + box2XIndex + "_" + k + "_matrix";
                        GrindUtils.saveGrindMatrixDataToFile(filePrefix, posPath, compositeMatrix, posDataWriter, false);
                        GrindUtils.saveGrindMatrixDataToFile(filePrefix + "_labels", posPath, labelsMatrix, posLabelWriter, false);

                        if (generateImages) {
                            String imagePrefix = filePrefix + "." + imgFileType;
                            GrindUtils.saveGrindMatrixDataToImage(imagePrefix, posImgPath, compositeMatrix, posImgWriter, false);
                            GrindUtils.saveGrindMatrixDataToImage(imagePrefix + "_labels." + imgFileType, posImgPath, labelsMatrix, posImgLabelWriter, true);
                        }
                    }
                }
            }

        } catch (Exception e) {

        }
    }

    private static float[][] generateCompositeMatrixWithNansCleanedFromZDS(MatrixZoomData zd1, MatrixZoomData zd2, MatrixZoomData zd12,
                                                                           int box1RectUL, int box1RectLR, int box2RectUL, int box2RectLR,
                                                                           int imgHalfSliceWidth, NormalizationType norm) throws Exception {
        RealMatrix localizedRegionDataBox1 = HiCFileTools.extractLocalBoundedRegion(zd1,
                box1RectUL, box1RectLR, box1RectUL, box1RectLR, imgHalfSliceWidth, imgHalfSliceWidth, norm, true);
        if (GrindUtils.mapRegionIsProblematic(localizedRegionDataBox1, .3)) return null;
        RealMatrix localizedRegionDataBox2 = HiCFileTools.extractLocalBoundedRegion(zd2,
                box2RectUL, box2RectLR, box2RectUL, box2RectLR, imgHalfSliceWidth, imgHalfSliceWidth, norm, true);
        if (GrindUtils.mapRegionIsProblematic(localizedRegionDataBox2, .3)) return null;
        RealMatrix localizedRegionDataBox12 = HiCFileTools.extractLocalBoundedRegion(zd12,
                box1RectUL, box1RectLR, box2RectUL, box2RectLR, imgHalfSliceWidth, imgHalfSliceWidth, norm, false);

        return MatrixTools.generateCompositeMatrixWithNansCleaned(localizedRegionDataBox1, localizedRegionDataBox2, localizedRegionDataBox12);
    }

    private void iterateAcrossIntraChromosomalRegion(MatrixZoomData zd, Chromosome chrom, int resolution) {

        // sliding along the diagonal
        int numberOfExamplesCounter = 0;
        int maxChrLength = (chrom.getLength() / resolution);
        for (int posIndex1 = 0; posIndex1 < maxChrLength - imgSliceWidth; posIndex1 += stride) {
            for (int posIndex2 = posIndex1 + imgHalfSliceWidth; posIndex2 < maxChrLength; posIndex2 += stride) {
                if (ignoreNumberOfExamplesForRegion) {
                    if (numberOfExamplesCounter++ < 1000) {
                        getTrainingDataAndSaveToFile(zd, zd, zd, posIndex1, posIndex2, chrom.getName(), chrom.getName(),
                                posIndex2 == posIndex1 + imgHalfSliceWidth, imgHalfSliceWidth, norm, generateImages, numManipulations, imgFileType,
                                posPath, negPath, posImgPath, negImgPath, posDataWriter, negDataWriter, posLabelWriter, negLabelWriter,
                                posImgWriter, negImgWriter, posImgLabelWriter, negImgLabelWriter, true);
                    } else {
                        updateLatestMainPaths();
                        getTrainingDataAndSaveToFile(zd, zd, zd, posIndex1, posIndex2, chrom.getName(), chrom.getName(),
                                posIndex2 == posIndex1 + imgHalfSliceWidth, imgHalfSliceWidth, norm, generateImages, numManipulations, imgFileType,
                                posPath, negPath, posImgPath, negImgPath, posDataWriter, negDataWriter, posLabelWriter, negLabelWriter,
                                posImgWriter, negImgWriter, posImgLabelWriter, negImgLabelWriter, true);
                        numberOfExamplesCounter = 1;
                    }

                } else {
                    if (numberOfExamplesCounter++ < maxNumberOfExamplesForRegion) {
                        getTrainingDataAndSaveToFile(zd, zd, zd, posIndex1, posIndex2, chrom.getName(), chrom.getName(),
                                posIndex2 == posIndex1 + imgHalfSliceWidth, imgHalfSliceWidth, norm, generateImages, numManipulations, imgFileType,
                                posPath, negPath, posImgPath, negImgPath, posDataWriter, negDataWriter, posLabelWriter, negLabelWriter,
                                posImgWriter, negImgWriter, posImgLabelWriter, negImgLabelWriter, true);
                    } else {
                        return;
                    }
                }
            }
        }
    }

    private void iterateDownDiagonalChromosomalRegion(MatrixZoomData zd, Chromosome chrom, int resolution) {
        // sliding along the diagonal
        int maxChrLength = (chrom.getLength() / resolution);
        for (int posIndex1 = 0; posIndex1 < maxChrLength - imgSliceWidth; posIndex1 += stride) {
            getTrainingDataAndSaveToFile(zd, zd, zd, posIndex1,
                    posIndex1 + imgHalfSliceWidth, chrom.getName(), chrom.getName(),
                    true, imgHalfSliceWidth, norm, generateImages, numManipulations, imgFileType,
                    posPath, negPath, posImgPath, negImgPath, posDataWriter, negDataWriter, posLabelWriter, negLabelWriter,
                    posImgWriter, negImgWriter, posImgLabelWriter, negImgLabelWriter, true);
        }
    }

    private void iterateBetweenInterChromosomalRegions(MatrixZoomData zd1, MatrixZoomData zd2, MatrixZoomData zd12, Chromosome chrom1, Chromosome chrom2,
                                                       int resolution) {

        // iterating across both chromosomes
        int maxChrLength1 = (chrom1.getLength() / resolution);
        int maxChrLength2 = (chrom2.getLength() / resolution);
        int numberOfExamplesCounter = 0;

        for (int posIndex1 = 0; posIndex1 < maxChrLength1 - imgHalfSliceWidth; posIndex1 += stride) {
            for (int posIndex2 = 0; posIndex2 < maxChrLength2 - imgHalfSliceWidth; posIndex2 += stride) {
                if (numberOfExamplesCounter++ < maxNumberOfExamplesForRegion) {
                    getTrainingDataAndSaveToFile(zd1, zd2, zd12, posIndex1, posIndex2, chrom1.getName(), chrom2.getName(),
                            false, imgHalfSliceWidth, norm, generateImages, numManipulations, imgFileType,
                            posPath, negPath, posImgPath, negImgPath, posDataWriter, negDataWriter, posLabelWriter, negLabelWriter,
                            posImgWriter, negImgWriter, posImgLabelWriter, negImgLabelWriter, true);
                } else {
                    return;
                }
            }
        }
    }

    @Override
    public void makeExamples() {
        updateLatestMainPaths();

        try {

            posDataWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(originalPath + "/" + specificResolution + "_pos_file_names.txt"), StandardCharsets.UTF_8));
            negDataWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(originalPath + "/" + specificResolution + "_neg_file_names.txt"), StandardCharsets.UTF_8));
            posLabelWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(originalPath + "/" + specificResolution + "_pos_label_file_names.txt"), StandardCharsets.UTF_8));
            negLabelWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(originalPath + "/" + specificResolution + "_neg_label_file_names.txt"), StandardCharsets.UTF_8));
            if (generateImages) {
                posImgWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(originalPath + "/" + specificResolution + "_pos_img_names.txt"), StandardCharsets.UTF_8));
                negImgWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(originalPath + "/" + specificResolution + "_neg_img_names.txt"), StandardCharsets.UTF_8));
                posImgLabelWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(originalPath + "/" + specificResolution + "_pos_label_img_names.txt"), StandardCharsets.UTF_8));
                negImgLabelWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(originalPath + "/" + specificResolution + "_neg_label_img_names.txt"), StandardCharsets.UTF_8));
            }
            Chromosome[] chromosomes = chromosomeHandler.getChromosomeArrayWithoutAllByAll();
            for (int chrIndexI = 0; chrIndexI < chromosomes.length; chrIndexI++) {
                Chromosome chromI = chromosomes[chrIndexI];
                for (int chrIndexJ = chrIndexI; chrIndexJ < chromosomes.length; chrIndexJ++) {
                    Chromosome chromJ = chromosomes[chrIndexJ];

                    final MatrixZoomData zd = HiCFileTools.getMatrixZoomData(ds, chromI, chromJ, specificResolution);
                    if (zd == null) continue;

                    boolean isIntraChromosomal = chrIndexI == chrIndexJ;

                    System.out.println("Currently processing: " + chromI.getName() + " - " + chromJ.getName() +
                            " at specificResolution " + specificResolution);

                    MatrixZoomData matrixZoomDataI, matrixZoomDataJ;
                    if (isIntraChromosomal) {
                        if (useDiagonal) {
                            iterateDownDiagonalChromosomalRegion(zd, chromI, specificResolution);
                        } else {
                            iterateAcrossIntraChromosomalRegion(zd, chromI, specificResolution);
                        }
                    } else {
                        // is Inter
                        matrixZoomDataI = HiCFileTools.getMatrixZoomData(ds, chromI, chromI, specificResolution);
                        matrixZoomDataJ = HiCFileTools.getMatrixZoomData(ds, chromJ, chromJ, specificResolution);
                        if (matrixZoomDataI == null || matrixZoomDataJ == null) continue;

                        iterateBetweenInterChromosomalRegions(matrixZoomDataI, matrixZoomDataJ, zd, chromI, chromJ, specificResolution);
                    }
                }
            }
            for (Writer writer : new Writer[]{posDataWriter, posLabelWriter, negDataWriter, negLabelWriter}) {
                writer.close();
            }
            if (generateImages) {
                for (Writer writer : new Writer[]{posImgWriter, posImgLabelWriter, negImgWriter, negImgLabelWriter}) {
                    writer.close();
                }
            }
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    private void updateLatestMainPaths() {

        negPath = UNIXTools.makeDir(originalPath + "/negative_" + specificResolution + "_" + counter);
        posPath = UNIXTools.makeDir(originalPath + "/positive_" + specificResolution + "_" + counter);

        if (generateImages) {
            negImgPath = UNIXTools.makeDir(originalPath + "/negativeImg_" + specificResolution + "_" + counter);
            posImgPath = UNIXTools.makeDir(originalPath + "/positiveImg_" + specificResolution + "_" + counter);
        }
        counter++;
    }
}
