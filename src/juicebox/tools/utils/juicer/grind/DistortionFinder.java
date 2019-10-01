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
import juicebox.tools.utils.common.MatrixTools;
import juicebox.tools.utils.common.UNIXTools;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.RealMatrix;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.util.Pair;

import java.io.*;
import java.nio.charset.StandardCharsets;

public class DistortionFinder implements RegionFinder {


    private Integer imgSliceWidth, imgHalfSliceWidth;
    private Integer numManipulations;
    private Dataset ds;
    private String path;
    private ChromosomeHandler chromosomeHandler;
    private NormalizationType norm;
    private boolean useObservedOverExpected;
    private boolean useDenseLabels;
    private final int resolution;
    private int stride;
    private Writer posDataWriter, posLabelWriter, negDataWriter, negLabelWriter, posImgWriter, posImgLabelWriter, negImgWriter, negImgLabelWriter;
    private String negPath, posPath, negImgPath, posImgPath;
    // grind -k KR -r 5000,10000,25000,100000 --stride 3 -c 1,2,3 --dense-labels --distort <hic file> null <64,64,1000> <directory>

    public DistortionFinder(int imgSliceWidth, int numManipulations, Dataset ds, File outputDirectory,
                            ChromosomeHandler chromosomeHandler, NormalizationType norm,
                            boolean useObservedOverExpected, boolean useDenseLabels, int resolution, int stride) {

        this.imgSliceWidth = imgSliceWidth;
        imgHalfSliceWidth = imgSliceWidth / 2;
        this.numManipulations = numManipulations;
        this.ds = ds;
        this.path = outputDirectory.getPath();
        this.chromosomeHandler = chromosomeHandler;
        this.norm = norm;
        this.useObservedOverExpected = useObservedOverExpected;
        this.useDenseLabels = useDenseLabels;
        this.resolution = resolution;
        this.stride = stride;
    }

    @Override
    public void makeExamples() {
        negPath = path + "/negative_" + resolution;
        posPath = path + "/positive_" + resolution;
        negImgPath = path + "/negativeImg_" + resolution;
        posImgPath = path + "/positiveImg_" + resolution;
        UNIXTools.makeDir(negPath);
        UNIXTools.makeDir(posPath);
        UNIXTools.makeDir(negImgPath);
        UNIXTools.makeDir(posImgPath);
        try {

            posDataWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path + "/" + resolution + "_pos_file_names.txt"), StandardCharsets.UTF_8));
            negDataWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path + "/" + resolution + "_neg_file_names.txt"), StandardCharsets.UTF_8));
            posLabelWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path + "/" + resolution + "_pos_label_file_names.txt"), StandardCharsets.UTF_8));
            negLabelWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path + "/" + resolution + "_neg_label_file_names.txt"), StandardCharsets.UTF_8));
            posImgWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path + "/" + resolution + "_pos_img_names.txt"), StandardCharsets.UTF_8));
            negImgWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path + "/" + resolution + "_neg_img_names.txt"), StandardCharsets.UTF_8));
            posImgLabelWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path + "/" + resolution + "_pos_label_img_names.txt"), StandardCharsets.UTF_8));
            negImgLabelWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path + "/" + resolution + "_neg_label_img_names.txt"), StandardCharsets.UTF_8));

            Chromosome[] chromosomes = chromosomeHandler.getChromosomeArrayWithoutAllByAll();
            for (int chrArrayI = 0; chrArrayI < chromosomes.length; chrArrayI++) {
                Chromosome chromI = chromosomes[chrArrayI];
                for (int chrArrayJ = chrArrayI; chrArrayJ < chromosomes.length; chrArrayJ++) {
                    Chromosome chromJ = chromosomes[chrArrayJ];

                    Matrix matrix = ds.getMatrix(chromI, chromJ);
                    if (matrix == null) continue;

                    HiCZoom zoom = ds.getZoomForBPResolution(resolution);
                    final MatrixZoomData zd = matrix.getZoomData(zoom);
                    if (zd == null) continue;

                    boolean isIntraChromosomal = chrArrayI == chrArrayJ;

                    System.out.println("Currently processing: " + chromI.getName() + " - " + chromJ.getName() +
                            " at resolution " + resolution);

                    MatrixZoomData matrixZoomDataI, matrixZoomDataJ;
                    if (isIntraChromosomal) {
                        iterateAcrossIntraChromosomalRegion(zd, chromI, resolution);
                    } else {
                        // is Inter
                        matrixZoomDataI = HiCFileTools.getMatrixZoomData(ds, chromI, chromI, zoom);
                        matrixZoomDataJ = HiCFileTools.getMatrixZoomData(ds, chromJ, chromJ, zoom);
                        if (matrixZoomDataI == null) continue;
                        if (matrixZoomDataJ == null) continue;

                        iterateBetweenInterChromosomalRegions(zd, matrixZoomDataI, matrixZoomDataJ, chromI, chromJ, resolution);
                    }
                }
            }
            for (Writer writer : new Writer[]{posDataWriter, posLabelWriter, negDataWriter, negLabelWriter, posImgWriter, posImgLabelWriter, negImgWriter, negImgLabelWriter}) {
                writer.close();
            }
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }


    private void iterateAcrossIntraChromosomalRegion(MatrixZoomData zd, Chromosome chrom, int resolution) {

        // sliding along the diagonal
        int maxChrLength = (chrom.getLength() / resolution);
        for (int posIndex1 = 0; posIndex1 < maxChrLength - imgSliceWidth; posIndex1 += stride) {
            for (int posIndex2 = posIndex1 + imgHalfSliceWidth; posIndex2 < maxChrLength; posIndex2 += stride) {
                getTrainingDataAndSaveToFile(zd, zd, zd, posIndex1, posIndex2,
                        chrom.getName(), chrom.getName(), posIndex2 == posIndex1 + imgHalfSliceWidth);
            }
        }
    }

    private void iterateBetweenInterChromosomalRegions(MatrixZoomData zd1, MatrixZoomData zd2, MatrixZoomData zd12, Chromosome chrom1, Chromosome chrom2,
                                                       int resolution) {

        // iterating across both chromosomes
        int maxChrLength1 = (chrom1.getLength() / resolution);
        int maxChrLength2 = (chrom2.getLength() / resolution);

        for (int posIndex1 = 0; posIndex1 < maxChrLength1 - imgHalfSliceWidth; posIndex1 += stride) {
            for (int posIndex2 = 0; posIndex2 < maxChrLength2 - imgHalfSliceWidth; posIndex2 += stride) {
                getTrainingDataAndSaveToFile(zd1, zd2, zd12, posIndex1, posIndex2, chrom1.getName(), chrom2.getName(), false);
            }
        }
    }


    private void getTrainingDataAndSaveToFile(MatrixZoomData zd1, MatrixZoomData zd2, MatrixZoomData zd12,
                                              int box1XIndex, int box2XIndex, String chrom1Name, String chrom2Name,
                                              boolean isContinuousRegion) {

        int box1RectUL = box1XIndex;
        int box1RectLR = box1XIndex + imgHalfSliceWidth;

        int box2RectUL = box2XIndex;
        int box2RectLR = box2XIndex + imgHalfSliceWidth;

        try {
            RealMatrix localizedRegionDataBox1 = HiCFileTools.extractLocalBoundedRegion(zd1,
                    box1RectUL, box1RectLR, box1RectUL, box1RectLR, imgHalfSliceWidth, imgHalfSliceWidth, norm, true);
            if (GrindUtils.mapRegionIsProblematic(localizedRegionDataBox1, .3)) return;
            RealMatrix localizedRegionDataBox2 = HiCFileTools.extractLocalBoundedRegion(zd2,
                    box2RectUL, box2RectLR, box2RectUL, box2RectLR, imgHalfSliceWidth, imgHalfSliceWidth, norm, true);
            if (GrindUtils.mapRegionIsProblematic(localizedRegionDataBox2, .3)) return;
            RealMatrix localizedRegionDataBox12 = HiCFileTools.extractLocalBoundedRegion(zd12,
                    box1RectUL, box1RectLR, box2RectUL, box2RectLR, imgHalfSliceWidth, imgHalfSliceWidth, norm, false);

            double[][] compositeMatrix = MatrixTools.generateCompositeMatrix(localizedRegionDataBox1, localizedRegionDataBox2, localizedRegionDataBox12);
            MatrixTools.cleanUpNaNs(compositeMatrix);
            double[][] labelsMatrix = GrindUtils.generateDefaultDistortionLabelsFile(compositeMatrix.length, 4, isContinuousRegion);
            GrindUtils.cleanUpLabelsMatrixBasedOnData(labelsMatrix, compositeMatrix);

            String filePrefix = "orig_" + chrom1Name + "_" + box1XIndex + "_" + chrom2Name + "_" + box2XIndex + "_matrix.txt";
            GrindUtils.saveGrindMatrixDataToFile(filePrefix, negPath, compositeMatrix, negDataWriter);
            GrindUtils.saveGrindMatrixDataToFile(filePrefix + "_labels.txt", negPath, labelsMatrix, negLabelWriter);

            String imagePrefix = "orig_" + chrom1Name + "_" + box1XIndex + "_" + chrom2Name + "_" + box2XIndex + "_matrix.png";
            GrindUtils.saveGrindMatrixDataToImage(imagePrefix, negImgPath, compositeMatrix, negImgWriter, false);
            GrindUtils.saveGrindMatrixDataToImage(imagePrefix + "_labels.png", negImgPath, labelsMatrix, negImgLabelWriter, true);

            for (int k = 0; k < numManipulations; k++) {
                Pair<double[][], double[][]> alteredMatrices = GrindUtils.randomlyManipulateMatrix(compositeMatrix, labelsMatrix);
                compositeMatrix = alteredMatrices.getFirst();
                labelsMatrix = alteredMatrices.getSecond();

                filePrefix = "dstrt_" + chrom1Name + "_" + box1XIndex + "_" + chrom2Name + "_" + box2XIndex + "_" + k + "_matrix.txt";
                GrindUtils.saveGrindMatrixDataToFile(filePrefix, posPath, compositeMatrix, posDataWriter);
                GrindUtils.saveGrindMatrixDataToFile(filePrefix + "_labels.txt", posPath, labelsMatrix, posLabelWriter);

                filePrefix = "dstrt_" + chrom1Name + "_" + box1XIndex + "_" + chrom2Name + "_" + box2XIndex + "_" + k + "_matrix.png";
                GrindUtils.saveGrindMatrixDataToImage(filePrefix, posImgPath, compositeMatrix, posImgWriter, false);
                GrindUtils.saveGrindMatrixDataToImage(filePrefix + "_labels.png", posImgPath, labelsMatrix, posImgLabelWriter, true);
            }

        } catch (Exception e) {

        }

    }
}
