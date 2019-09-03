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

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.Set;

public class DistortionFinder implements RegionFinder {


    private Integer imgSliceWidth, imgHalfSliceWidth;
    private Integer numExamples;
    private Dataset ds;
    private String path;
    private ChromosomeHandler chromosomeHandler;
    private NormalizationType norm;
    private boolean useObservedOverExpected;
    private boolean useDenseLabels;
    private Set<Integer> resolutions;
    private int stride;

    // grind -k KR -r 5000,10000,25000,100000 --stride 3 -c 1,2,3 --dense-labels --distort <hic file> null <64,64,1000> <directory>

    public DistortionFinder(int imgSliceWidth, Dataset ds, File outputDirectory,
                            ChromosomeHandler chromosomeHandler, NormalizationType norm,
                            boolean useObservedOverExpected, boolean useDenseLabels, Set<Integer> resolutions, int stride) {

        this.imgSliceWidth = imgSliceWidth;
        imgHalfSliceWidth = imgSliceWidth / 2;
        this.ds = ds;
        this.path = outputDirectory.getPath();
        this.chromosomeHandler = chromosomeHandler;
        this.norm = norm;
        this.useObservedOverExpected = useObservedOverExpected;
        this.useDenseLabels = useDenseLabels;
        this.resolutions = resolutions;
        this.stride = stride;
    }

    @Override
    public void makeExamples() {

        final String negPath = path + "/negative";
        final String posPath = path + "/positive";
        UNIXTools.makeDir(negPath);
        UNIXTools.makeDir(posPath);

        try {

            final Writer posWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path + "/pos_file_names.txt"), StandardCharsets.UTF_8));
            final Writer negWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path + "/neg_file_names.txt"), StandardCharsets.UTF_8));
            final Writer posLabelWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path + "/pos_label_file_names.txt"), StandardCharsets.UTF_8));
            final Writer negLabelWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path + "/neg_label_file_names.txt"), StandardCharsets.UTF_8));

            for (int resolution : resolutions) {
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
                            iterateAcrossIntraChromosomalRegion(zd, chromI, resolution,
                                    posPath, negPath, posWriter, posLabelWriter, negWriter, negLabelWriter);
                        } else {
                            // is Inter
                            matrixZoomDataI = HiCFileTools.getMatrixZoomData(ds, chromI, chromI, zoom);
                            matrixZoomDataJ = HiCFileTools.getMatrixZoomData(ds, chromJ, chromJ, zoom);
                            if (matrixZoomDataI == null) continue;
                            if (matrixZoomDataJ == null) continue;

                            iterateBetweenInterChromosomalRegions(zd, matrixZoomDataI, matrixZoomDataJ, chromI, chromJ, resolution,
                                    posPath, negPath, posWriter, posLabelWriter, negWriter, negLabelWriter);
                        }
                    }
                }
            }
            posWriter.close();
            negWriter.close();
            posLabelWriter.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }


    private void iterateAcrossIntraChromosomalRegion(MatrixZoomData zd, Chromosome chrom, int resolution, String posPath, String negPath, Writer posWriter, Writer posLabelWriter,
                                                     Writer negWriter, Writer negLabelWriter) {

        // sliding along the diagonal
        int maxChrLength = (chrom.getLength() / resolution);
        for (int posIndex1 = 0; posIndex1 < maxChrLength - imgSliceWidth; posIndex1 += stride) {
            for (int posIndex2 = posIndex1 + imgHalfSliceWidth; posIndex2 < maxChrLength; posIndex2 += stride) {
                getTrainingDataAndSaveToFile(zd, zd, zd, posIndex1, posIndex2,
                        posPath, negPath, posWriter, posLabelWriter, negWriter, negLabelWriter);
            }
        }
    }

    private void iterateBetweenInterChromosomalRegions(MatrixZoomData zd1, MatrixZoomData zd2, MatrixZoomData zd12, Chromosome chrom1, Chromosome chrom2,
                                                       int resolution, String posPath, String negPath, Writer posWriter, Writer posLabelWriter,
                                                       Writer negWriter, Writer negLabelWriter) {

        // iterating across both chromosomes
        int maxChrLength1 = (chrom1.getLength() / resolution);
        int maxChrLength2 = (chrom2.getLength() / resolution);

        for (int posIndex1 = 0; posIndex1 < maxChrLength1 - imgHalfSliceWidth; posIndex1 += stride) {
            for (int posIndex2 = 0; posIndex2 < maxChrLength2 - imgHalfSliceWidth; posIndex2 += stride) {
                getTrainingDataAndSaveToFile(zd1, zd2, zd12, posIndex1, posIndex2,
                        posPath, negPath, posWriter, posLabelWriter, negWriter, negLabelWriter);
            }
        }
    }


    private void getTrainingDataAndSaveToFile(MatrixZoomData zd1, MatrixZoomData zd2, MatrixZoomData zd12,
                                              int box1XIndex, int box2XIndex,
                                              String posPath, String negPath,
                                              Writer posDataWriter, Writer posLabelWriter,
                                              Writer negDataWriter, Writer negLabelWriter) {

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

            double[][] labelsMatrix = GrindUtils.generateDefaultDistortionLabelsFile(compositeMatrix.length, 4);

            // todo GrindUtils.cleanUpLabelsMatrixBasedOnData(labelsMatrix, compositeMatrix);


        } catch (Exception e) {

        }

    }
}
