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

import juicebox.data.ChromosomeHandler;
import juicebox.data.HiCFileTools;
import juicebox.data.MatrixZoomData;
import juicebox.mapcolorui.Feature2DHandler;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.tools.utils.common.UNIXTools;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.Feature2DTools;
import juicebox.track.feature.FeatureFunction;
import org.apache.commons.math.linear.RealMatrix;
import org.broad.igv.feature.Chromosome;

import java.awt.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;


public class IterateOnFeatureListFinder extends RegionFinder {

    public IterateOnFeatureListFinder(ParameterConfigurationContainer container) {
        super(container);
    }

    @Override
    public void makeExamples() {
        if (useDenseLabelsNotBinary) {
            makeAllPositiveExamples();
        } else {
            makeRandomPositiveExamples();
        }
        if (!onlyMakePositiveExamples) {
            makeNegativeExamples();
        }
    }

    private void makeAllPositiveExamples() {

        File file = new File(originalPath);
        if (!file.isDirectory()) {
            file.mkdir();
        }

        final int fullWidthI = x;
        final int fullWidthJ = y;

        final Feature2DHandler feature2DHandler = new Feature2DHandler(inputFeature2DList);

        for (int resolution : resolutions) {

            inputFeature2DList.parallelizedProcessLists(new FeatureFunction() {
                @Override
                public void process(String chr, List<Feature2D> feature2DList) {
                    Chromosome chrom = chromosomeHandler.getChromosomeFromName(feature2DList.get(0).getChr1());

                    final MatrixZoomData zd = HiCFileTools.getMatrixZoomData(ds, chrom, chrom, resolution);
                    if (zd == null) return;
                    System.out.println("Currently on: " + chr);

                    try {
                        int tenKCounter = 0;
                        String resPath = UNIXTools.makeDir(originalPath + "/positives_res" + resolution);
                        String posPath = UNIXTools.makeDir(resPath + "/chr" + chrom.getName() + "_" + tenKCounter + "k");

                        int printingCounter = 0;
                        Writer posWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(posPath + ".txt"), StandardCharsets.UTF_8));
                        Writer posLabelWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(posPath + "_label.txt"), StandardCharsets.UTF_8));

                        for (Feature2D feature2D : feature2DList) {
                            int i0 = Math.max(0, feature2D.getMidPt1() / resolution - fullWidthI);
                            int j0 = Math.max(0, feature2D.getMidPt2() / resolution - fullWidthJ);
                            int iMax = Math.min(feature2D.getMidPt1() / resolution + fullWidthI, chrom.getLength() / resolution);
                            int jMax = Math.min(feature2D.getMidPt2() / resolution + fullWidthJ, chrom.getLength() / resolution);

                            for (int rowIndex = i0; rowIndex < iMax; rowIndex += stride) {
                                for (int colIndex = j0; colIndex < jMax; colIndex += stride) {
                                    try {
                                        getTrainingDataAndSaveToFile(zd, chrom, rowIndex, colIndex, resolution, feature2DHandler, x, y,
                                                posPath, null, posWriter, posLabelWriter, null, false);
                                        printingCounter += 3;
                                    } catch (Exception e) {
                                        System.err.println("Error reading from row " + rowIndex + " col " + colIndex + " at res " + resolution);
                                    }
                                }
                            }
                            if (printingCounter > 10000) {
                                tenKCounter++;
                                printingCounter = 0;
                                posPath = UNIXTools.makeDir(resPath + "/chr" + chrom.getName() + "_" + tenKCounter + "k");
                                posWriter.close();
                                posLabelWriter.close();
                                posWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(posPath + ".txt"), StandardCharsets.UTF_8));
                                posLabelWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(posPath + "_label.txt"), StandardCharsets.UTF_8));
                            }
                        }
                        posWriter.close();
                        posLabelWriter.close();
                    } catch (Exception ex) {
                        ex.printStackTrace();
                    }
                    System.out.println("Done with: " + chr);
                }
            });
        }
    }

    private void makeRandomPositiveExamples() {

        File file = new File(originalPath);
        if (!file.isDirectory()) {
            file.mkdir();
        }

        final int resolution = (int) resolutions.toArray()[0];

        final int halfWidthI = x / 2;
        final int halfWidthJ = y / 2;
        final int maxk = Math.max(z / inputFeature2DList.getNumTotalFeatures(), 1);

        inputFeature2DList.processLists(new FeatureFunction() {
            @Override
            public void process(String chr, List<Feature2D> feature2DList) {
                Chromosome chrom = chromosomeHandler.getChromosomeFromName(feature2DList.get(0).getChr1());
                final MatrixZoomData zd = HiCFileTools.getMatrixZoomData(ds, chrom, chrom, resolution);
                if (zd == null) return;

                System.out.println("Currently on: " + chr);
                try {
                    final Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(originalPath + "/pos_res_" + resolution + "_" + chrom.getName() + "_file_names.txt"), StandardCharsets.UTF_8));

                    for (Feature2D feature2D : feature2DList) {
                        int i0 = feature2D.getMidPt1() / resolution - halfWidthI;
                        int j0 = feature2D.getMidPt2() / resolution - halfWidthJ;

                        for (int k = 0; k < maxk; k++) {

                            int i = i0 + getNonZeroPosOrNegIntegerNearby(halfWidthI, 2. / 3.);
                            int j = j0 + getNonZeroPosOrNegIntegerNearby(halfWidthI, 2. / 3.);

                            try {
                                RealMatrix localizedRegionData = HiCFileTools.extractLocalBoundedRegion(zd,
                                        i, i + x, j, j + y, x, y, norm, true);
                                if (MatrixTools.sum(localizedRegionData.getData()) > 0) {

                                    String exactFileName = chrom.getName() + "_" + i + "_" + j + ".txt";

                                    MatrixTools.saveMatrixTextV2(originalPath + exactFileName, localizedRegionData);
                                    writer.write(exactFileName + "\n");
                                }
                            } catch (Exception ignored) {

                            }
                        }
                    }
                    writer.close();
                } catch (Exception ex) {
                    ex.printStackTrace();
                }
            }
        });
    }

    private void makeNegativeExamples() {
        Random generator = new Random();
        ChromosomeHandler chromosomeHandler = ds.getChromosomeHandler();
        Feature2DList badlist = new Feature2DList();
        final int resolution = (int) resolutions.toArray()[0];
        final int halfWidthI = x / 2;
        final int halfWidthJ = y / 2;
        final int maxk = z / inputFeature2DList.getNumTotalFeatures();


        for (final Chromosome chromosome : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {
            if (chromosome.getName().toLowerCase().contains("m") || chromosome.getName().toLowerCase().contains("y")) {
                continue;
            }

            System.out.println("Start " + chromosome.getName());
            List<Feature2D> badFeaturesForChromosome = new ArrayList<>();
            badFeaturesForChromosome.addAll(addRandomlyGeneratedPointsWithinDistanceOfDiagonal(chromosome, 1000, x + y, x, y));
            badFeaturesForChromosome.addAll(addRandomlyGeneratedPointsWithinDistanceOfDiagonal(chromosome, 1000, 100000, x, y));
            badFeaturesForChromosome.addAll(addRandomlyGeneratedPointsWithinDistanceOfDiagonal(chromosome, 1000, 5000000, x, y));
            badFeaturesForChromosome.addAll(addRandomlyGeneratedPointsWithinDistanceOfDiagonal(chromosome, 1000, 20000000, x, y));
            badFeaturesForChromosome.addAll(addRandomlyGeneratedPointsWithinDistanceOfDiagonal(chromosome, 1000, 100000, x, y));

            badlist.addByKey(Feature2DList.getKey(chromosome, chromosome), badFeaturesForChromosome);
        }
        Feature2D.tolerance = 100000;
        Feature2DList actualNegativeExamplesNoLoops = Feature2DTools.subtract(badlist, inputFeature2DList);

        actualNegativeExamplesNoLoops.processLists(new FeatureFunction() {
            @Override
            public void process(String chr, List<Feature2D> feature2DList) {

                Chromosome chrom = chromosomeHandler.getChromosomeFromName(chr);
                final MatrixZoomData zd = HiCFileTools.getMatrixZoomData(ds, chrom, chrom, resolution);
                if (zd == null) return;

                try {
                    final Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(originalPath + "/pos_res_" + resolution + "_" + chrom.getName() + "_file_names.txt"), StandardCharsets.UTF_8));

                    for (Feature2D feature2D : feature2DList) {
                        int i0 = feature2D.getMidPt1() / resolution - halfWidthI;
                        int j0 = feature2D.getMidPt2() / resolution - halfWidthJ;

                        for (int k = 0; k < maxk; k++) {

                            int i = i0 + getNonZeroPosOrNegIntegerNearby(halfWidthI, 2. / 3.);
                            int j = j0 + getNonZeroPosOrNegIntegerNearby(halfWidthI, 2. / 3.);

                            RealMatrix localizedRegionData = HiCFileTools.extractLocalBoundedRegion(zd, i, j, x, y, norm, true);
                            if (MatrixTools.sum(localizedRegionData.getData()) > 0) {

                                String exactFileName = chrom.getName() + "_" + i + "_" + j;
                                GrindUtils.saveGrindMatrixDataToFile(exactFileName, originalPath, localizedRegionData, writer, useTxtInsteadOfNPY);

                            }
                        }
                    }
                    writer.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }

            }
        });
    }

    private List<Feature2D> addRandomlyGeneratedPointsWithinDistanceOfDiagonal(Chromosome chromosome, int numPointsToMake, int distanceNearDiagonal, Integer numRows, Integer numCols) {
        List<Feature2D> featureList = new ArrayList<>();
        for (int i = 0; i < numPointsToMake; i++) {
            int x1 = generator.nextInt(chromosome.getLength() - distanceNearDiagonal);
            int y1 = x1 + generator.nextInt(distanceNearDiagonal);
            Feature2D feature = new Feature2D(Feature2D.FeatureType.PEAK, chromosome.getName(), x1, x1 + numRows,
                    chromosome.getName(), y1, y1 + numCols, Color.BLACK, new HashMap<>());
            featureList.add(feature);
        }
        return featureList;
    }

    private int getNonZeroPosOrNegIntegerNearby(int width, double scaleDownBy) {
        int newWidth = (int) Math.floor(width * scaleDownBy);
        int offset = 0;
        while (offset == 0) {
            offset = (newWidth / 2) - generator.nextInt(newWidth);
        }
        return offset;
    }
}

