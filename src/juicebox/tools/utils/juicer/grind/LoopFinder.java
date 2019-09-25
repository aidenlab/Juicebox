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
import juicebox.tools.utils.common.UNIXTools;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.Feature2DTools;
import juicebox.track.feature.FeatureFunction;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.RealMatrix;
import org.broad.igv.feature.Chromosome;

import java.awt.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.*;


public class LoopFinder implements RegionFinder {

    private Integer x;
    private Integer y;
    private Integer z;
    private Integer stride;
    private Dataset ds;
    private Feature2DList features;
    private String originalPath;
    private NormalizationType norm;
    private Set<Integer> resolutions;
    private ChromosomeHandler chromosomeHandler;
    private int overallWidth;
    private boolean onlyMakePositiveExamples, dimensionOfLabelIsSameAsOutput;

    public LoopFinder(int x, int y, int z, int stride, Dataset ds, Feature2DList features, File outputDirectory, ChromosomeHandler chromosomeHandler, NormalizationType norm,
                      boolean useObservedOverExpected, boolean dimensionOfLabelIsSameAsOutput, Set<Integer> resolutions, boolean onlyMakePositiveExamples) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.stride = stride;
        this.ds = ds;
        this.features = features;
        this.originalPath = outputDirectory.getPath();
        this.norm = norm;
        this.resolutions = resolutions;
        this.chromosomeHandler = chromosomeHandler;
        this.onlyMakePositiveExamples = onlyMakePositiveExamples;
        this.dimensionOfLabelIsSameAsOutput = dimensionOfLabelIsSameAsOutput;
    }

    @Override
    public void makeExamples() {
        if (dimensionOfLabelIsSameAsOutput) {
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


        final Feature2DHandler feature2DHandler = new Feature2DHandler(features);

        for (int resolution : resolutions) {

            features.parallelizedProcessLists(new FeatureFunction() {
                @Override
                public void process(String chr, List<Feature2D> feature2DList) {
                    Chromosome chrom = chromosomeHandler.getChromosomeFromName(feature2DList.get(0).getChr1());

                    Matrix matrix = ds.getMatrix(chrom, chrom);
                    if (matrix == null) return;

                    HiCZoom zoom = ds.getZoomForBPResolution(resolution);
                    final MatrixZoomData zd = matrix.getZoomData(zoom);

                    if (zd == null) return;

                    System.out.println("Currently on: " + chr);



                    try {
                        int tenKCounter = 0;
                        String resPath = originalPath + "/positives_res" + resolution;
                        UNIXTools.makeDir(resPath);
                        String posPath = resPath + "/chr" + chrom.getName() + "_" + tenKCounter + "k";
                        UNIXTools.makeDir(posPath);
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
                                        StripeFinder.getTrainingDataAndSaveToFile(ds, norm, zd, chrom, rowIndex, colIndex, resolution, feature2DHandler, x, y,
                                                posPath, null, posWriter, posLabelWriter, null, false,
                                                false, true, true, true);
                                        printingCounter += 3;
                                    } catch (Exception e) {
                                        System.err.println("Error reading from row " + rowIndex + " col " + colIndex + " at res " + resolution);
                                    }
                                }
                            }
                            if (printingCounter > 10000) {
                                tenKCounter++;
                                printingCounter = 0;
                                posPath = resPath + "/chr" + chrom.getName() + "_" + tenKCounter + "k";
                                UNIXTools.makeDir(posPath);
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
        final Random generator = new Random();

        File file = new File(originalPath);
        if (!file.isDirectory()) {
            file.mkdir();
        }

        final int resolution = (int) resolutions.toArray()[0];

        final int halfWidthI = x / 2;
        final int halfWidthJ = y / 2;
        final int maxk = Math.max(z / features.getNumTotalFeatures(), 1);

        features.processLists(new FeatureFunction() {
            @Override
            public void process(String chr, List<Feature2D> feature2DList) {
                Chromosome chrom = chromosomeHandler.getChromosomeFromName(feature2DList.get(0).getChr1());
                Matrix matrix = ds.getMatrix(chrom, chrom);
                if (matrix == null) return;
                HiCZoom zoom = ds.getZoomForBPResolution(resolution);
                final MatrixZoomData zd = matrix.getZoomData(zoom);
                if (zd == null) return;

                System.out.println("Currently on: " + chr);
                try {
                    final Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(originalPath + "/pos_res_" + resolution + "_" + chrom.getName() + "_file_names.txt"), StandardCharsets.UTF_8));

                    for (Feature2D feature2D : feature2DList) {
                        int i0 = feature2D.getMidPt1() / resolution - halfWidthI;
                        int j0 = feature2D.getMidPt2() / resolution - halfWidthJ;

                        for (int k = 0; k < maxk; k++) {

                            int di = 10 - generator.nextInt(21);
                            while (di == 0) {
                                di = 10 - generator.nextInt(21);
                            }

                            int dj = 10 - generator.nextInt(21);
                            while (dj == 0) {
                                dj = 10 - generator.nextInt(21);
                            }

                            int i = i0 + di;
                            int j = j0 + dj;

                            try {
                                RealMatrix localizedRegionData = HiCFileTools.extractLocalBoundedRegion(zd,
                                        i, i + x,
                                        j, j + y, x, y, norm, true);
                                if (MatrixTools.sum(localizedRegionData.getData()) > 0) {

                                    String exactFileName = chrom.getName() + "_" + i + "_" + j + ".txt";

                                    //process
                                    //DescriptiveStatistics yStats = statistics(localizedRegionData.getData());
                                    //mm = (m-yStats.getMean())/Math.max(yStats.getStandardDeviation(),1e-7);
                                    //ZscoreLL = (centralVal - yStats.getMean()) / yStats.getStandardDeviation();

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
        final int maxk = z / features.getNumTotalFeatures();


        for (final Chromosome chromosome : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {
            if (chromosome.getName().toLowerCase().contains("m") || chromosome.getName().toLowerCase().contains("y")) {
                continue;
            }

            System.out.println("Start " + chromosome.getName());
            List<Feature2D> badFeaturesForChromosome = new ArrayList<>();

            for (int i = 0; i < 1000; i++) {
                int x1 = generator.nextInt(chromosome.getLength() - 100000);
                int y1 = x1 + generator.nextInt(100000);
                Feature2D feature = new Feature2D(Feature2D.FeatureType.PEAK, chromosome.getName(), x1, x1 + 33,
                        chromosome.getName(), y1, y1 + 33, Color.BLACK, new HashMap<>());
                badFeaturesForChromosome.add(feature);
            }

            for (int i = 0; i < 1000; i++) {
                int x1 = generator.nextInt(chromosome.getLength() - 5000000);
                int y1 = x1 + generator.nextInt(5000000);
                Feature2D feature = new Feature2D(Feature2D.FeatureType.PEAK, chromosome.getName(), x1, x1 + 33,
                        chromosome.getName(), y1, y1 + 33, Color.BLACK, new HashMap<>());
                badFeaturesForChromosome.add(feature);
            }

            for (int i = 0; i < 1000; i++) {
                int x1 = generator.nextInt(chromosome.getLength() - 20000000);
                int y1 = x1 + generator.nextInt(20000000);
                Feature2D feature = new Feature2D(Feature2D.FeatureType.PEAK, chromosome.getName(), x1, x1 + 33,
                        chromosome.getName(), y1, y1 + 33, Color.BLACK, new HashMap<>());
                badFeaturesForChromosome.add(feature);
            }

            for (int i = 0; i < 1000; i++) {
                int x1 = generator.nextInt(chromosome.getLength() - 40);
                int y1 = x1 + generator.nextInt(chromosome.getLength() - x1);
                Feature2D feature = new Feature2D(Feature2D.FeatureType.PEAK, chromosome.getName(), x1, x1 + 33,
                        chromosome.getName(), y1, y1 + 33, Color.BLACK, new HashMap<>());
                badFeaturesForChromosome.add(feature);
            }
            badlist.addByKey(Feature2DList.getKey(chromosome, chromosome), badFeaturesForChromosome);
        }
        Feature2D.tolerance = 100000;
        Feature2DList featureList = Feature2DTools.subtract(badlist, features);

        featureList.processLists(new FeatureFunction() {
            @Override
            public void process(String chr, List<Feature2D> feature2DList) {

                Chromosome chrom = chromosomeHandler.getChromosomeFromName(chr);
                Matrix matrix = ds.getMatrix(chrom, chrom);
                if (matrix == null) {
                    return;
                }
                HiCZoom zoom = ds.getZoomForBPResolution(resolution);
                final MatrixZoomData zd = matrix.getZoomData(zoom);
                if (zd == null) {
                    return;
                }

                try {
                    final Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(originalPath + "/pos_res_" + resolution + "_" + chrom.getName() + "_file_names.txt"), StandardCharsets.UTF_8));

                    for (Feature2D feature2D : feature2DList) {
                        int i0 = feature2D.getMidPt1() / resolution - halfWidthI;
                        int j0 = feature2D.getMidPt2() / resolution - halfWidthJ;

                        for (int k = 0; k < maxk; k++) {

                            int di = 10 - generator.nextInt(21);
                            while (di == 0) {
                                di = 10 - generator.nextInt(21);
                            }

                            int dj = 10 - generator.nextInt(21);
                            while (dj == 0) {
                                dj = 10 - generator.nextInt(21);
                            }

                            int i = i0 + di;
                            int j = j0 + dj;

                            RealMatrix localizedRegionData = HiCFileTools.extractLocalBoundedRegion(zd,
                                    i, i + x,
                                    j, j + y, x, y, norm, true);
                            if (MatrixTools.sum(localizedRegionData.getData()) > 0) {

                                String exactFileName = chrom.getName() + "_" + i + "_" + j + ".txt";

                                //process
                                //DescriptiveStatistics yStats = statistics(localizedRegionData.getData());
                                //mm = (m-yStats.getMean())/Math.max(yStats.getStandardDeviation(),1e-7);
                                //ZscoreLL = (centralVal - yStats.getMean()) / yStats.getStandardDeviation();

                                MatrixTools.saveMatrixTextV2(originalPath + exactFileName, localizedRegionData);
                                writer.write(exactFileName + "\n");
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
}

