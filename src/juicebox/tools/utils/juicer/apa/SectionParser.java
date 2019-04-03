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

package juicebox.tools.utils.juicer.apa;

import juicebox.data.*;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.Feature2DParser;
import juicebox.track.feature.FeatureFunction;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationHandler;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.RealMatrix;
import org.broad.igv.feature.Chromosome;

import java.awt.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.*;

class SectionParser {

    public static void buildLoopSlicesRandom(final String savepath, String loopListPath, final int maxk, String hicFilePaths) {
        final Random generator = new Random();

        //String loopListPath = "";

        File file = new File(savepath);
        if (!file.isDirectory()) {
            file.mkdir();
        }


        final Dataset ds = HiCFileTools.extractDatasetForCLT(Arrays.asList(hicFilePaths.split("\\+")), false);
        final ChromosomeHandler chromosomeHandler = ds.getChromosomeHandler();
        final NormalizationType norm = NormalizationHandler.KR;

        final int resolution = 5000;

        final int submatrixSize = 33;
        final int halfwidth = submatrixSize / 2;

        try {
            final Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(savepath + "all_file_names.txt"), StandardCharsets.UTF_8));

            Feature2DList features = Feature2DParser.loadFeatures(loopListPath, chromosomeHandler, false, null, false);

            features.processLists(new FeatureFunction() {
                @Override
                public void process(String chr, List<Feature2D> feature2DList) {

                    System.out.println("Doing " + chr);

                    Chromosome chrom = chromosomeHandler.getChromosomeFromName(feature2DList.get(0).getChr1());

                    Matrix matrix = ds.getMatrix(chrom, chrom);
                    if (matrix == null) return;

                    HiCZoom zoom = ds.getZoomForBPResolution(resolution);
                    final MatrixZoomData zd = matrix.getZoomData(zoom);

                    if (zd == null) return;

                    for (Feature2D feature2D : feature2DList) {
                        int i0 = feature2D.getMidPt1() / resolution - halfwidth;
                        int j0 = feature2D.getMidPt2() / resolution - halfwidth;

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
                                        i, i + submatrixSize,
                                        j, j + submatrixSize, submatrixSize, submatrixSize, norm);
                                if (MatrixTools.sum(localizedRegionData.getData()) > 0) {

                                    String exactFileName = chrom.getName() + "_" + i + "_" + j + ".txt";

                                    //process
                                    //DescriptiveStatistics yStats = statistics(localizedRegionData.getData());
                                    //mm = (m-yStats.getMean())/Math.max(yStats.getStandardDeviation(),1e-7);
                                    //ZscoreLL = (centralVal - yStats.getMean()) / yStats.getStandardDeviation();

                                    saveMatrixText2(savepath + exactFileName, localizedRegionData);
                                    writer.write(exactFileName + "\n");
                                }
                            } catch (Exception e) {

                            }
                        }
                    }

                }
            });

            writer.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }


    }

    public static void buildLoopSlices1(final String savepath, String loopListPath, String hicFilePaths) {


        // /Users/muhammad/Desktop/examples/good_vs_evil_165000_v2_BBB.bedpe
        // /Users/muhammad/Desktop/examples/combined_short_peaks_f7.txt


        File file = new File(savepath);
        if (!file.isDirectory()) {
            file.mkdir();
        }


        final Dataset ds = HiCFileTools.extractDatasetForCLT(Arrays.asList(hicFilePaths.split("\\+")), false);
        final ChromosomeHandler chromosomeHandler = ds.getChromosomeHandler();
        final NormalizationType norm = NormalizationHandler.KR;

        final int resolution = 5000;

        final int submatrixSize = 33;

        try {
            final Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(savepath + "all_file_names.txt"), StandardCharsets.UTF_8));

            Feature2DList features = Feature2DParser.loadFeatures(loopListPath, chromosomeHandler, false, null, false);

            features.processLists(new FeatureFunction() {
                @Override
                public void process(String chr, List<Feature2D> feature2DList) {

                    System.out.println("Doing " + chr);

                    Chromosome chrom = chromosomeHandler.getChromosomeFromName(feature2DList.get(0).getChr1());

                    Matrix matrix = ds.getMatrix(chrom, chrom);
                    if (matrix == null) return;

                    HiCZoom zoom = ds.getZoomForBPResolution(resolution);
                    final MatrixZoomData zd = matrix.getZoomData(zoom);

                    if (zd == null) return;

                    for (Feature2D feature2D : feature2DList) {
                        int i = feature2D.getMidPt1() / resolution - 16;
                        int j = feature2D.getMidPt2() / resolution - 16;

                        try {
                            RealMatrix localizedRegionData = HiCFileTools.extractLocalBoundedRegion(zd,
                                    i, i + submatrixSize,
                                    j, j + submatrixSize, submatrixSize, submatrixSize, norm);
                            if (MatrixTools.sum(localizedRegionData.getData()) > 0) {

                                String exactFileName = chrom.getName() + "_" + i + "_" + j + ".txt";

                                //process
                                //DescriptiveStatistics yStats = statistics(localizedRegionData.getData());
                                //mm = (m-yStats.getMean())/Math.max(yStats.getStandardDeviation(),1e-7);
                                //ZscoreLL = (centralVal - yStats.getMean()) / yStats.getStandardDeviation();

                                saveMatrixText2(savepath + exactFileName, localizedRegionData);
                                writer.write(exactFileName + "\n");
                            }
                        } catch (Exception e) {

                        }
                    }

                }
            });

            writer.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }

    }


    public static void buildRandomLooplistV2(String outputfile, String hicpath) {

        Random generator = new Random();
        Dataset ds = HiCFileTools.extractDatasetForCLT(Arrays.asList(hicpath), false);
        ChromosomeHandler chromosomeHandler = ds.getChromosomeHandler();

        Feature2DList badlist = new Feature2DList();

        for (final Chromosome chromosome : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {
            if (chromosome.getName().toLowerCase().contains("m") || chromosome.getName().toLowerCase().contains("y"))
                continue;

            System.out.println("Start " + chromosome.getName());
            List<Feature2D> badFeaturesForChromosome = new ArrayList<>();

            for (int i = 0; i < 1000; i++) {
                int x1 = generator.nextInt(chromosome.getLength() - 100000);
                int y1 = x1 + generator.nextInt(100000);
                Feature2D feature = new Feature2D(Feature2D.FeatureType.PEAK, chromosome.getName(), x1, x1 + 33,
                        chromosome.getName(), y1, y1 + 33, Color.BLACK, new HashMap<String, String>());
                badFeaturesForChromosome.add(feature);
            }

            for (int i = 0; i < 1000; i++) {
                int x1 = generator.nextInt(chromosome.getLength() - 5000000);
                int y1 = x1 + generator.nextInt(5000000);
                Feature2D feature = new Feature2D(Feature2D.FeatureType.PEAK, chromosome.getName(), x1, x1 + 33,
                        chromosome.getName(), y1, y1 + 33, Color.BLACK, new HashMap<String, String>());
                badFeaturesForChromosome.add(feature);
            }

            for (int i = 0; i < 1000; i++) {
                int x1 = generator.nextInt(chromosome.getLength() - 20000000);
                int y1 = x1 + generator.nextInt(20000000);
                Feature2D feature = new Feature2D(Feature2D.FeatureType.PEAK, chromosome.getName(), x1, x1 + 33,
                        chromosome.getName(), y1, y1 + 33, Color.BLACK, new HashMap<String, String>());
                badFeaturesForChromosome.add(feature);
            }

            for (int i = 0; i < 1000; i++) {
                int x1 = generator.nextInt(chromosome.getLength() - 40);
                int y1 = x1 + generator.nextInt(chromosome.getLength() - x1);
                Feature2D feature = new Feature2D(Feature2D.FeatureType.PEAK, chromosome.getName(), x1, x1 + 33,
                        chromosome.getName(), y1, y1 + 33, Color.BLACK, new HashMap<String, String>());
                badFeaturesForChromosome.add(feature);
            }
            System.out.println("End " + chromosome.getName());

            badlist.addByKey(Feature2DList.getKey(chromosome, chromosome), badFeaturesForChromosome);
        }

        badlist.exportFeatureList(new File(outputfile), false, Feature2DList.ListFormat.NA);

    }

    public static void buildRandomLooplist(String hicpath, String outputpath) {

        Random generator = new Random();
        Dataset ds = HiCFileTools.extractDatasetForCLT(Arrays.asList(hicpath), false);
        ChromosomeHandler chromosomeHandler = ds.getChromosomeHandler();

        Feature2DList badlist = new Feature2DList();

        for (final Chromosome chromosome : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {
            if (chromosome.getName().toLowerCase().contains("m") || chromosome.getName().toLowerCase().contains("y"))
                continue;

            System.out.println("Start " + chromosome.getName());
            List<Feature2D> badFeaturesForChromosome = new ArrayList<>();

            for (int i = 0; i < 20000; i++) {
                int x1 = generator.nextInt(chromosome.getLength() - 5000000);
                int y1 = x1 + generator.nextInt(5000000);
                Feature2D feature = new Feature2D(Feature2D.FeatureType.PEAK, chromosome.getName(), x1, x1 + 33,
                        chromosome.getName(), y1, y1 + 33, Color.BLACK, new HashMap<String, String>());
                badFeaturesForChromosome.add(feature);
            }

            for (int i = 0; i < 20000; i++) {
                int x1 = generator.nextInt(chromosome.getLength() - 40);
                int y1 = x1 + generator.nextInt(chromosome.getLength() - x1);
                Feature2D feature = new Feature2D(Feature2D.FeatureType.PEAK, chromosome.getName(), x1, x1 + 33,
                        chromosome.getName(), y1, y1 + 33, Color.BLACK, new HashMap<String, String>());
                badFeaturesForChromosome.add(feature);
            }
            System.out.println("End " + chromosome.getName());

            badlist.addByKey(Feature2DList.getKey(chromosome, chromosome), badFeaturesForChromosome);
        }

        badlist.exportFeatureList(new File(outputpath), false, Feature2DList.ListFormat.NA);

    }


    public static void buildSlices(String[] hicFilePathsArray, String[] savepathArray) {

        for (int k = 0; k < hicFilePathsArray.length; k++) {
            String hicFilePaths = hicFilePathsArray[k];
            String savepath = savepathArray[k];

            File file = new File(savepath);
            if (!file.isDirectory()) {
                file.mkdir();
            }

            Dataset ds = HiCFileTools.extractDatasetForCLT(Arrays.asList(hicFilePaths.split("\\+")), false);
            ChromosomeHandler chromosomeHandler = ds.getChromosomeHandler();
            NormalizationType norm = NormalizationHandler.KR;

            int resolution = 5000;

            int submatrixSize = 33;
            int incrementSize = 16;

            Writer writer = null;
            try {
                writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(savepath + "all_file_names.txt"), StandardCharsets.UTF_8));

                for (final Chromosome chromosome : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {

                    // skip these matrices
                    Matrix matrix = ds.getMatrix(chromosome, chromosome);
                    if (matrix == null) continue;

                    if (chromosome.getName().toLowerCase().contains("m") || chromosome.getName().toLowerCase().contains("y"))
                        continue;

                    if (chromosome.getIndex() > 2) continue;

                    HiCZoom zoom = ds.getZoomForBPResolution(resolution);
                    final MatrixZoomData zd = matrix.getZoomData(zoom);

                    if (zd == null) continue;

                    int maxBin = chromosome.getLength() / resolution + 1;

                    for (int i = 0; i < maxBin; i += incrementSize) {
                        for (int j = i; j < 1600 + i; j += incrementSize) {
                            try {
                                RealMatrix localizedRegionData = HiCFileTools.extractLocalBoundedRegion(zd, i, i + submatrixSize, j, j + submatrixSize, submatrixSize, submatrixSize, norm);
                                if (MatrixTools.sum(localizedRegionData.getData()) > 0) {

                                    String exactFileName = chromosome.getName() + "_" + i + "_" + j + ".txt";

                                    //process
                                    //DescriptiveStatistics yStats = statistics(localizedRegionData.getData());
                                    //mm = (m-yStats.getMean())/Math.max(yStats.getStandardDeviation(),1e-7);
                                    //ZscoreLL = (centralVal - yStats.getMean()) / yStats.getStandardDeviation();

                                    saveMatrixText2(savepath + exactFileName, localizedRegionData);
                                    writer.write(exactFileName + "\n");
                                }
                            } catch (Exception e) {

                            }
                        }
                    }
                }

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
    }

    private static void saveMatrixText2(String filename, RealMatrix realMatrix) {
        Writer writer = null;
        try {
            writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), StandardCharsets.UTF_8));
            double[][] matrix = realMatrix.getData();
            for (double[] row : matrix) {
                String s = Arrays.toString(row);//.replaceAll().replaceAll("]","").trim();
                s = s.replaceAll("\\[", "").replaceAll("\\]", "").trim();
                writer.write(s + "\n");
            }
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
}
