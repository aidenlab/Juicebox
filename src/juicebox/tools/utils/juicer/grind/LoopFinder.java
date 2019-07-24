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

import static juicebox.tools.utils.juicer.grind.SectionParser.saveMatrixText2;

public class LoopFinder implements RegionFinder {

    private Integer x;
    private Integer y;
    private Integer z;
    private Dataset ds;
    private Feature2DList features;
    private String path;
    private NormalizationType norm;
    private Set<Integer> resolutions;
    private Writer writer = null;

    public LoopFinder(int x, int y, int z, Dataset ds, Feature2DList features, File outputDirectory, Set<String> givenChromosomes, NormalizationType norm,
                      boolean useObservedOverExpected, boolean useDenseLabels, Set<Integer> resolutions) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.ds = ds;
        this.features = features;
        this.path = outputDirectory.getPath();
        this.norm = norm;
        this.resolutions = resolutions;
        try {
            writer =
                new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path + "all_file_names.txt"),
                    StandardCharsets.UTF_8));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void makePositiveExamples() {
        final Random generator = new Random();

        //String loopListPath = "";

        File file = new File(path);
        if (!file.isDirectory()) {
            file.mkdir();
        }


        final ChromosomeHandler chromosomeHandler = ds.getChromosomeHandler();

        final int resolution = (int) resolutions.toArray()[0];

        final int halfWidthI = x / 2;
        final int halfWidthJ = y / 2;
        final int maxk = z / features.getNumTotalFeatures();

        try {
            // Feature2DList features = Feature2DParser.loadFeatures(loopListPath, chromosomeHandler, false, null, false);

            features.processLists(new FeatureFunction() {
                @Override
                public void process(String chr, List<Feature2D> feature2DList) {

                    System.out.println("Currently on: " + chr);

                    Chromosome chrom = chromosomeHandler.getChromosomeFromName(feature2DList.get(0).getChr1());

                    Matrix matrix = ds.getMatrix(chrom, chrom);
                    if (matrix == null) return;

                    HiCZoom zoom = ds.getZoomForBPResolution(resolution);
                    final MatrixZoomData zd = matrix.getZoomData(zoom);

                    if (zd == null) return;

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
                                        j, j + y, x, y, norm);
                                if (MatrixTools.sum(localizedRegionData.getData()) > 0) {

                                    String exactFileName = chrom.getName() + "_" + i + "_" + j + ".txt";

                                    //process
                                    //DescriptiveStatistics yStats = statistics(localizedRegionData.getData());
                                    //mm = (m-yStats.getMean())/Math.max(yStats.getStandardDeviation(),1e-7);
                                    //ZscoreLL = (centralVal - yStats.getMean()) / yStats.getStandardDeviation();

                                    saveMatrixText2(path + exactFileName, localizedRegionData);
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

    @Override
    public void makeNegativeExamples() {
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
                                    j, j + y, x, y, norm);
                            if (MatrixTools.sum(localizedRegionData.getData()) > 0) {

                                String exactFileName = chrom.getName() + "_" + i + "_" + j + ".txt";

                                //process
                                //DescriptiveStatistics yStats = statistics(localizedRegionData.getData());
                                //mm = (m-yStats.getMean())/Math.max(yStats.getStandardDeviation(),1e-7);
                                //ZscoreLL = (centralVal - yStats.getMean()) / yStats.getStandardDeviation();

                                saveMatrixText2(path + exactFileName, localizedRegionData);
                                writer.write(exactFileName + "\n");
                            }
                        } catch (Exception e) {

                        }
                    }
                }

            }
        });


        try {
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }



    }

