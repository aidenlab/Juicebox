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

package juicebox.tools.clt.juicer;

import juicebox.HiCGlobals;
import juicebox.data.*;
import juicebox.data.feature.FeatureFilter;
import juicebox.data.feature.GenomeWideList;
import juicebox.tools.clt.CommandLineParserForJuicer;
import juicebox.tools.clt.JuicerCLT;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.tools.utils.juicer.curse.DataCleaner;
import juicebox.tools.utils.juicer.curse.ExtractingOEDataUtils;
import juicebox.tools.utils.juicer.curse.ScaledGenomeWideMatrix;
import juicebox.tools.utils.juicer.curse.SubcompartmentInterval;
import juicebox.tools.utils.juicer.curse.kmeans.Cluster;
import juicebox.tools.utils.juicer.curse.kmeans.ConcurrentKMeans;
import juicebox.tools.utils.juicer.curse.kmeans.KMeansListener;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.RealMatrix;
import org.broad.igv.feature.Chromosome;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by muhammadsaadshamim on 9/14/15.
 */
public class Curse extends JuicerCLT {

    private boolean doDifferentialClustering = false;
    private int resolution = 100000;
    private Dataset ds;
    private File outputDirectory;
    private AtomicInteger uniqueClusterID = new AtomicInteger(1);
    private int numClusters = 20;
    private double coverageThreshold = 0.7;
    private int maxIters = 10000;
    private double logThreshold = 1.5;

    public Curse() {
        super("curse [-r resolution] [-k NONE/VC/VC_SQRT/KR] <input_HiC_file(s)> <output_file>");
        HiCGlobals.useCache = false;
    }

    @Override
    protected void readJuicerArguments(String[] args, CommandLineParserForJuicer juicerParser) {
        if (args.length != 3) {
            printUsageAndExit();
        }

        NormalizationType preferredNorm = juicerParser.getNormalizationTypeOption();
        if (preferredNorm != null) norm = preferredNorm;

        ds = HiCFileTools.extractDatasetForCLT(Arrays.asList(args[1].split("\\+")), true);
        outputDirectory = HiCFileTools.createValidDirectory(args[2]);

        List<String> possibleResolutions = juicerParser.getMultipleResolutionOptions();
        if (possibleResolutions != null) {
            if (possibleResolutions.size() > 1)
                System.err.println("Only one resolution can be specified for Curse\nUsing " + possibleResolutions.get(0));
            resolution = Integer.parseInt(possibleResolutions.get(0));
        }
    }

    @Override
    public void run() {

        ChromosomeHandler chromosomeHandler = ds.getChromosomeHandler();

        GenomeWideList<SubcompartmentInterval> intraSubcompartments = extractAllInitialIntraSubcompartments(ds, chromosomeHandler);

        File outputFile = new File(outputDirectory, "result_intra_initial.bed");
        intraSubcompartments.simpleExport(outputFile);

        SubcompartmentInterval.collapseGWList(intraSubcompartments);

        File outputFile2 = new File(outputDirectory, "result_intra_initial_collapsed.bed");
        intraSubcompartments.simpleExport(outputFile2);

        GenomeWideList<SubcompartmentInterval> finalSubcompartments = extractFinalGWSubcompartments(ds, chromosomeHandler, intraSubcompartments);
    }


    private GenomeWideList<SubcompartmentInterval> extractFinalGWSubcompartments(Dataset ds, ChromosomeHandler chromosomeHandler,
                                                                                 GenomeWideList<SubcompartmentInterval> intraSubcompartments) {
        final ScaledGenomeWideMatrix gwMatrix = new ScaledGenomeWideMatrix(chromosomeHandler, ds, norm,
                resolution, intraSubcompartments, logThreshold);


        final GenomeWideList<SubcompartmentInterval> finalSubcompartments = new GenomeWideList<>(chromosomeHandler);


        final AtomicBoolean gwRunNotDone = new AtomicBoolean(true);
        System.out.println("preprintingFile");
        if (gwMatrix.getLength() > 0) {

            System.out.println("printingFile");
            File outputFile = new File(outputDirectory, "gw_matrix_data.txt");
            MatrixTools.exportData(gwMatrix.getCleanedData(), outputFile);

            /*
            ConcurrentKMeans kMeans = new ConcurrentKMeans(gwMatrix.getCleanedData(), numClusters,
                    maxIters, 128971L); //Runtime.getRuntime().availableProcessors()/2
            //BasicKMeans kMeans = new BasicKMeans(dataCleaner.getCleanedData(), numClusters, maxIters, 128971L);

            KMeansListener kMeansListener = new KMeansListener() {
                @Override
                public void kmeansMessage(String s) { }

                @Override
                public void kmeansComplete(Cluster[] clusters, long l) {
                    gwRunNotDone.set(false);
                    gwMatrix.processGWKmeansResult(clusters, finalSubcompartments);
                }

                @Override
                public void kmeansError(Throwable throwable) {
                    System.err.println("gw curse - err - " + throwable.getLocalizedMessage());
                    throwable.printStackTrace();
                    System.exit(98);
                }
            };
            kMeans.addKMeansListener(kMeansListener);
            kMeans.run();

            */
        }

        while (gwRunNotDone.get()) {
            System.out.println("So far size is " + finalSubcompartments.size());
            System.out.println("Wait another minute");
            try {
                TimeUnit.MINUTES.sleep(1);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        return finalSubcompartments;
    }

    private GenomeWideList<SubcompartmentInterval> extractAllInitialIntraSubcompartments(Dataset ds, ChromosomeHandler chromosomeHandler) {

        final GenomeWideList<SubcompartmentInterval> subcompartments = new GenomeWideList<>(chromosomeHandler);
        final AtomicInteger numRunsToExpect = new AtomicInteger();
        final AtomicInteger numRunsDone = new AtomicInteger();

        for (final Chromosome chromosome : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {

            // skip these matrices
            Matrix matrix = ds.getMatrix(chromosome, chromosome);
            if (matrix == null) continue;

            HiCZoom zoom = ds.getZoomForBPResolution(resolution);
            final MatrixZoomData zd = matrix.getZoomData(zoom);
            if (zd == null) continue;

            try {

                ExpectedValueFunction df = ds.getExpectedValues(zd.getZoom(), norm);
                if (df == null) {
                    System.err.println("O/E data not available at " + chromosome.getName() + " " + zoom + " " + norm);
                    System.exit(14);
                }

                int maxBin = chromosome.getLength() / resolution + 1;
                int maxSize = maxBin;

                RealMatrix localizedRegionData = ExtractingOEDataUtils.extractLocalThresholdedLogOEBoundedRegion(zd, 0, maxBin,
                        0, maxBin, maxSize, maxSize, norm, true, df, chromosome.getIndex(), logThreshold);

                final DataCleaner dataCleaner = new DataCleaner(localizedRegionData.getData(), coverageThreshold);

                if (dataCleaner.getLength() > 0) {

                    ConcurrentKMeans kMeans = new ConcurrentKMeans(dataCleaner.getCleanedData(), numClusters,
                            maxIters, 128971L); //Runtime.getRuntime().availableProcessors()/2
                    //BasicKMeans kMeans = new BasicKMeans(dataCleaner.getCleanedData(), numClusters, maxIters, 128971L);

                    numRunsToExpect.incrementAndGet();

                    KMeansListener kMeansListener = new KMeansListener() {
                        @Override
                        public void kmeansMessage(String s) {
                        }

                        @Override
                        public void kmeansComplete(Cluster[] clusters, long l) {
                            numRunsDone.incrementAndGet();
                            processKmeansResult(chromosome, dataCleaner, subcompartments, clusters);
                        }

                        @Override
                        public void kmeansError(Throwable throwable) {
                            System.err.println("curse chr " + chromosome.getName() + " - err - " + throwable.getLocalizedMessage());
                            throwable.printStackTrace();
                            System.exit(98);
                        }
                    };
                    kMeans.addKMeansListener(kMeansListener);
                    kMeans.run();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        while (numRunsDone.get() < numRunsToExpect.get()) {
            System.out.println("So far size is " + subcompartments.size());
            System.out.println("Wait another minute");
            try {
                TimeUnit.MINUTES.sleep(1);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        return subcompartments;
    }

    private void processKmeansResult(Chromosome chromosome, DataCleaner dataCleaner,
                                     GenomeWideList<SubcompartmentInterval> subcompartments, Cluster[] clusters) {

        List<SubcompartmentInterval> subcompartmentIntervals = new ArrayList<>();
        System.out.println("Chromosome " + chromosome.getName() + " clustered into " + clusters.length + " clusters");

        for (Cluster cluster : clusters) {
            int currentClusterID = uniqueClusterID.getAndIncrement();
            for (int i : cluster.getMemberIndexes()) {
                int x1 = dataCleaner.getOriginalIndexRow(i) * resolution;
                int x2 = x1 + resolution;

                subcompartmentIntervals.add(
                        new SubcompartmentInterval(chromosome.getIndex(), chromosome.getName(), x1, x2, currentClusterID));
            }
        }

        // resort
        reSort(subcompartments);

        subcompartments.addAll(subcompartmentIntervals);
    }

    public void reSort(GenomeWideList<SubcompartmentInterval> subcompartments) {
        subcompartments.filterLists(new FeatureFilter<SubcompartmentInterval>() {
            @Override
            public List<SubcompartmentInterval> filter(String chr, List<SubcompartmentInterval> featureList) {
                Collections.sort(featureList);
                return featureList;
            }
        });
    }

    private void writeClusterCenterToWig(Chromosome chromosome, double[] center, File file) {
        try {
            final FileWriter fw = new FileWriter(file);
            fw.write("fixedStep chrom=chr" + chromosome.getName() + " start=1" + " step=" + resolution + "\n");
            for (double d : center) {
                fw.write(d + "\n");
            }
            fw.close();

        } catch (Exception e) {
            System.err.println("Unable to make file for exporting center");
        }

    }
}
