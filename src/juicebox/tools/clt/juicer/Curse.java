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
import juicebox.data.feature.GenomeWideList;
import juicebox.tools.clt.CommandLineParserForJuicer;
import juicebox.tools.clt.JuicerCLT;
import juicebox.tools.utils.juicer.curse.*;
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
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by muhammadsaadshamim on 9/14/15.
 */
public class Curse extends JuicerCLT {

    private boolean doDifferentialClustering = false;
    private int resolution = 100000;
    private Dataset ds;
    private File outputDirectory;
    private int numClusters = 20;
    private double maxPercentAllowedToBeZeroThreshold = 0.3;
    private int maxIters = 10000;
    private double logThreshold = 2;
    private int connectedComponentThreshold = 50;
    private int whichApproachtoUse = 0;

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

        if (whichApproachtoUse == 0) {

            GenomeWideList<SubcompartmentInterval> intraSubcompartments = extractAllInitialIntraSubcompartments(ds, chromosomeHandler);

            SubcompartmentInterval.collapseGWList(intraSubcompartments);

            File outputFile2 = new File(outputDirectory, "result_intra_initial_collapsed.bed");
            intraSubcompartments.simpleExport(outputFile2);


        } else if (whichApproachtoUse == 1) {

            GenomeWideList<SubcompartmentInterval> intraSubcompartments = extractAllInitialIntraSubcompartments(ds, chromosomeHandler);

            File outputFile = new File(outputDirectory, "result_intra_initial.bed");
            intraSubcompartments.simpleExport(outputFile);

            SubcompartmentInterval.collapseGWList(intraSubcompartments);

            File outputFile2 = new File(outputDirectory, "result_intra_initial_collapsed.bed");
            intraSubcompartments.simpleExport(outputFile2);

            GenomeWideList<SubcompartmentInterval> finalSubcompartments = OriginalGWApproach.extractFinalGWSubcompartments(
                    ds, chromosomeHandler, resolution, norm, outputDirectory, numClusters, maxIters, logThreshold,
                    intraSubcompartments);
            File outputFile3 = new File(outputDirectory, "gw_result_initial.bed");
            finalSubcompartments.simpleExport(outputFile3);

            SubcompartmentInterval.collapseGWList(finalSubcompartments);

            File outputFile4 = new File(outputDirectory, "gw_result_collapsed.bed");
            finalSubcompartments.simpleExport(outputFile4);
        } else if (whichApproachtoUse == 2) {
            GenomeWideList<SubcompartmentInterval> intraSubcompartments = extractAllInitialIntraSubcompartments(ds, chromosomeHandler);

            File outputFile = new File(outputDirectory, "result_intra_initial.bed");
            intraSubcompartments.simpleExport(outputFile);

            SubcompartmentInterval.collapseGWList(intraSubcompartments);

            File outputFile2 = new File(outputDirectory, "result_intra_initial_collapsed.bed");
            intraSubcompartments.simpleExport(outputFile2);


            GenomeWideList<SubcompartmentInterval> finalSubcompartments = SecondGWApproach.extractFinalGWSubcompartments(
                    ds, chromosomeHandler, resolution, norm, outputDirectory, numClusters, maxIters, logThreshold,
                    intraSubcompartments, connectedComponentThreshold);

            outputFile2 = new File(outputDirectory, "final_stitched_collapsed_subcompartments.bed");
            finalSubcompartments.simpleExport(outputFile2);

        } else if (whichApproachtoUse == 3) {
            GenomeWideList<SubcompartmentInterval> intraSubcompartments = extractAllInitialIntraSubcompartments(ds, chromosomeHandler);

            File outputFile = new File(outputDirectory, "result_intra_initial.bed");
            intraSubcompartments.simpleExport(outputFile);

            SubcompartmentInterval.collapseGWList(intraSubcompartments);

            File outputFile2 = new File(outputDirectory, "result_intra_initial_collapsed.bed");
            intraSubcompartments.simpleExport(outputFile2);


            GenomeWideList<SubcompartmentInterval> finalSubcompartments = ThirdGWApproach.extractFinalGWSubcompartments(
                    ds, chromosomeHandler, resolution, norm, outputDirectory, numClusters, maxIters, logThreshold,
                    intraSubcompartments, connectedComponentThreshold);

            outputFile2 = new File(outputDirectory, "final_stitched_collapsed_subcompartments.bed");
            finalSubcompartments.simpleExport(outputFile2);

        }
    }

    private GenomeWideList<SubcompartmentInterval> extractAllInitialIntraSubcompartments(Dataset ds, ChromosomeHandler chromosomeHandler) {

        final GenomeWideList<SubcompartmentInterval> subcompartments = new GenomeWideList<>(chromosomeHandler);
        final AtomicInteger numRunsToExpect = new AtomicInteger();
        final AtomicInteger numRunsDone = new AtomicInteger();

        for (final Chromosome chromosome : chromosomeHandler.getAutosomalChromosomesArray()) {

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

                final DataCleaner dataCleaner = new DataCleaner(localizedRegionData.getData(), maxPercentAllowedToBeZeroThreshold, resolution);

                if (dataCleaner.getLength() > 0) {

                    ConcurrentKMeans kMeans = new ConcurrentKMeans(dataCleaner.getCleanedData(), numClusters,
                            maxIters, 128971L);

                    numRunsToExpect.incrementAndGet();
                    KMeansListener kMeansListener = new KMeansListener() {
                        @Override
                        public void kmeansMessage(String s) {
                        }

                        @Override
                        public void kmeansComplete(Cluster[] clusters, long l) {
                            numRunsDone.incrementAndGet();
                            dataCleaner.processKmeansResult(chromosome, subcompartments, clusters);
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
