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

package juicebox.tools.utils.juicer.drink;

import juicebox.data.*;
import juicebox.data.feature.GenomeWideList;
import juicebox.tools.utils.juicer.drink.kmeans.Cluster;
import juicebox.tools.utils.juicer.drink.kmeans.ConcurrentKMeans;
import juicebox.tools.utils.juicer.drink.kmeans.KMeansListener;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.RealMatrix;
import org.broad.igv.feature.Chromosome;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class Clustering {

    public static GenomeWideList<SubcompartmentInterval> extractAllInitialIntraSubcompartments(
            Dataset ds, ChromosomeHandler chromosomeHandler, int resolution, NormalizationType norm, double logThreshold,
            double maxPercentAllowedToBeZeroThreshold, int numClusters, int maxIters) {

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
                            System.err.println("drink chr " + chromosome.getName() + " - err - " + throwable.getLocalizedMessage());
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


    /**
     * version with multiple hic files
     */

    public static void extractAllComparativeIntraSubcompartments(
            List<Dataset> datasets, ChromosomeHandler chromosomeHandler, int resolution, NormalizationType norm, double logThreshold,
            double maxPercentAllowedToBeZeroThreshold, int numClusters, int maxIters, File outputDirectory) {

        final List<GenomeWideList<SubcompartmentInterval>> comparativeSubcompartments = new ArrayList<>();
        //GenomeWideList<>(chromosomeHandler);
        final AtomicInteger numRunsToExpect = new AtomicInteger();
        final AtomicInteger numRunsDone = new AtomicInteger();

        final Map<Integer, double[]> idToCentroidMap = new HashMap<>();

        for (Dataset ds : datasets) {
            // each ds will need a respective list of assigned subcompartments
            comparativeSubcompartments.add(new GenomeWideList<SubcompartmentInterval>(chromosomeHandler));
        }

        for (final Chromosome chromosome : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {

            try {

                List<double[][]> matrices = new ArrayList<>();

                for (Dataset ds : datasets) {
                    // skip these matrices
                    Matrix matrix = ds.getMatrix(chromosome, chromosome);
                    if (matrix == null) continue;

                    HiCZoom zoom = ds.getZoomForBPResolution(resolution);
                    final MatrixZoomData zd = matrix.getZoomData(zoom);
                    if (zd == null) continue;

                    ExpectedValueFunction df = ds.getExpectedValues(zd.getZoom(), norm);
                    if (df == null) {
                        System.err.println("O/E data not available at " + chromosome.getName() + " " + zoom + " " + norm);
                        System.exit(14);
                    }

                    int maxBin = chromosome.getLength() / resolution + 1;
                    int maxSize = maxBin;

                    RealMatrix localizedRegionData = ExtractingOEDataUtils.extractLocalThresholdedLogOEBoundedRegion(zd, 0, maxBin,
                            0, maxBin, maxSize, maxSize, norm, true, df, chromosome.getIndex(), logThreshold);

                    matrices.add(localizedRegionData.getData());
                }

                // e.g. diploid Y chromosome; can't assess vs non existent map
                if (matrices.size() != datasets.size()) continue;


                final DataCleanerV2 dataCleanerV2 = new DataCleanerV2(matrices, chromosome.getIndex(),
                        maxPercentAllowedToBeZeroThreshold, resolution, outputDirectory, new ArrayList<Integer>());

                if (dataCleanerV2.getLength() > 0) {

                    /*
                    File outputFile2 = new File(outputDirectory, "allFilesChr"+chromosome.getIndex()+".txt");
                    MatrixTools.exportData(dataCleanerV2.getCleanedData(), outputFile2);
                    */


                    ConcurrentKMeans kMeans = new ConcurrentKMeans(dataCleanerV2.getCleanedData(), numClusters,
                            maxIters, 128971L);

                    numRunsToExpect.incrementAndGet();
                    KMeansListener kMeansListener = new KMeansListener() {
                        @Override
                        public void kmeansMessage(String s) {
                        }

                        @Override
                        public void kmeansComplete(Cluster[] clusters, long l) {
                            numRunsDone.incrementAndGet();
                            dataCleanerV2.processKmeansResultV2(chromosome, comparativeSubcompartments, clusters,
                                    idToCentroidMap);
                        }

                        @Override
                        public void kmeansError(Throwable throwable) {
                            System.err.println("drink chr " + chromosome.getName() + " - err - " + throwable.getLocalizedMessage());
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
            System.out.println("Wait another minute");
            try {
                TimeUnit.MINUTES.sleep(1);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        for (GenomeWideList<SubcompartmentInterval> gwList : comparativeSubcompartments) {
            SubcompartmentInterval.reSort(gwList);
        }

        // process differences for diff vector
        //SubcompartmentInterval.extractDiffVectors(comparativeSubcompartments, idToCentroidMap, outputDirectory);

        GenomeWideList<SubcompartmentInterval> consensus = SubcompartmentInterval.calculateConsensus(comparativeSubcompartments);

        for (GenomeWideList<SubcompartmentInterval> gwList : comparativeSubcompartments) {
            SubcompartmentInterval.collapseGWList(gwList);
        }

        for (int i = 0; i < datasets.size(); i++) {
            File outputFile2 = new File(outputDirectory, "result_intra_compare_file" + i + ".bed");
            comparativeSubcompartments.get(i).simpleExport(outputFile2);
        }

        File outputFile3 = new File(outputDirectory, "consensus_result_intra_compare_file.bed");
        consensus.simpleExport(outputFile3);
    }

    // todo mss
    // variableStep chrom=chr2 span=5
    // 300701  12.5
    private void writeClusterCenterToWig(Chromosome chromosome, double[] center, File file, int resolution) {
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
