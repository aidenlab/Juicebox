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
import juicebox.tools.utils.juicer.curse.DataCleaner;
import juicebox.tools.utils.juicer.curse.SubcompartmentInterval;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import kmeans.BasicKMeans;
import kmeans.Cluster;
import kmeans.KMeansListener;
import org.apache.commons.math.linear.RealMatrix;
import org.broad.igv.feature.Chromosome;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
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
    private File outputFile;
    private AtomicInteger uniqueClusterID = new AtomicInteger(1);

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
        outputFile = new File(args[2]);

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
                int maxSize = maxBin + 1;

                RealMatrix localizedRegionData = HiCFileTools.extractLocalLogOEBoundedRegion(zd, 0, maxBin,
                        0, maxBin, maxSize, maxSize, norm, true, df, chromosome.getIndex());

                final DataCleaner dataCleaner = new DataCleaner(localizedRegionData.getData(), 0.3);

                //ConcurrentKMeans kMeans = new ConcurrentKMeans(dataCleaner.getCleanedData(), 20, 20000, 128971L);
                BasicKMeans kMeans = new BasicKMeans(dataCleaner.getCleanedData(), 20, 20000, 128971L);

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

        subcompartments.simpleExport(outputFile);
    }

    private void processKmeansResult(Chromosome chromosome, DataCleaner dataCleaner,
                                     GenomeWideList<SubcompartmentInterval> subcompartments, Cluster[] clusters) {

        List<SubcompartmentInterval> subcompartmentIntervals = new ArrayList<>();
        //if(HiCGlobals.printVerboseComments) { }
        System.out.println("Chromosome " + chromosome.getName() + " clustered into " + clusters.length + " clusters");

        for (Cluster cluster : clusters) {
            int currentClusterID = uniqueClusterID.getAndIncrement();
            for (int i : cluster.getMemberIndexes()) {
                int x1 = dataCleaner.getOriginalIndexRow(i) * resolution;
                int x2 = x1 + resolution;

                subcompartmentIntervals.add(
                        new SubcompartmentInterval(chromosome.getIndex(), chromosome.getName(),
                                x1, x2, currentClusterID));
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


    /*
     * @param data to cluster - each
     * @param n

    public void cluster(double[][] data, int n) {


        OpdfMultiGaussianFactory factory = new OpdfMultiGaussianFactory(6);
        new ObservationVector(data[0]);
        Hmm<ObservationVector> hmm = new Hmm<>(6, factory);

        List<ObservationVector> sequences = new ArrayList<>();

        /* todo
        for (double[] row : data)

            sequences.add(new ArrayList(new ObservationVector(row)));
        //sequences.add(mg.observationSequence(100));



        BaumWelchLearner bwl = new BaumWelchLearner();
        Hmm<?> learntHmm = bwl.learn(hmm, sequences);

        for (int i = 0; i < 10; i++) {
            bwl.iterate(learntHmm);
        }


        List<List<ObservationVector>> sequences2 = new ArrayList<List<ObservationVector>>();

        KMeansLearner<ObservationVector> kml =
                new KMeansLearner <ObservationVector>(3 , new OpdfMultiGaussianFactory(6) , sequences2);
        Hmm <ObservationVector> initHmm = kml.iterate() ;



    }


    private int[][] normalizeMatrix(int[][] matrix, int v0, int v1, int z0, int z1) {


        if (v1 == 1) {
            matrix = MatrixTools.normalizeMatrixUsingRowSum(matrix);
        }
        if (v0 == 1) {
            matrix = MatrixTools.normalizeMatrixUsingColumnSum(matrix);
        }
        if (z1 == 1) {
            //matrix=stats.zscore(matrix,axis=1);
        }
        if (z0 == 1) {
            //matrix=stats.zscore(matrix,axis=0);
        }
        return matrix;
    }

    // TODO
    private void runKmeansClustering(int[][] matrix, int v0, int v1, int z0, int z1) {
        matrix = normalizeMatrix(matrix, v0, v1, z0, z1);
        //centroids = kmeans(x,k)
        //idx = vq(x,centroids)
        //export(idx+1)

        /*
        KMeansLearner<ObservationVector> kml = new KMeansLearner<ObservationVector>(2,new OpdfMultiGaussianFactory<?>(21), sequences);
        Hmm<ObservationVector> fittedHmm = kml.iterate();//kml.learn();

        BaumWelchLearner<?> bwl = new BaumWelchLearner<?>();
        Hmm<ObservationVector> learntHmm = bwl.learn(fittedHmm, sequences);
        System.out.println(learntHmm.toString());



    }

    // TODO
    private void runHMMClustering(int[][] matrix, int v0, int v1, int z0, int z1) {
        matrix = normalizeMatrix(matrix, v0, v1, z0, z1);
        //from sklearn.hmm import GaussianHMM
        //model = GaussianHMM(n_components=k, covariance_type="diag",n_iter=1000)
        //model.fit([x])
        //idx = model.predict(x)
        //export(idx+1)
    }
    */

}
