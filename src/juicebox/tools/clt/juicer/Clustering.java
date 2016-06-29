/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2016 Broad Institute, Aiden Lab
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

import be.ac.ulg.montefiore.run.jahmm.Hmm;
import be.ac.ulg.montefiore.run.jahmm.ObservationVector;
import be.ac.ulg.montefiore.run.jahmm.OpdfMultiGaussianFactory;
import juicebox.HiCGlobals;
import juicebox.tools.clt.CommandLineParserForJuicer;
import juicebox.tools.clt.JuicerCLT;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.windowui.NormalizationType;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by muhammadsaadshamim on 9/14/15.
 */
public class Clustering extends JuicerCLT {

    double[][] data = new double[][]{{1, 2, 3, 2.1}, {1, 2, 3, 2.1}, {1, 1.8, 3, 2}};

    private boolean doDifferentialClustering = false;

    private int resolution = 1000000;
    private NormalizationType norm = NormalizationType.KR;
    private String inputFiles, outputPath;

    public Clustering() {
        super("clustering [-r resolution] [-k NONE/VC/VC_SQRT/KR] <input_HiC_file(s)> <output_file>");
        HiCGlobals.useCache = false;
    }

    @Override
    protected void readJuicerArguments(String[] args, CommandLineParserForJuicer juicerParser) {
        if (args.length != 3) {
            printUsageAndExit();
        }

        NormalizationType preferredNorm = juicerParser.getNormalizationTypeOption();
        if (preferredNorm != null)
            norm = preferredNorm;

        inputFiles = args[1];
        outputPath = args[2];

        List<String> possibleResolutions = juicerParser.getMultipleResolutionOptions();
        if (possibleResolutions != null) {
            if (possibleResolutions.size() > 1)
                System.err.println("Only one resolution can be specified for Clustering\nUsing " + possibleResolutions.get(0));
            resolution = Integer.parseInt(possibleResolutions.get(0));
        }
    }

    @Override
    public void run() {

    }


    /**
     * @param data to cluster - each
     * @param n
     */
    public void cluster(double[][] data, int n) {


        OpdfMultiGaussianFactory factory = new OpdfMultiGaussianFactory(6);
        new ObservationVector(data[0]);
        Hmm<ObservationVector> hmm = new Hmm<ObservationVector>(6, factory);

        List<ObservationVector> sequences = new ArrayList<ObservationVector>();

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

        */

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
        */


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

}
