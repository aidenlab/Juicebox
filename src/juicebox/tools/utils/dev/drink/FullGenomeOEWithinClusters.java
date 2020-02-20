/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab
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

package juicebox.tools.utils.dev.drink;

import juicebox.HiCGlobals;
import juicebox.data.ChromosomeHandler;
import juicebox.data.Dataset;
import juicebox.data.feature.GenomeWideList;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.tools.utils.dev.drink.kmeansfloat.Cluster;
import juicebox.tools.utils.dev.drink.kmeansfloat.ClusterTools;
import juicebox.windowui.NormalizationType;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class FullGenomeOEWithinClusters {
    private final Dataset ds;
    private final ChromosomeHandler chromosomeHandler;
    private final int resolution;
    private final NormalizationType norm;
    private final GenomeWideList<SubcompartmentInterval> origIntraSubcompartments;
    private final int numRounds = 15;
    private final int minIntervalSizeAllowed = 5;
    private final int numAttemptsForKMeans = 10;
    private final CompositeGenomeWideDensityMatrix interMatrix;
    private final float oeThreshold;

    public FullGenomeOEWithinClusters(Dataset ds, ChromosomeHandler chromosomeHandler, int resolution, NormalizationType norm,
                                      GenomeWideList<SubcompartmentInterval> origIntraSubcompartments, float oeThreshold, int derivativeStatus, boolean useNormalizationOfRows) {
        this.ds = ds;
        this.chromosomeHandler = chromosomeHandler;
        this.resolution = resolution;
        this.norm = norm;
        this.oeThreshold = oeThreshold;
        DrinkUtils.collapseGWList(origIntraSubcompartments);
        this.origIntraSubcompartments = origIntraSubcompartments;

        interMatrix = new CompositeGenomeWideDensityMatrix(
                chromosomeHandler, ds, norm, resolution, origIntraSubcompartments, oeThreshold, derivativeStatus, useNormalizationOfRows, minIntervalSizeAllowed);
        System.gc();
    }

    public Map<Integer, GenomeWideList<SubcompartmentInterval>> extractFinalGWSubcompartments(File outputDirectory, Random generator) {

        Map<Integer, GenomeWideList<SubcompartmentInterval>> numItersToResults = new HashMap<>();

        if (HiCGlobals.printVerboseComments) {
            interMatrix.exportData(outputDirectory);
        }

        GenomeWideKmeansRunner kmeansRunner = new GenomeWideKmeansRunner(chromosomeHandler, interMatrix, generator);

        double[][] iterToMSE = new double[2][numRounds];
        Arrays.fill(iterToMSE[1], Double.MAX_VALUE);

        for (int z = 0; z < numRounds; z++) {

            int k = z + 2;
            Cluster[] bestClusters = null;
            int[] bestIDs = null;

            for (int p = 0; p < numAttemptsForKMeans; p++) {


                kmeansRunner.prepareForNewRun(k);
                kmeansRunner.launchKmeansGWMatrix();

                int numActualClustersThisAttempt = kmeansRunner.getNumActualClusters();
                double mseThisAttempt = kmeansRunner.getMeanSquaredError();

                if (mseThisAttempt < iterToMSE[1][z]) {
                    iterToMSE[0][z] = numActualClustersThisAttempt;
                    iterToMSE[1][z] = mseThisAttempt;
                    numItersToResults.put(k, kmeansRunner.getFinalCompartments());
                    bestClusters = kmeansRunner.getRecentClustersClone();
                    bestIDs = kmeansRunner.getRecentIDsClone();
                }
            }

            ClusterTools.performStatisticalAnalysisBetweenClusters(outputDirectory, "final_gw_" + k, bestClusters, bestIDs);
        }

        if (minIntervalSizeAllowed > 0) {
            LeftOverClusterIdentifier.identify(chromosomeHandler, ds, norm, resolution, numItersToResults, origIntraSubcompartments, minIntervalSizeAllowed, oeThreshold);
        }

        MatrixTools.saveMatrixTextNumpy(new File(outputDirectory, "clusterSizeToMeanSquaredError.npy").getAbsolutePath(), iterToMSE);

        return numItersToResults;
    }
}
