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

package juicebox.tools.utils.juicer.drink;

import juicebox.data.ChromosomeHandler;
import juicebox.data.Dataset;
import juicebox.data.feature.GenomeWideList;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.tools.utils.juicer.drink.kmeans.Cluster;
import juicebox.tools.utils.juicer.drink.kmeans.ConcurrentKMeans;
import juicebox.tools.utils.juicer.drink.kmeans.KMeansListener;
import juicebox.windowui.NormalizationType;

import java.io.File;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public class OriginalGWApproach {

    public static GenomeWideList<SubcompartmentInterval>
    extractFinalGWSubcompartments(Dataset ds, ChromosomeHandler chromosomeHandler, int resolution, NormalizationType norm,
                                  File outputDirectory, int numClusters, int maxIters, double logThreshold,
                                  GenomeWideList<SubcompartmentInterval> intraSubcompartments) {
        final ScaledGenomeWideMatrix gwMatrix = new ScaledGenomeWideMatrix(chromosomeHandler, ds, norm,
                resolution, intraSubcompartments, logThreshold);


        final GenomeWideList<SubcompartmentInterval> finalSubcompartments = new GenomeWideList<>(chromosomeHandler);


        final AtomicBoolean gwRunNotDone = new AtomicBoolean(true);
        if (gwMatrix.getLength() > 0) {

            //System.out.println("printing GW matrix file");
            File outputFile = new File(outputDirectory, "gw_matrix_data.txt");
            MatrixTools.exportData(gwMatrix.getCleanedData(), outputFile);

            ConcurrentKMeans kMeans = new ConcurrentKMeans(gwMatrix.getCleanedData(), numClusters,
                    maxIters, 128971L);

            KMeansListener kMeansListener = new KMeansListener() {
                @Override
                public void kmeansMessage(String s) {
                }

                @Override
                public void kmeansComplete(Cluster[] clusters, long l) {
                    gwRunNotDone.set(false);
                    gwMatrix.processGWKmeansResult(clusters, finalSubcompartments);
                }

                @Override
                public void kmeansError(Throwable throwable) {
                    throwable.printStackTrace();
                    System.err.println("gw drink - err - " + throwable.getLocalizedMessage());
                    System.exit(98);
                }
            };
            kMeans.addKMeansListener(kMeansListener);
            kMeans.run();
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
}
