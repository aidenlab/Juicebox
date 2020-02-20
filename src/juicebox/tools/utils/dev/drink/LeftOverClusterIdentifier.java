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

import juicebox.data.ChromosomeHandler;
import juicebox.data.Dataset;
import juicebox.data.HiCFileTools;
import juicebox.data.MatrixZoomData;
import juicebox.data.feature.GenomeWideList;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.tools.utils.dev.drink.kmeansfloat.ClusterTools;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.RealMatrix;
import org.broad.igv.feature.Chromosome;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class LeftOverClusterIdentifier {
    public static void identify(ChromosomeHandler chromosomeHandler, Dataset ds, NormalizationType norm, int resolution,
                                Map<Integer, GenomeWideList<SubcompartmentInterval>> results, GenomeWideList<SubcompartmentInterval> preSubcompartments, int minIntervalSizeAllowed, float threshold) {

        for (Chromosome chr1 : chromosomeHandler.getAutosomalChromosomesArray()) {
            final MatrixZoomData zd = HiCFileTools.getMatrixZoomData(ds, chr1, chr1, resolution);
            if (zd == null) continue;

            float[][] allDataForRegion = null;
            try {
                RealMatrix localizedRegionData = HiCFileTools.getRealOEMatrixForChromosome(ds, zd, chr1, resolution,
                        norm, threshold, ExtractingOEDataUtils.ThresholdType.LINEAR_INVERSE_OE_BOUNDED_SCALED_BTWN_ZERO_ONE, true);
                allDataForRegion = MatrixTools.convertToFloatMatrix(localizedRegionData.getData());
            } catch (Exception e) {
                e.printStackTrace();
                System.exit(99);
            }

            if (allDataForRegion == null) {
                System.err.println("Missing Data " + zd.getKey());
                return;
            }

            for (int i = 0; i < allDataForRegion.length; i++) {
                for (int j = 0; j < allDataForRegion[0].length; j++) {
                    if (Float.isNaN(allDataForRegion[i][j]) || Float.isInfinite(allDataForRegion[i][j]) || Math.abs(allDataForRegion[i][j]) < 1E-30) {
                        allDataForRegion[i][j] = 0;
                    }
                }
            }

            List<SubcompartmentInterval> preIntervals = preSubcompartments.getFeatures("" + chr1.getIndex());
            List<Integer> indicesMissing = new ArrayList<>();

            for (SubcompartmentInterval preInterv : preIntervals) {
                if (preInterv.getWidthForResolution(resolution) < minIntervalSizeAllowed) {
                    int binXStart = preInterv.getX1() / resolution;
                    int binXEnd = preInterv.getX2() / resolution;

                    for (int j = binXStart; j < binXEnd; j++) {
                        indicesMissing.add(j);
                    }
                }
            }

            for (Integer key : results.keySet()) {
                GenomeWideList<SubcompartmentInterval> listForKey = results.get(key);
                Map<Integer, float[]> cIDToCenter = getClusterCenters(allDataForRegion, listForKey.getFeatures("" + chr1.getIndex()), resolution);

                List<SubcompartmentInterval> newlyAssignedSubcompartments = getNewlyAssignedCompartments(chr1, cIDToCenter, indicesMissing, allDataForRegion, resolution);

                listForKey.addAll(newlyAssignedSubcompartments);

                results.put(key, listForKey);
            }
        }
    }

    private static Map<Integer, float[]> getClusterCenters(float[][] allDataForRegion, List<SubcompartmentInterval> intervals, int resolution) {

        Map<Integer, float[]> cIDToCenter = new HashMap<>();
        Map<Integer, Integer> cIDToSize = new HashMap<>();


        for (SubcompartmentInterval interval : intervals) {
            int binXStart = interval.getX1() / resolution;
            int binXEnd = interval.getX2() / resolution;
            int cID = interval.getClusterID();
            float[] total = new float[allDataForRegion[0].length];

            for (int i = binXStart; i < binXEnd; i++) {
                for (int j = 0; j < allDataForRegion[i].length; j++) {
                    total[j] += allDataForRegion[i][j];
                }
            }

            if (cIDToSize.containsKey(cID)) {
                cIDToSize.put(cID, cIDToSize.get(cID) + binXEnd - binXStart);
                float[] vec = cIDToCenter.get(cID);
                for (int j = 0; j < vec.length; j++) {
                    total[j] += vec[j];
                }
            } else {
                cIDToSize.put(cID, binXEnd - binXStart);
            }
            cIDToCenter.put(cID, total);
        }

        for (Integer key : cIDToCenter.keySet()) {
            cIDToCenter.put(key, ClusterTools.normalize(cIDToCenter.get(key), cIDToSize.get(key)));
        }


        return cIDToCenter;
    }


    private static List<SubcompartmentInterval> getNewlyAssignedCompartments(Chromosome chromosome, Map<Integer, float[]> cIDToCenter, List<Integer> indicesMissing, float[][] allDataForRegion, int resolution) {

        List<SubcompartmentInterval> intervals = new ArrayList<>();

        for (int indx : indicesMissing) {

            int metaID = getClosestClusterID(allDataForRegion[indx], cIDToCenter);

            int x1 = indx * resolution;
            int x2 = x1 + resolution;

            intervals.add(new SubcompartmentInterval(chromosome.getIndex(), chromosome.getName(), x1, x2, metaID));
        }
        return intervals;
    }

    private static int getClosestClusterID(float[] vector, Map<Integer, float[]> cIDToCenter) {
        int currID = Integer.MAX_VALUE;
        double overallDistance = Double.MAX_VALUE;
        boolean nothingChanged = true;

        for (Integer key : cIDToCenter.keySet()) {
            double newDistance = ClusterTools.getDistance(cIDToCenter.get(key), vector);
            if (newDistance < overallDistance) {
                overallDistance = newDistance;
                currID = key;
                nothingChanged = false;
            }
        }
        if (nothingChanged) {
            System.err.println(" - WTF " + overallDistance + " - " + cIDToCenter.keySet());
        }
        return currID;
    }


}
