/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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
import juicebox.data.basics.Chromosome;
import juicebox.tools.utils.juicer.hiccups.HiCCUPSConfiguration;
import juicebox.tools.utils.juicer.hiccups.HiCCUPSRegionContainer;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationHandler;
import juicebox.windowui.NormalizationType;
import org.broad.igv.util.Pair;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class HiCCUPSRegionHandler {

    final List<HiCCUPSRegionContainer> allRegionContainers = new ArrayList<>();
    final Map<Pair<Integer, HiCZoom>, MatrixZoomData> zoomDataMap = new HashMap<>();
    final Map<Pair<Integer, HiCZoom>, double[]> normVectorMap = new HashMap<>();
    final Map<Pair<Integer, HiCZoom>, double[]> expectedVectorMap = new HashMap<>();

    public HiCCUPSRegionHandler(Dataset ds, ChromosomeHandler chromosomeHandler, HiCZoom zoom, NormalizationType norm,
                                HiCCUPSConfiguration conf, int regionWidth, int regionMargin, boolean restrictSearchRegions) {

        for (final Chromosome chromosome : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {

            // skip these matrices
            Matrix matrix = ds.getMatrix(chromosome, chromosome);
            if (matrix == null) continue;

            // get matrix data access
            long start_time = System.currentTimeMillis();
            final MatrixZoomData zd = matrix.getZoomData(zoom);

            Pair<Integer, HiCZoom> pairKey = new Pair<>(chromosome.getIndex(), zoom);
            zoomDataMap.put(pairKey, zd);

            //NormalizationType preferredNormalization = HiCFileTools.determinePreferredNormalization(ds);
            NormalizationVector normVector = ds.getNormalizationVector(chromosome.getIndex(), zoom, norm);
            if (normVector != null || norm.equals(NormalizationHandler.NONE)) {
                if (!norm.equals(NormalizationHandler.NONE)) {
                    final double[] normalizationVector = normVector.getData().getValues().get(0);
                    normVectorMap.put(pairKey, normalizationVector);
                }
    
                final double[] expectedVector = HiCFileTools.extractChromosomeExpectedVector(ds, chromosome.getIndex(),
                        zoom, norm).getValues().get(0);
                expectedVectorMap.put(pairKey, expectedVector);

                // need overall bounds for the chromosome
                int chrMatrixWidth = (int) Math.ceil((double) chromosome.getLength() / conf.getResolution());
                if (norm.equals(NormalizationHandler.NONE)) {
                    final double[] normalizationVector = new double[chrMatrixWidth];
                    for (int i=0;i<normalizationVector.length;i++) {
                        normalizationVector[i]=1;
                    }
                    normVectorMap.put(pairKey, normalizationVector);
                }
                double chrWidthInTermsOfMatrixDimension = Math.ceil(chrMatrixWidth * 1.0 / regionWidth) + 1;
                long load_time = System.currentTimeMillis();
                if (HiCGlobals.printVerboseComments) {
                    System.out.println("Time to load chr " + chromosome.getName() + " matrix: " + (load_time - start_time) + "ms");
                }

                for (int i = 0; i < chrWidthInTermsOfMatrixDimension; i++) {
                    final int[] rowBounds = calculateRegionBounds(i, regionWidth, chrMatrixWidth, regionMargin);

                    if (rowBounds[4] < chrMatrixWidth - regionMargin) {
                        for (int j = i; j < chrWidthInTermsOfMatrixDimension; j++) {
                            if (restrictSearchRegions && (j - i) * regionWidth * conf.getResolution() > 8000000) {
                                continue;
                            }

                            final int[] columnBounds = calculateRegionBounds(j, regionWidth, chrMatrixWidth, regionMargin);
                            if (HiCGlobals.printVerboseComments) {
                                System.out.print(".");
                            }

                            if (columnBounds[4] < chrMatrixWidth - regionMargin) {

                                allRegionContainers.add(new HiCCUPSRegionContainer(chromosome,
                                        rowBounds, columnBounds));
                            }
                        }
                    }
                }
            } else {
                System.err.println("Data not available for " + chromosome + " at " + conf.getResolution() + " resolution");
            }
        }
    }

    private int[] calculateRegionBounds(int index, int regionWidth, int chrMatrixWidth, int regionMargin) {

        int bound1R = Math.min(regionMargin + (index * regionWidth), chrMatrixWidth - regionMargin);
        int bound1 = bound1R - regionMargin;
        int bound2R = Math.min(bound1R + regionWidth, chrMatrixWidth - regionMargin);
        int bound2 = bound2R + regionMargin;

        int diff1 = bound1R - bound1;
        int diff2 = bound2 - bound2R;

        return new int[]{bound1, bound2, diff1, diff2, bound1R, bound2R};
    }

    public int getSize() {
        return allRegionContainers.size();
    }

    public synchronized HiCCUPSRegionContainer getRegionFromIndex(int indexOfRegionForThread) {
        return allRegionContainers.get(indexOfRegionForThread);
    }

    public MatrixZoomData getZoomData(HiCCUPSRegionContainer regionContainer, HiCZoom zoom) {
        return zoomDataMap.get(new Pair<>(regionContainer.getChromosome().getIndex(), zoom));
    }

    public double[] getNormalizationVector(HiCCUPSRegionContainer regionContainer, HiCZoom zoom) {
        return normVectorMap.get(new Pair<>(regionContainer.getChromosome().getIndex(), zoom));
    }

    public double[] getExpectedVector(HiCCUPSRegionContainer regionContainer, HiCZoom zoom) {
        return expectedVectorMap.get(new Pair<>(regionContainer.getChromosome().getIndex(), zoom));
    }
}
