/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2021 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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
 *  FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.tools.utils.original;

import juicebox.HiC;
import juicebox.data.ChromosomeHandler;
import juicebox.data.ContactRecord;
import juicebox.data.basics.Chromosome;
import juicebox.windowui.HiCZoom;

import java.io.File;
import java.io.IOException;
import java.util.Map;

public class TensorPP {

    private final int MAX_SQRT = (int) Math.sqrt(Integer.MAX_VALUE);
    private final int chr1Idx;
    private final int chr2Idx;
    private final int chr3Idx;
    private final TensorZoomDataPP[] zoomData;

    /**
     * Constructor for creating a matrix and initializing zoomed data at predefined resolution scales.  This
     * constructor is used when parsing alignment files.
     * c
     *
     * @param chr1Idx             Chromosome 1
     * @param chromosomeHandler
     * @param bpBinSizes
     * @param chr2Idx             Chromosome 2
     */
    public TensorPP(int chr1Idx, int chr2Idx, int chr3Idx, ChromosomeHandler chromosomeHandler, int[] bpBinSizes,
                    int countThreshold, int BLOCK_CAPACITY) {
        this.chr1Idx = chr1Idx;
        this.chr2Idx = chr2Idx;
        this.chr3Idx = chr3Idx;

        int nResolutions = bpBinSizes.length;

        zoomData = new TensorZoomDataPP[nResolutions];

        int zoom = 0; //
        for (int idx = 0; idx < bpBinSizes.length; idx++) {
			int binSize = bpBinSizes[zoom];
			Chromosome chrom1 = chromosomeHandler.getChromosomeFromIndex(chr1Idx);
			Chromosome chrom2 = chromosomeHandler.getChromosomeFromIndex(chr2Idx);
            Chromosome chrom3 = chromosomeHandler.getChromosomeFromIndex(chr3Idx);

			// Size block (subtensors) to be ~500 bins wide.

            long len = Math.max(chrom1.getLength(), chrom2.getLength());
            len = Math.max(len, chrom3.getLength());
            // for now, this will not be a long
            int nBins = (int) (len / binSize + 1);   // Size of chrom in bins
            int nColumns;
            nColumns = getNumColumnsFromNumBins(nBins, binSize, 0);
            // TODO: need to figure out the proper ways to divide bins! Use same xyz blockColumnCount for now;
//            if (chrom1.equals(chrom2)) {
//                nColumns = getNumColumnsFromNumBins(nBins, binSize, INTRA_CUTOFF);
//            } else {
//                nColumns = getNumColumnsFromNumBins(nBins, binSize, INTER_CUTOFF);
//            }
            zoomData[idx] = new TensorZoomDataPP(chrom1, chrom2, chrom3, binSize, nColumns, nColumns,
                    nColumns, zoom, countThreshold, BLOCK_CAPACITY);
            zoom++;

        }
    }

    /*TODO: questions about dividing block size in terms of bins! performance design*/
    private int getNumColumnsFromNumBins(int nBins, int binSize, int cutoff) {
        int nColumns = nBins / Preprocessor.BLOCK_SIZE + 1;
        if (binSize < cutoff) {
            long numerator = (long) nBins * binSize;
            long denominator = (long) Preprocessor.BLOCK_SIZE * cutoff;
            nColumns = (int) (numerator / denominator) + 1;
        }
        return Math.min(nColumns, MAX_SQRT - 1);
    }

    /**
     * Constructor for creating a matrix with a single zoom level at a specified bin size.  This is provided
     * primarily for constructing a whole-genome view.
     *
     * @param chr1Idx Chromosome 1
     * @param chr2Idx Chromosome 2
     * @param binSize Bin size
     */
    TensorPP(int chr1Idx, int chr2Idx, int chr3Idx, int binSize, int blockColumnCount, ChromosomeHandler chromosomeHandler,
             int countThreshold) {
        this.chr1Idx = chr1Idx;
        this.chr2Idx = chr2Idx;
        this.chr3Idx = chr3Idx;
        zoomData = new TensorZoomDataPP[1];
        zoomData[0] = new TensorZoomDataPP(chromosomeHandler.getChromosomeFromIndex(chr1Idx), chromosomeHandler.getChromosomeFromIndex(chr2Idx),
                chromosomeHandler.getChromosomeFromIndex(chr3Idx), binSize, blockColumnCount, blockColumnCount, blockColumnCount,
                0, countThreshold);
    }


    String getKey() {
        return "" + chr1Idx + "_" + chr2Idx;
    }


    void incrementCount(int pos1, int pos2, int pos3, float score, Map<String, ExpectedValueCalculation> expectedValueCalculations, File tmpDir) throws IOException {
        for (TensorZoomDataPP aZoomData : zoomData) {
            aZoomData.incrementCount(pos1, pos2, pos3, score, expectedValueCalculations, tmpDir);
        }
    }

//    public void incrementCount(ContactRecord cr, Map<String, ExpectedValueCalculation> expectedValueCalculations, File tmpDir, HiCZoom zoom) throws IOException {
//        for (MatrixZoomDataPP aZoomData : zoomData) {
//            if (aZoomData.isFrag && zoom.getUnit().equals(HiC.Unit.FRAG)) {
//                aZoomData.incrementCount(cr, expectedValueCalculations, tmpDir, zoom);
//            } else if (!aZoomData.isFrag && zoom.getUnit().equals(HiC.Unit.BP)) {
//                aZoomData.incrementCount(cr, expectedValueCalculations, tmpDir, zoom);
//            }
//        }
//    }

    public void parsingComplete() {
        for (TensorZoomDataPP zd : zoomData) {
            if (zd != null) // fragment level could be null
                zd.parsingComplete();
        }
    }

    int getChr1Idx() {
        return chr1Idx;
    }

    int getChr2Idx() {
        return chr2Idx;
    }

    int getChr3Idx() {
        return chr3Idx;
    }

    TensorZoomDataPP[] getZoomData() {
        return zoomData;
    }

}
