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

package juicebox.tools.utils.juicer.curse;

import juicebox.data.ChromosomeHandler;
import juicebox.data.Dataset;
import juicebox.data.feature.GenomeWideList;
import juicebox.windowui.NormalizationType;

public class ScaledGenomeWideMatrix {

    public ScaledGenomeWideMatrix(ChromosomeHandler chromosomeHandler, Dataset ds, NormalizationType norm,
                                  int resolution, GenomeWideList<SubcompartmentInterval> subcompartments) {

        /*
        ExpectedValueFunction df = ds.getExpectedValues(zd.getZoom(), norm);

        // skip these matrices
        Matrix matrix = ds.getMatrix(chromosome, chromosome);
        if (matrix == null) continue;

        HiCZoom zoom = ds.getZoomForBPResolution(resolution);
        final MatrixZoomData zd = matrix.getZoomData(zoom);
        if (zd == null) continue;



        if (df == null) {
            System.err.println("O/E data not available at " + chromosome.getName() + " " + zoom + " " + norm);
            System.exit(14);
        }

        int maxBin = chromosome.getLength() / resolution + 1;
        int maxSize = maxBin + 1;

        RealMatrix localizedRegionData = HiCFileTools.extractLocalLogOEBoundedRegion(zd, 0, maxBin,
                0, maxBin, maxSize, maxSize, norm, true, df, chromosome.getIndex());

                */
    }
}
