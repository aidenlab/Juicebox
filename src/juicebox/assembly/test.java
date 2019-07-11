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

package juicebox.assembly;

import juicebox.HiC;
import juicebox.data.Dataset;
import juicebox.data.HiCFileTools;
import juicebox.data.Matrix;
import juicebox.data.MatrixZoomData;
import juicebox.windowui.HiCZoom;
import org.apache.commons.math.genetics.Chromosome;

import java.util.ArrayList;

public class test {
    ArrayList<String> files = new ArrayList<>();
        files.add("/Users/muhammad/Desktop/testtemp/imr90_intra_nofrag_30.hic"); // replace with hic file paths
    Dataset ds = HiCFileTools.extractDatasetForCLT(files, false); // see this class and its functions
    Chromosome[] chromosomes = ds.getChromosomeHandler().getAutosomalChromosomesArray()
        for(int i = 0; i < chromosomes.length; i++) {
        Chromosome chromosome1 = chromosomes[i];
        for (int j = i; j < chromosomes.length; j++) {
            Chromosome chromosome2 = chromosomes[j];
            Matrix matrix = ds.getMatrix(chromosome1, chromosome2);
            MatrixZoomData zd = matrix.getZoomData(new HiCZoom(HiC.Unit.BP, 1000000)); // 1,000,000 resolution
            // do the summing, iterate over contact records in matrixZoomData object
        }
    }
    // save result
}
