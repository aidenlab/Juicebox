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

package juicebox.tools.dev;

import juicebox.HiC;
import juicebox.data.*;
import juicebox.windowui.HiCZoom;
import org.broad.igv.feature.Chromosome;


import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class ChromosomeCalculation {

    public ArrayList<Float> sum(String filePath) {
        ArrayList<String> files = new ArrayList<>();
        ArrayList<Float> res = new ArrayList<>();
        files.add(filePath); // replace with hic file paths
        Dataset ds = HiCFileTools.extractDatasetForCLT(files, false); // see this class and its functions
        Chromosome[] chromosomes = ds.getChromosomeHandler().getAutosomalChromosomesArray();
        for (int i = 0; i < chromosomes.length; i++) {
            Chromosome chromosome1 = chromosomes[i];
            for (int j = i; j < chromosomes.length; j++) {
                Chromosome chromosome2 = chromosomes[j];
                Matrix matrix = ds.getMatrix(chromosome1, chromosome2);
                MatrixZoomData zd = matrix.getZoomData(new HiCZoom(HiC.Unit.BP, 1000000)); // 1,000,000 resolution
                // do the summing, iterate over contact records in matrixZoomData object
                res.add(helper(zd));
            }
        }
        // save result
        return res;
    }

    private float helper(MatrixZoomData m) {
        float total = 0;
        final List<ContactRecord> contactRecordList  = m.getContactRecordList();
        for (ContactRecord contact: contactRecordList) {
            total += contact.getCounts();

        }

        return total;


    }


}
