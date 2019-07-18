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

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class ChromosomeCalculation {

    public static void sum(String filePath) {
        ArrayList<String> files = new ArrayList<>();
        File outputFile = new File("ChromosomeCalculationResult.bedgraph");

        files.add(filePath); // replace with hic file paths
        Dataset ds = HiCFileTools.extractDatasetForCLT(files, false); // see this class and its functions
        Chromosome[] chromosomes = ds.getChromosomeHandler().getAutosomalChromosomesArray();
        HashMap<Chromosome, HashMap<Integer, Float>> chromosomeToColumnSumsMap = new HashMap<>();
        int resolution = 1000000;

        BufferedWriter bw = null;
        try {
            bw = new BufferedWriter((new OutputStreamWriter((new FileOutputStream(outputFile)))));


            for (int i = 0; i < chromosomes.length; i++) {
                Chromosome chromosome1 = chromosomes[i];
                for (int j = i; j < chromosomes.length; j++) {

                    Chromosome chromosome2 = chromosomes[j];
                    Matrix matrix = ds.getMatrix(chromosome1, chromosome2);
                    MatrixZoomData
                        zd =
                        matrix.getZoomData(new HiCZoom(HiC.Unit.BP, resolution)); // 1,000,000 resolution
                    // do the summing, iterate over contact records in matrixZoomData object
                    sumColumn(zd, chromosomeToColumnSumsMap, chromosome1, chromosome2);
                    //linearize(chromosomeToColumnSumsMap);
                    for (Chromosome key : chromosomeToColumnSumsMap.keySet()) {
                        for (int index : chromosomeToColumnSumsMap.get(key).keySet()) {

                            String
                                s =
                                key.getName() +
                                    "\t" +
                                    (index) * resolution +
                                    "\t" +
                                    (index + 1) * resolution +
                                    "\t" +
                                    chromosomeToColumnSumsMap.get(key).get(index);

                                bw.write(s);


                            bw.newLine();


                        }
                    }


                }
            }
            bw.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    private static void sumColumn(MatrixZoomData m,
                                  HashMap<Chromosome, HashMap<Integer, Float>> map,
                                  Chromosome chrI,
                                  Chromosome chrJ) {
        final List<ContactRecord> contactRecordList  = m.getContactRecordList();

            if (chrI.getIndex() == chrJ.getIndex()) {
                HashMap<Integer, Float> subMap = map.getOrDefault(chrI, new HashMap<>());
                for (ContactRecord contact: contactRecordList) {
                    float count = contact.getCounts();
                    int x = contact.getBinX();
                    int y = contact.getBinY();
                    if (x == y) {
                        subMap.put(x, subMap.getOrDefault(x, 0f) + count);


                    }
                    else {
                        subMap.put(x, subMap.getOrDefault(x,  count));
                        subMap.put(y, subMap.getOrDefault(y, 0f) + count);

                    }
                }
                map.put(chrI, subMap);
            }
            else {
                HashMap<Integer, Float> subMap = map.getOrDefault(chrI, new HashMap<>());
                HashMap<Integer, Float> subMap2 = map.getOrDefault(chrJ, new HashMap<>());
                for (ContactRecord contact: contactRecordList) {
                    float count = contact.getCounts();
                    int x = contact.getBinX();
                    int y = contact.getBinY();
                    subMap.put(x, subMap.getOrDefault(x, 0f) + count);
                    subMap2.put(y, subMap.getOrDefault(y, 0f) + count);
                }
                map.put(chrI, subMap);
                map.put(chrJ, subMap2);
            }

    }

    public static void linearize(HashMap<Chromosome, HashMap<Integer, Float>> map) {

        for (Chromosome key : map.keySet()) {
            float currTotal = 0;
            int count = 0;

            for (int subKey : map.get(key).keySet()) {
                currTotal += map.get(key).get(subKey);
            }


            float initialAverage = currTotal / map.get(key).keySet().size();

            for (int subKey : map.get(key).keySet()) {
                if (map.get(key).get(subKey) < initialAverage * 0.01) {
                    count += 1;
                    currTotal -= map.get(key).get(subKey);
                }
            }

            float finalAverage = currTotal / (map.get(key).keySet().size() - count);

            for (int subKey : map.get(key).keySet()) {
                map.get(key).put(subKey, finalAverage);
            }

        }
    }


}
