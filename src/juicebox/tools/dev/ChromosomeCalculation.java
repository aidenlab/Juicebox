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
import juicebox.HiCGlobals;
import juicebox.data.*;
import juicebox.windowui.HiCZoom;
import org.broad.igv.feature.Chromosome;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.*;

public class ChromosomeCalculation {

    public static void sum(String filePath) {
        ArrayList<String> files = new ArrayList<>();
        File outputFile = new File("ChromosomeCalculationResult.bedgraph");

        HiCGlobals.useCache = false;

        files.add(filePath); // replace with hic file paths
        Dataset ds = HiCFileTools.extractDatasetForCLT(files, false); // see this class and its functions
        Chromosome[] chromosomes = ds.getChromosomeHandler().getAutosomalChromosomesArray();
        Map<Chromosome, Map<Integer, Float>> chromosomeToColumnSumsMap = new HashMap<>();
        int resolution = 2500000;


        for (int i = 0; i < chromosomes.length; i++) {
            Chromosome chromosome1 = chromosomes[i];
            for (int j = i; j < chromosomes.length; j++) {
                Chromosome chromosome2 = chromosomes[j];
                Matrix matrix = ds.getMatrix(chromosome1, chromosome2);
                MatrixZoomData zd =
                    matrix.getZoomData(new HiCZoom(HiC.Unit.BP, resolution)); // 1,000,000 resolution
                // do the summing, iterate over contact records in matrixZoomData object
                sumColumn(zd, chromosomeToColumnSumsMap, chromosome1, chromosome2);
            }
        }

        linearize(chromosomeToColumnSumsMap);

        try {
            BufferedWriter bw = new BufferedWriter((new OutputStreamWriter((new FileOutputStream(outputFile)))));

            for (Chromosome key : chromosomeToColumnSumsMap.keySet()) {
                List<Integer> indices = new ArrayList<>(chromosomeToColumnSumsMap.get(key).keySet());
                Collections.sort(indices);
                for (int index : indices) {
                    String s = key.getName() + "\t" +
                                   (index) * resolution + "\t" +
                                   (index + 1) * resolution + "\t" +
                                   chromosomeToColumnSumsMap.get(key).get(index);

                    bw.write(s);
                    bw.newLine();
                }
            }

            bw.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    private static void sumColumn(MatrixZoomData m,
                                  Map<Chromosome, Map<Integer, Float>> map,
                                  Chromosome chrI,
                                  Chromosome chrJ) {

        if (chrI.getIndex() == chrJ.getIndex()) {
            Map<Integer, Float> subMap = map.getOrDefault(chrI, new HashMap<>());
            for (ContactRecord contact : m.getContactRecordList()) {
                float count = contact.getCounts();
                int x = contact.getBinX();
                int y = contact.getBinY();
                if (x == y) {
                    subMap.put(x, subMap.getOrDefault(x, 0f) + count);
                } else {
                    subMap.put(x, subMap.getOrDefault(x, 0f) + count);
                    subMap.put(y, subMap.getOrDefault(y, 0f) + count);
                }
            }
            map.put(chrI, subMap);
        } else {
            Map<Integer, Float> subMap = map.getOrDefault(chrI, new HashMap<>());
            Map<Integer, Float> subMap2 = map.getOrDefault(chrJ, new HashMap<>());
            for (ContactRecord contact : m.getContactRecordList()) {
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

    public static void linearize(Map<Chromosome, Map<Integer, Float>> map) {
        float total = 0;
        int size = 0;

        for (Chromosome key : map.keySet()) {
            for (int subKey : map.get(key).keySet()) {
                float val = map.get(key).get(subKey);
                if (val > 1) {
                    total += val;
                    size += 1;
                }
            }
        }

        float expectedVal = total / size;

        System.out.println(expectedVal);
        for (Chromosome key : map.keySet()) {
            for (int subKey : map.get(key).keySet()) {
                float newVal = (float) Math.log(map.get(key).get(subKey) / expectedVal);
                map.get(key).put(subKey, newVal);
            }
        }
    }
}
