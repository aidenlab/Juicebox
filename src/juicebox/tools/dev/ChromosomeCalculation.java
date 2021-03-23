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

package juicebox.tools.dev;

import juicebox.HiCGlobals;
import juicebox.data.ContactRecord;
import juicebox.data.Dataset;
import juicebox.data.HiCFileTools;
import juicebox.data.MatrixZoomData;
import juicebox.data.basics.Chromosome;
import juicebox.tools.utils.common.ArrayTools;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.*;

public class ChromosomeCalculation {

    public static void sum(int resolution, int slidingWindow, String filePath, String outputFolder) {
        ArrayList<String> files = new ArrayList<>();
        File outFolder = new File(outputFolder);
        if (!outFolder.exists()) {
            outFolder.mkdir();
        }

        String colString = "column_sum_" + resolution + "_" + slidingWindow + ".bedgraph";
        String diagString = "diagonal_val_" + resolution + "_" + slidingWindow + ".bedgraph";

        File columnSumsFile = new File(outputFolder, colString);
        File diagValFile = new File(outputFolder, diagString);
        File slidingAvgColumnSumsFile = new File(outputFolder, "slide_avg_" + colString);
        File slidingAvgDiagValFile = new File(outputFolder, "slide_avg_" + diagString);
        File logEnrichColumnSumsFile = new File(outputFolder, "log_enrich_" + colString);
        File logEnrichDiagValFile = new File(outputFolder, "log_enrich_" + diagString);

        HiCGlobals.useCache = false;

        files.add(filePath); // replace with hic file paths
        Dataset ds = HiCFileTools.extractDatasetForCLT(files, false); // see this class and its functions
        Chromosome[] chromosomes = ds.getChromosomeHandler().getAutosomalChromosomesArray();
        Map<Chromosome, Map<Integer, Float>> chromosomeToColumnSumsMap = new HashMap<>();
        Map<Chromosome, Map<Integer, Float>> chromosomeToDiagonalValueMap = new HashMap<>();


        for (int i = 0; i < chromosomes.length; i++) {
            Chromosome chromosome1 = chromosomes[i];
            for (int j = i; j < chromosomes.length; j++) {
                Chromosome chromosome2 = chromosomes[j];
                MatrixZoomData zd = HiCFileTools.getMatrixZoomData(ds, chromosome1, chromosome2, resolution); // 1,000,000 resolution
                if (zd == null) continue;
                // do the summing, iterate over contact records in matrixZoomData object
                sumColumn(zd, chromosomeToColumnSumsMap, chromosomeToDiagonalValueMap, chromosome1, chromosome2);
            }
        }

        writeDataToFile(chromosomeToColumnSumsMap, columnSumsFile, resolution);
        writeDataToFile(chromosomeToDiagonalValueMap, diagValFile, resolution);

        slidingAverageAcrossData(slidingWindow, chromosomeToColumnSumsMap);
        slidingAverageAcrossData(slidingWindow, chromosomeToDiagonalValueMap);

        writeDataToFile(chromosomeToColumnSumsMap, slidingAvgColumnSumsFile, resolution);
        writeDataToFile(chromosomeToDiagonalValueMap, slidingAvgDiagValFile, resolution);

        //calculateLogEnrichmentOfObservedOverExpected(chromosomeToColumnSumsMap);
        //calculateLogEnrichmentOfObservedOverExpected(chromosomeToDiagonalValueMap);

        //writeDataToFile(chromosomeToColumnSumsMap, logEnrichColumnSumsFile, resolution);
        //writeDataToFile(chromosomeToDiagonalValueMap, logEnrichDiagValFile, resolution);


    }

    public static void slidingAverageAcrossData(int slidingWindow, Map<Chromosome, Map<Integer, Float>> dataMap) {

        for (Map<Integer, Float> dataMapForChromosome : dataMap.values()) {

            int maxIndex = Collections.max(dataMapForChromosome.keySet());
            float[] values = new float[maxIndex + 1];

            for (int idx : dataMapForChromosome.keySet()) {
                values[idx] = dataMapForChromosome.get(idx);
            }

            float[] newAvgAfterSliding = ArrayTools.runSlidingAverageOnArray(slidingWindow, values);

            for (int idx : dataMapForChromosome.keySet()) {
                if (values[idx] > 0) {
                    dataMapForChromosome.put(idx, newAvgAfterSliding[idx]);
                }
            }
        }
    }

    private static void writeDataToFile(Map<Chromosome, Map<Integer, Float>> dataHashMap, File outputFile, int resolution) {
        try {
            BufferedWriter bw = new BufferedWriter((new OutputStreamWriter((new FileOutputStream(outputFile)))));


            for (Chromosome key : dataHashMap.keySet()) {
                List<Integer> indices = new ArrayList<>(dataHashMap.get(key).keySet());
                Collections.sort(indices);
                for (int index : indices) {
                    String s = key.getName() + "\t" +
                            (index) * resolution + "\t" +
                            (index + 1) * resolution + "\t" +
                            dataHashMap.get(key).get(index);

                    bw.write(s);
                    bw.newLine();
                }
            }

            bw.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    private static void sumColumn(MatrixZoomData zd,
                                  Map<Chromosome, Map<Integer, Float>> mapOfSums,
                                  Map<Chromosome, Map<Integer, Float>> mapOfDiagValues,
                                  Chromosome chrI,
                                  Chromosome chrJ) {

        if (chrI.getIndex() == chrJ.getIndex()) {
            Map<Integer, Float> subMapOfSumsForChr = mapOfSums.getOrDefault(chrI, new HashMap<>());
            Map<Integer, Float> subMapOfDiagForChr = mapOfDiagValues.getOrDefault(chrI, new HashMap<>());

            Iterator<ContactRecord> iterator = zd.getIteratorContainer().getNewContactRecordIterator();
            while (iterator.hasNext()) {
                ContactRecord contact = iterator.next();

                float count = contact.getCounts();
                int x = contact.getBinX();
                int y = contact.getBinY();
                if (x == y) {
                    subMapOfSumsForChr.put(x, subMapOfSumsForChr.getOrDefault(x, 0f) + count);
                    subMapOfDiagForChr.put(x, count);
                } else {
                    subMapOfSumsForChr.put(x, subMapOfSumsForChr.getOrDefault(x, 0f) + count);
                    subMapOfSumsForChr.put(y, subMapOfSumsForChr.getOrDefault(y, 0f) + count);
                }
            }
            mapOfSums.put(chrI, subMapOfSumsForChr);
            mapOfDiagValues.put(chrI, subMapOfDiagForChr);
        } else {
            Map<Integer, Float> subMap = mapOfSums.getOrDefault(chrI, new HashMap<>());
            Map<Integer, Float> subMap2 = mapOfSums.getOrDefault(chrJ, new HashMap<>());

            Iterator<ContactRecord> iterator = zd.getIteratorContainer().getNewContactRecordIterator();
            while (iterator.hasNext()) {
                ContactRecord contact = iterator.next();
                float count = contact.getCounts();
                int x = contact.getBinX();
                int y = contact.getBinY();
                subMap.put(x, subMap.getOrDefault(x, 0f) + count);
                subMap2.put(y, subMap.getOrDefault(y, 0f) + count);
            }
            mapOfSums.put(chrI, subMap);
            mapOfSums.put(chrJ, subMap2);
        }

    }

    public static void calculateLogEnrichmentOfObservedOverExpected(Map<Chromosome, Map<Integer, Float>> map) {
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

        float currentAverage = total / size;

        System.out.println(currentAverage);

        for (Chromosome key : map.keySet()) {
            for (int subKey : map.get(key).keySet()) {
                float newVal = (float) Math.log(map.get(key).get(subKey) / currentAverage);
                map.get(key).put(subKey, newVal);
            }
        }
    }
}
