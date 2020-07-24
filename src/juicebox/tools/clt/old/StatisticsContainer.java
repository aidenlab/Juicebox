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

package juicebox.tools.clt.old;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.text.NumberFormat;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public class StatisticsContainer {
    //Variables for calculating statistics
    public final Map<Integer,Integer>[] hindIII = new HashMap[2];
    public final Map<Integer,Integer>[] mapQ = new HashMap[2];
    public final Map<Integer,Integer>[] mapQInter = new HashMap[2];
    public final Map<Integer,Integer>[] mapQIntra = new HashMap[2];
    public final Map<Integer,Integer>[] innerM = new HashMap[2];
    public final Map<Integer,Integer>[] outerM = new HashMap[2];
    public final Map<Integer,Integer>[] rightM = new HashMap[2];
    public final Map<Integer,Integer>[] leftM = new HashMap[2];

    public int unique = 0;
    public int[] intraFragment = new int[2];
    public int[] threePrimeEnd = new int[2];
    public int[] fivePrimeEnd = new int[2];
    public int[] dangling = new int[2];
    public int[] ligation = new int[2];
    public int[] inner = new int[2];
    public int[] outer = new int[2];
    public int[] left = new int[2];
    public int[] right = new int[2];
    public int[] intra = new int[2];
    public int[] inter = new int[2];
    public int[] small = new int[2];
    public int[] large = new int[2];
    public int[] verySmall = new int[2];
    public int[] verySmallDangling = new int[2];
    public int[] smallDangling = new int[2];
    public int[] largeDangling = new int[2];
    public int[] interDangling = new int[2];
    public int[] trueDanglingIntraSmall = new int[2];
    public int[] trueDanglingIntraLarge = new int[2];
    public int[] trueDanglingInter = new int[2];
    public int[] totalCurrent = new int[2];
    public int[] underMapQ = new int[2];

    public int[] oneKBRes = new int[2];
    public int[] twoKBRes = new int[2];
    public int[] fiveKBRes = new int[2];
    public int[] oneKBResDangling = new int[2];
    public int[] twoKBResDangling = new int[2];
    public int[] fiveKBResDangling = new int[2];


    private static final long[] bins = {10, 12, 15, 19, 23, 28, 35, 43, 53, 66, 81, 100, 123, 152, 187, 231, 285, 351, 433, 534, 658, 811, 1000, 1233, 1520, 1874, 2310, 2848, 3511, 4329, 5337, 6579, 8111, 10000, 12328, 15199, 18738, 23101, 28480, 35112, 43288, 53367, 65793, 81113, 100000, 123285, 151991, 187382, 231013, 284804, 351119, 432876, 533670, 657933, 811131, 1000000, 1232847, 1519911, 1873817, 2310130, 2848036, 3511192, 4328761, 5336699, 6579332, 8111308, 10000000, 12328467, 15199111, 18738174, 23101297, 28480359, 35111917, 43287613, 53366992, 65793322, 81113083, 100000000, 123284674, 151991108, 187381742, 231012970, 284803587, 351119173, 432876128, 533669923, 657933225, 811130831, 1000000000, 1232846739, 1519911083, 1873817423, 2310129700L, 2848035868L, 3511191734L, 4328761281L, 5336699231L, 6579332247L, 8111308308L, 10000000000L};

    public void initMaps(){
        for(int i=0; i<hindIII.length; i++){
            hindIII[i] = new HashMap<>();
            mapQ[i] = new HashMap<>();
            mapQInter[i] = new HashMap<>();
            mapQIntra[i] = new HashMap<>();
            innerM[i] = new HashMap<>();
            outerM[i] = new HashMap<>();
            rightM[i] = new HashMap<>();
            leftM[i] = new HashMap<>();
        }
    }
    public void add(StatisticsContainer individualContainer, int numberOfMapQValues) {
        unique += individualContainer.unique;

        for (int j=0; j<numberOfMapQValues; j++) {
            for (int i = 1; i <= 2000; i++) {
                hindIII[j].put(i, hindIII[j].getOrDefault(i, 0) + individualContainer.hindIII[j].getOrDefault(i, 0));
            }
            for (int i = 1; i <= 200; i++) {
                mapQ[j].put(i, mapQ[j].getOrDefault(i, 0) + individualContainer.mapQ[j].getOrDefault(i, 0));
                mapQInter[j].put(i, mapQInter[j].getOrDefault(i, 0) + individualContainer.mapQInter[j].getOrDefault(i, 0));
                mapQIntra[j].put(i, mapQIntra[j].getOrDefault(i, 0) + individualContainer.mapQIntra[j].getOrDefault(i, 0));
            }
            for (int i = 1; i <= 100; i++) {
                innerM[j].put(i, innerM[j].getOrDefault(i, 0) + individualContainer.innerM[j].getOrDefault(i, 0));
                outerM[j].put(i, outerM[j].getOrDefault(i, 0) + individualContainer.outerM[j].getOrDefault(i, 0));
                rightM[j].put(i, rightM[j].getOrDefault(i, 0) + individualContainer.rightM[j].getOrDefault(i, 0));
                leftM[j].put(i, leftM[j].getOrDefault(i, 0) + individualContainer.leftM[j].getOrDefault(i, 0));

            }
        }

        for(int i=0;i<numberOfMapQValues;i++) {
            intraFragment[i] += individualContainer.intraFragment[i];
            threePrimeEnd[i] += individualContainer.threePrimeEnd[i];
            fivePrimeEnd[i] += individualContainer.fivePrimeEnd[i];
            dangling[i] += individualContainer.dangling[i];
            ligation[i] += individualContainer.ligation[i];
            inner[i] += individualContainer.inner[i];
            outer[i] += individualContainer.outer[i];
            left[i] += individualContainer.left[i];
            right[i] += individualContainer.right[i];
            intra[i] += individualContainer.intra[i];
            inter[i] += individualContainer.inter[i];
            small[i] += individualContainer.small[i];
            large[i] += individualContainer.large[i];
            verySmall[i] += individualContainer.verySmall[i];
            verySmallDangling[i] += individualContainer.verySmallDangling[i];
            smallDangling[i] += individualContainer.smallDangling[i];
            largeDangling[i] += individualContainer.largeDangling[i];
            interDangling[i] += individualContainer.interDangling[i];
            trueDanglingIntraSmall[i] += individualContainer.trueDanglingIntraSmall[i];
            trueDanglingIntraLarge[i] += individualContainer.trueDanglingIntraLarge[i];
            trueDanglingInter[i] += individualContainer.trueDanglingInter[i];
            totalCurrent[i] += individualContainer.totalCurrent[i];
            underMapQ[i] += individualContainer.underMapQ[i];
        }
    }

    private String commify(int value) {
        return NumberFormat.getNumberInstance(Locale.US).format(value);
    }

    public void outputStatsFile(List<String> statsFiles) {
        for (int i =0; i<statsFiles.size();i++) {
            boolean seq = false;
            int reads = 1;
            File statFile = new File(statsFiles.get(i));
            //output statistics file for first mapq calculation
            if (statFile.exists()) {
                try {
                    BufferedReader stats = new BufferedReader(new FileReader(statFile));
                    String statsData = stats.readLine();
                    while (statsData != null) {
                        if (statsData.contains("Sequenced")) {
                            seq = true;
                            String[] tokens = statsData.split(":");
                            reads = Integer.parseInt(tokens[1].replaceAll("[, ]", ""));
                        }
                        statsData = stats.readLine();
                    }
                    stats.close();
                } catch (IOException error) {
                    error.printStackTrace();
                }
            }
            if (statFile.exists()) {
                try {
                    BufferedWriter statsOut = new BufferedWriter(new FileWriter(statFile, true));
                    if (unique == 0) {
                        unique++;
                    }
                    statsOut.write("Intra-fragment Reads: " + commify(intraFragment[i]) + " (");
                    if (seq) {
                        statsOut.write(String.format("%.2f", (float) intraFragment[i] * 100 / reads) + "%");
                    } else {
                        statsOut.write("(");
                    }
                    statsOut.write(" / " + String.format("%.2f", (float) intraFragment[i] * 100 / unique) + "%)" + "\n");

                    statsOut.write("Below MAPQ Threshold: " + commify(underMapQ[i]) + " (");
                    if (seq) {
                        statsOut.write(String.format("%.2f", (float) underMapQ[i] * 100 / reads) + "%");
                    } else {
                        statsOut.write("(");
                    }
                    statsOut.write(" / " + String.format("%.2f", (float) underMapQ[i] * 100 / unique) + "%)" + "\n");

                    statsOut.write("Hi-C Contacts: " + commify(totalCurrent[i]) + " (");
                    if (seq) {
                        statsOut.write(String.format("%.2f", (float) totalCurrent[i] * 100 / reads) + "%");
                    } else {
                        statsOut.write("(");
                    }
                    statsOut.write(" / " + String.format("%.2f", (float) totalCurrent[i] * 100 / unique) + "%)" + "\n");

                    statsOut.write(" Ligation Motif Present: " + commify(ligation[i]) + " (");
                    if (seq) {
                        statsOut.write(String.format("%.2f", (float) ligation[i] * 100 / reads) + "%");
                    } else {
                        statsOut.write("(");
                    }
                    statsOut.write(" / " + String.format("%.2f", (float) ligation[i] * 100 / unique) + "%)" + "\n");

                    if ((fivePrimeEnd[i] + threePrimeEnd[i]) > 0) {
                        float f1 = (float) threePrimeEnd[i] * 100f / (threePrimeEnd[i] + fivePrimeEnd[i]);
                        float f2 = (float) fivePrimeEnd[i] * 100f / (threePrimeEnd[i] + fivePrimeEnd[i]);
                        statsOut.write(" 3' Bias (Long Range): " + (String.format("%.0f", f1)) + "%");
                        statsOut.write(" - " + (String.format("%.0f", f2)) + "%" + "\n");
                    } else {
                        statsOut.write(" 3' Bias (Long Range): 0\\% \\- 0\\%\n");
                    }
                    if (large[i] > 0) {
                        statsOut.write(" Pair Type %(L-I-O-R): " + (String.format("%.0f", (float) left[i] * 100 / large[i])) + "%");
                        statsOut.write(" - " + (String.format("%.0f", (float) inner[i] * 100 / large[i])) + "%");
                        statsOut.write(" - " + (String.format("%.0f", (float) outer[i] * 100 / large[i])) + "%");
                        statsOut.write(" - " + (String.format("%.0f", (float) right[i] * 100 / large[i])) + "%" + "\n");
                    } else {
                        statsOut.write(" Pair Type %(L-I-O-R): 0\\% - 0\\% - 0\\% - 0\\%\n");
                    }

                    statsOut.write("Inter-chromosomal: " + commify(inter[i]) + " (");
                    if (seq) {
                        statsOut.write(String.format("%.2f", (float) inter[i] * 100 / reads) + "%");
                    } else {
                        statsOut.write("(");
                    }
                    statsOut.write(" / " + String.format("%.2f", (float) inter[i] * 100 / unique) + "%)" + "\n");

                    statsOut.write("Intra-chromosomal: %s " + commify(intra[i]) + " (");
                    if (seq) {
                        statsOut.write(String.format("%.2f", (float) intra[i] * 100 / reads) + "%");
                    } else {
                        statsOut.write("(");
                    }
                    statsOut.write(" / " + String.format("%.2f", (float) intra[i] * 100 / unique) + "%)" + "\n");

                    statsOut.write("Short Range (<20Kb): " + commify(small[i]) + " (");
                    if (seq) {
                        statsOut.write(String.format("%.2f", (float) small[i] * 100 / reads) + "%");
                    } else {
                        statsOut.write("(");
                    }
                    statsOut.write(" / " + String.format("%.2f", (float) small[i] * 100 / unique) + "%)" + "\n");

                    statsOut.write("Long Range (<20Kb): " + commify(large[i]) + " (");
                    if (seq) {
                        statsOut.write(String.format("%.2f", (float) large[i] * 100 / reads) + "%");
                    } else {
                        statsOut.write("(");
                    }
                    statsOut.write(" / " + String.format("%.2f", (float) large[i] * 100 / unique) + "%)" + "\n");
                    statsOut.close();
                } catch (IOException error) {
                    error.printStackTrace();
                }
            }
        }
    }

    public void writeHistFile(List<String> statsFiles) {
        //write for mapq if file exists
        for (int j = 0; j < statsFiles.size(); j++) {
            if (new File(statsFiles.get(j)).exists()) {
                //separate stats file name
                int index = statsFiles.get(j).lastIndexOf("\\");
                String statsFilePath = statsFiles.get(j).substring(0, index + 1); //directories
                String statsFileName = statsFiles.get(j).substring(index + 1).replaceAll(".txt", ""); //filename
                String histsFile = statsFilePath + statsFileName + "_hists.m";
                try {
                    BufferedWriter hist = new BufferedWriter(new FileWriter(histsFile, StandardCharsets.UTF_8, false));
                    hist.write("A = [\n");
                    for (int i = 1; i <= 2000; i++) {
                        int tmp = hindIII[j].getOrDefault(i, 0);
                        hist.write(tmp + " ");
                    }
                    hist.write("\n];\n");

                    hist.write("B = [\n");
                    for (int i = 1; i <= 200; i++) {
                        int tmp = mapQ[j].getOrDefault(i, 0);
                        int tmp2 = mapQIntra[j].getOrDefault(i, 0);
                        int tmp3 = mapQInter[j].getOrDefault(i, 0);
                        hist.write(tmp + " " + tmp2 + " " + tmp3 + "\n");
                    }
                    hist.write("\n];\n");

                    hist.write("D = [\n");
                    for (int i = 0; i < bins.length; i++) {
                        int tmp = innerM[j].getOrDefault(i, 0);
                        int tmp2 = outerM[j].getOrDefault(i, 0);
                        int tmp3 = rightM[j].getOrDefault(i, 0);
                        int tmp4 = leftM[j].getOrDefault(i, 0);
                        hist.write(tmp + " " + tmp2 + " " + tmp3 + " " + tmp4 + "\n");
                    }
                    hist.write("\n];");

                    hist.write("x = [\n");
                    for (long bin : bins) {
                        hist.write(bin + " ");
                    }
                    hist.write("\n];\n");
                    hist.close();
                } catch (IOException error) {
                    error.printStackTrace();
                }
            }
        }
    }
}
