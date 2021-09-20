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

package juicebox.tools.utils.original.stats;

import juicebox.tools.clt.old.LibraryComplexity;

import java.io.*;
import java.text.NumberFormat;
import java.util.*;

public class StatisticsContainer {

    private final static float CONVERGENCE_THRESHOLD = 0.01f;
    private final static int CONVERGENCE_REGION = 3;
    private final static int SEQ_INDEX = 0, DUPS_INDEX = 1, UNIQUE_INDEX = 2;
    private final static int LC_INDEX = 3, SINGLE_ALIGNMENT_INDEX = 4;
    private final static int SINGLE_ALIGN_DUPS_INDEX = 5, SINGLE_ALIGN_UNIQUE_INDEX = 6;
    private final static int NUM_TO_READ = 7;
    private final NumberFormat nf = NumberFormat.getNumberInstance(Locale.US);

    //Variables for calculating statistics
    public final List<Map<Integer, Long>> hindIII = new ArrayList<>();
    public final List<Map<Integer, Long>> mapQ = new ArrayList<>();
    public final List<Map<Integer, Long>> mapQInter = new ArrayList<>();
    public final List<Map<Integer, Long>> mapQIntra = new ArrayList<>();
    public final List<Map<Integer, Long>> innerM = new ArrayList<>();
    public final List<Map<Integer, Long>> outerM = new ArrayList<>();
    public final List<Map<Integer, Long>> rightM = new ArrayList<>();
    public final List<Map<Integer, Long>> leftM = new ArrayList<>();
    private final List<Integer> convergenceIndices = new ArrayList<>();

    public long unique = 0;
    public long[] intraFragment = new long[2];
    public long[] threePrimeEnd = new long[2];
    public long[] fivePrimeEnd = new long[2];
    public long[] dangling = new long[2];
    public long[] ligation = new long[2];
    public long[] inner = new long[2];
    public long[] outer = new long[2];
    public long[] left = new long[2];
    public long[] right = new long[2];
    public long[] intra = new long[2];
    public long[] inter = new long[2];

    public long[] interDangling = new long[2];
    public long[] trueDanglingIntraSmall = new long[2];
    public long[] trueDanglingIntraLarge = new long[2];
    public long[] trueDanglingInter = new long[2];
    public long[] totalCurrent = new long[2];
    public long[] underMapQ = new long[2];
    public long[] fiveHundredBPRes = new long[2];
    public long[] fiveHundredBPResDangling = new long[2];
    public long[] fiveKBRes = new long[2];
    public long[] fiveKBResDangling = new long[2];
    public long[] twentyKBRes = new long[2];
    public long[] twentyKBResDangling = new long[2];
    public long[] large = new long[2];
    public long[] largeDangling = new long[2];

    private static final long[] bins = {10, 12, 15, 19, 23, 28, 35, 43, 53, 66, 81, 100, 123, 152, 187, 231, 285, 351, 433, 534, 658, 811, 1000, 1233, 1520, 1874, 2310, 2848, 3511, 4329, 5337, 6579, 8111, 10000, 12328, 15199, 18738, 23101, 28480, 35112, 43288, 53367, 65793, 81113, 100000, 123285, 151991, 187382, 231013, 284804, 351119, 432876, 533670, 657933, 811131, 1000000, 1232847, 1519911, 1873817, 2310130, 2848036, 3511192, 4328761, 5336699, 6579332, 8111308, 10000000, 12328467, 15199111, 18738174, 23101297, 28480359, 35111917, 43287613, 53366992, 65793322, 81113083, 100000000, 123284674, 151991108, 187381742, 231012970, 284803587, 351119173, 432876128, 533669923, 657933225, 811130831, 1000000000, 1232846739, 1519911083, 1873817423, 2310129700L, 2848035868L, 3511191734L, 4328761281L, 5336699231L, 6579332247L, 8111308308L, 10000000000L};

    public StatisticsContainer() {
        for (int i = 0; i < 2; i++) {
            hindIII.add(new HashMap<>());
            mapQ.add(new HashMap<>());
            mapQInter.add(new HashMap<>());
            mapQIntra.add(new HashMap<>());
            innerM.add(new HashMap<>());
            outerM.add(new HashMap<>());
            rightM.add(new HashMap<>());
            leftM.add(new HashMap<>());
        }
    }
    
    public void add(StatisticsContainer individualContainer, int numberOfMapQValues) {
        unique += individualContainer.unique;

        for (int j=0; j<numberOfMapQValues; j++) {
            for (int i = 1; i <= 2000; i++) {
                hindIII.get(j).put(i, hindIII.get(j).getOrDefault(i, 0L) + individualContainer.hindIII.get(j).getOrDefault(i, 0L));
            }
            for (int i = 1; i <= 200; i++) {
                mapQ.get(j).put(i, mapQ.get(j).getOrDefault(i, 0L) + individualContainer.mapQ.get(j).getOrDefault(i, 0L));
                mapQInter.get(j).put(i, mapQInter.get(j).getOrDefault(i, 0L) + individualContainer.mapQInter.get(j).getOrDefault(i, 0L));
                mapQIntra.get(j).put(i, mapQIntra.get(j).getOrDefault(i, 0L) + individualContainer.mapQIntra.get(j).getOrDefault(i, 0L));
            }
            for (int i = 1; i <= 100; i++) {
                innerM.get(j).put(i, innerM.get(j).getOrDefault(i, 0L) + individualContainer.innerM.get(j).getOrDefault(i, 0L));
                outerM.get(j).put(i, outerM.get(j).getOrDefault(i, 0L) + individualContainer.outerM.get(j).getOrDefault(i, 0L));
                rightM.get(j).put(i, rightM.get(j).getOrDefault(i, 0L) + individualContainer.rightM.get(j).getOrDefault(i, 0L));
                leftM.get(j).put(i, leftM.get(j).getOrDefault(i, 0L) + individualContainer.leftM.get(j).getOrDefault(i, 0L));
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
            large[i] += individualContainer.large[i];
            twentyKBRes[i] += individualContainer.twentyKBRes[i];
            fiveKBRes[i] += individualContainer.fiveKBRes[i];
            fiveHundredBPRes[i] += individualContainer.fiveHundredBPRes[i];
            fiveHundredBPResDangling[i] += individualContainer.fiveHundredBPResDangling[i];
            fiveKBResDangling[i] += individualContainer.fiveKBResDangling[i];
            twentyKBResDangling[i] += individualContainer.twentyKBResDangling[i];
            largeDangling[i] += individualContainer.largeDangling[i];
            interDangling[i] += individualContainer.interDangling[i];
            trueDanglingIntraSmall[i] += individualContainer.trueDanglingIntraSmall[i];
            trueDanglingIntraLarge[i] += individualContainer.trueDanglingIntraLarge[i];
            trueDanglingInter[i] += individualContainer.trueDanglingInter[i];
            totalCurrent[i] += individualContainer.totalCurrent[i];
            underMapQ[i] += individualContainer.underMapQ[i];
        }
    }
    
    private String commify(long value) {
        return nf.format(value);
    }
    
    private String percentify(long num, long total) {
        return String.format("%.2f", num * 100f / total) + "%";
    }
    
    private String wholePercentify(long num, long total) {
        return String.format("%.0f", num * 100f / total) + "%";
    }
    
    public void outputStatsFile(List<String> statsFiles) {
        for (int i = 0; i < statsFiles.size(); i++) {
            File statFile = new File(statsFiles.get(i));
            //output statistics file for first mapq calculation
            boolean[] valsWereFound = new boolean[NUM_TO_READ];
            long[] valsFound = new long[NUM_TO_READ]; // seqReads, duplicates
            attempReadingDataFromExistingFile(valsWereFound, valsFound, statFile);

         //   if (statFile.exists()) {
                try {
                    BufferedWriter statsOut = new BufferedWriter(new FileWriter(statFile, true));
                    writeLibComplexityIfNeeded(valsWereFound, valsFound, statsOut);
                    if (unique == 0) unique = 1;
                    writeOut(statsOut, "Intra-fragment Reads: ", valsWereFound, intraFragment[i], valsFound, unique, true);
                    //if (!isUTLibrary(valsWereFound, valsFound, underMapQ[i])) {
                    attemptMapqCorrection(valsWereFound, valsFound, underMapQ, unique, i);
                    writeOut(statsOut, "Below MAPQ Threshold: ", valsWereFound, underMapQ[i], valsFound, unique, true);
                    //}
                    writeOut(statsOut, "Hi-C Contacts: ", valsWereFound, totalCurrent[i], valsFound, unique, false);
                    //writeOut(statsOut, " Ligation Motif Present: ", valsWereFound, ligation[i], valsFound, unique, true);
                    appendPairTypeStatsOutputToFile(i, statsOut);
                    writeOut(statsOut, "Inter-chromosomal: ", valsWereFound, inter[i], valsFound, unique, false);
                    writeOut(statsOut, "Intra-chromosomal: ", valsWereFound, intra[i], valsFound, unique, false);
                    statsOut.write("Short Range (<20Kb):\n");
                    writeOut(statsOut, "  <500BP: ", valsWereFound, fiveHundredBPRes[i], valsFound, unique, false);
                    writeOut(statsOut, "  500BP-5kB: ", valsWereFound, fiveKBRes[i], valsFound, unique, false);
                    writeOut(statsOut, "  5kB-20kB: ", valsWereFound, twentyKBRes[i], valsFound, unique, false);
                    writeOut(statsOut, "Long Range (>20Kb): ", valsWereFound, large[i], valsFound, unique, false);
                    statsOut.close();
                } catch (IOException error) {
                    error.printStackTrace();
                }
            //  }
        }
    }

    private boolean isUTLibrary(boolean[] valsWereFound, long[] valsFound, long belowMapQ) {
        return valsWereFound[SINGLE_ALIGNMENT_INDEX]
                && valsFound[SINGLE_ALIGNMENT_INDEX] > 0
                && belowMapQ < 1;
    }

    private void writeLibComplexityIfNeeded(boolean[] valsWereFound, long[] valsFound, BufferedWriter statsOut) throws IOException {
        if (!valsWereFound[LC_INDEX]) {
            boolean isUTExperiment = false;
            long lcTotal = 0L;
            if (valsWereFound[SINGLE_ALIGN_UNIQUE_INDEX] && valsWereFound[SINGLE_ALIGN_DUPS_INDEX]) {
                long resultLC1 = LibraryComplexity.estimateLibrarySize(valsFound[SINGLE_ALIGN_DUPS_INDEX], valsFound[SINGLE_ALIGN_UNIQUE_INDEX]);
                isUTExperiment = true;
                if (resultLC1 > 0) {
                    lcTotal += resultLC1;
                    statsOut.write("Library Complexity Estimate (1 alignment)*: " + commify(resultLC1) + "\n");
                } else {
                    statsOut.write("Library Complexity Estimate (1 alignment)*: N/A\n");
                }
            }

            if (valsWereFound[UNIQUE_INDEX] && valsWereFound[DUPS_INDEX]) {
                long resultLC2 = LibraryComplexity.estimateLibrarySize(valsFound[DUPS_INDEX], valsFound[UNIQUE_INDEX]);
                String description = "";
                if (isUTExperiment) {
                    description = " (2 alignments)";
                }
                if (resultLC2 > 0) {
                    lcTotal += resultLC2;
                    statsOut.write("Library Complexity Estimate" + description + "*: " + commify(resultLC2) + "\n");
                } else {
                    statsOut.write("Library Complexity Estimate" + description + "*: N/A\n");
                }
            }

            if (isUTExperiment) {
                if (lcTotal > 0) {
                    statsOut.write("Library Complexity Estimate (1+2 above)*: " + commify(lcTotal) + "\n");
                } else {
                    statsOut.write("Library Complexity Estimate (1+2 above)*: N/A\n");
                }
            }
        }
    }

    private void attempReadingDataFromExistingFile(boolean[] valsWereFound, long[] valsFound, File statFile) {
        Arrays.fill(valsWereFound, false);
        Arrays.fill(valsFound, 0);
        if (statFile.exists()) {
            try {
                BufferedReader stats = new BufferedReader(new FileReader(statFile));
                String statsData = stats.readLine();
                while (statsData != null) {
                    statsData = statsData.toLowerCase();
                    if (statsData.contains("sequenced")) {
                        populateFoundVals(statsData, valsWereFound, valsFound, SEQ_INDEX);
                    } else if (statsData.contains("unique")) {
                        if (isSingleAlignment(statsData)) {
                            populateFoundVals(statsData, valsWereFound, valsFound, SINGLE_ALIGN_UNIQUE_INDEX);
                        } else {
                            populateFoundVals(statsData, valsWereFound, valsFound, UNIQUE_INDEX);
                        }
                    } else if (statsData.contains("duplicate") && !statsData.contains("optical")) {
                        if (isSingleAlignment(statsData)) {
                            populateFoundVals(statsData, valsWereFound, valsFound, SINGLE_ALIGN_DUPS_INDEX);
                        } else {
                            populateFoundVals(statsData, valsWereFound, valsFound, DUPS_INDEX);
                        }
                    } else if (statsData.contains("complexity")) {
                        populateFoundVals(statsData, valsWereFound, valsFound, LC_INDEX);
                    } else if (statsData.contains("single") && statsData.contains("alignment")) {
                        populateFoundVals(statsData, valsWereFound, valsFound, SINGLE_ALIGNMENT_INDEX);
                    }
                    statsData = stats.readLine();
                }
                stats.close();
            } catch (IOException error) {
                error.printStackTrace();
            }
        }
    }

    private boolean isSingleAlignment(String text) {
        String[] tokens = text.split(":");
        String description = tokens[0].toLowerCase();
        return (description.contains("1") || description.contains("one")) &&
                description.contains("alignment");
    }

    private void populateFoundVals(String statsData, boolean[] valsWereFound, long[] valsFound, int index) {
        if (!valsWereFound[index]) {
            valsWereFound[index] = true;
            String[] tokens = statsData.split(":");
            String substring1 = tokens[1].replaceAll("[, ]", "");
            if (substring1.contains("(")) {
                substring1 = substring1.split("\\(")[0];
            }
            valsFound[index] = Long.parseLong(substring1.trim());
        }
    }

    private void attemptMapqCorrection(boolean[] valsWereGiven, long[] valsFound, long[] underMapQ, long unique, int i) {
        if (underMapQ[i] < 1 && valsWereGiven[UNIQUE_INDEX]) {
            underMapQ[i] = valsFound[UNIQUE_INDEX] - unique;
        }
    }

    void appendPairTypeStatsOutputToFile(int i, BufferedWriter statsOut) throws IOException {
        if ((fivePrimeEnd[i] + threePrimeEnd[i]) > 0) {
            statsOut.write(" 3' Bias (Long Range): " + wholePercentify(threePrimeEnd[i], threePrimeEnd[i] + fivePrimeEnd[i]));
            statsOut.write(" - " + wholePercentify(fivePrimeEnd[i], threePrimeEnd[i] + fivePrimeEnd[i]) + "\n");
        } else {
            statsOut.write(" 3' Bias (Long Range): N/A\n");
        }
        if (large[i] > 0) {
            statsOut.write(" Pair Type %(L-I-O-R): " + wholePercentify(left[i], large[i]));
            statsOut.write(" - " + wholePercentify(inner[i], large[i]));
            statsOut.write(" - " + wholePercentify(outer[i], large[i]));
            statsOut.write(" - " + wholePercentify(right[i], large[i]) + "\n");
            statsOut.write(" L-I-O-R Convergence: " + bins[convergenceIndices.get(i)] + "\n");
        } else {
            statsOut.write(" Pair Type %(L-I-O-R): N/A\n");
        }
    }

    private void writeOut(BufferedWriter statsOut, String description, boolean[] valsWereGiven, long value,
                          long[] valsFound, long unique, boolean checkNA) throws IOException {
        if (!checkNA || value > 0) {
            if (valsWereGiven[SEQ_INDEX] && valsWereGiven[UNIQUE_INDEX]) {
                statsOut.write(description + commify(value) + " (" + percentify(value, valsFound[SEQ_INDEX]) + " / " + percentify(value, valsFound[UNIQUE_INDEX]) + ")\n");
            } else {
                statsOut.write(description + commify(value) + " (" + percentify(value, unique) + ")\n");
            }
        } else {
            statsOut.write(description + "N/A\n");
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
                    BufferedWriter hist = new BufferedWriter(new FileWriter(histsFile, false));
                    hist.write("A = [\n");
                    for (int i = 1; i <= 2000; i++) {
                        long tmp = hindIII.get(j).getOrDefault(i, 0L);
                        hist.write(tmp + " ");
                    }
                    hist.write("\n];\n");

                    hist.write("B = [\n");
                    for (int i = 1; i <= 200; i++) {
                        long tmp = mapQ.get(j).getOrDefault(i, 0L);
                        long tmp2 = mapQIntra.get(j).getOrDefault(i, 0L);
                        long tmp3 = mapQInter.get(j).getOrDefault(i, 0L);
                        hist.write(tmp + " " + tmp2 + " " + tmp3 + "\n");
                    }
                    hist.write("\n];\n");

                    hist.write("D = [\n");
                    for (int i = 0; i < bins.length; i++) {
                        long tmp = innerM.get(j).getOrDefault(i, 0L);
                        long tmp2 = outerM.get(j).getOrDefault(i, 0L);
                        long tmp3 = rightM.get(j).getOrDefault(i, 0L);
                        long tmp4 = leftM.get(j).getOrDefault(i, 0L);
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


    public void calculateConvergence(int numMapQs) {
        convergenceIndices.clear();
        for (int q = 0; q < numMapQs; q++) {
            int index = -1;
            boolean solutionFound = false;
            while (!solutionFound && index < bins.length - 1) {
                index++;
                if (getConvergenceError(q, index) < CONVERGENCE_THRESHOLD) {
                    solutionFound = confirmConvergenceMaintained(q, index + 1, bins.length);
                }
            }
            convergenceIndices.add(index);
        }
    }

    private boolean confirmConvergenceMaintained(int q, int startIndex, int maxIndex) {
        boolean convergenceMaintained = true;
        for (int i = startIndex; i < Math.min(startIndex + CONVERGENCE_REGION, maxIndex); i++) {
            double error = getConvergenceError(q, i);
            convergenceMaintained &= (error < CONVERGENCE_THRESHOLD);
        }
        return convergenceMaintained;
    }

    private double getConvergenceError(int q, int i) {
        long[] vals = new long[]{
                innerM.get(q).getOrDefault(i, 0L),
                outerM.get(q).getOrDefault(i, 0L),
                rightM.get(q).getOrDefault(i, 0L),
                leftM.get(q).getOrDefault(i, 0L)};
        double total = 0.0;
        for (long val : vals) {
            total += val;
        }
        if (total < 1) return 1e3;

        double logAvg = Math.log(total / 4.0);
        double error = 0;
        for (long val : vals) {
            double tempErr = (logAvg - Math.log(val)) / logAvg;
            error = Math.max(error, Math.abs(tempErr));
        }
        return error;
    }
}
