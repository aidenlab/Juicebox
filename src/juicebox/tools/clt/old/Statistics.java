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
import juicebox.HiCGlobals;
import juicebox.data.HiCFileTools;
import juicebox.tools.clt.CommandLineParser;
import juicebox.tools.clt.JuiceboxCLT;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.text.NumberFormat;
import java.util.*;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import juicebox.data.ChromosomeHandler;
import juicebox.tools.utils.original.FragmentCalculation;

public class Statistics extends JuiceboxCLT {

    private int mapqThreshold;
    private int mapqThreshold2;
    private int numThreads;
    private int mndIndexSize;
    private String siteFile;
    private String statsFile;
    private String statsFile2;
    private String ligationJunction;
    private String inFile;
    private String mndIndexFile;
    private ChromosomeHandler localHandler;
    private Map<Integer, Long> mndIndex;
    private FragmentCalculation chromosomes;
    private static final long[] bins = {10, 12, 15, 19, 23, 28, 35, 43, 53, 66, 81, 100, 123, 152, 187, 231, 285, 351, 433, 534, 658, 811, 1000, 1233, 1520, 1874, 2310, 2848, 3511, 4329, 5337, 6579, 8111, 10000, 12328, 15199, 18738, 23101, 28480, 35112, 43288, 53367, 65793, 81113, 100000, 123285, 151991, 187382, 231013, 284804, 351119, 432876, 533670, 657933, 811131, 1000000, 1232847, 1519911, 1873817, 2310130, 2848036, 3511192, 4328761, 5336699, 6579332, 8111308, 10000000, 12328467, 15199111, 18738174, 23101297, 28480359, 35111917, 43287613, 53366992, 65793322, 81113083, 100000000, 123284674, 151991108, 187381742, 231012970, 284803587, 351119173, 432876128, 533669923, 657933225, 811130831, 1000000000, 1232846739, 1519911083, 1873817423, 2310129700L, 2848035868L, 3511191734L, 4328761281L, 5336699231L, 6579332247L, 8111308308L, 10000000000L};

    public Statistics() {
        //constructor
        super(getUsage());
    }

    public static String getUsage() {
        return " Usage: statistics [--ligation NNNN] [--mapqs mapq1,maqp2] [--mndindex mnd] [--threads thr]\n " +
                "                   <site file> <stats file> <infile> <genome ID> [stats file 2] [outfile]\n" +
                " --ligation: ligation junction\n" +
                " --mapqs: mapping quality threshold(s), do not consider reads < threshold\n" +
                " --mndindex: file of indices for merged nodups to read from\n" +
                " --threads: number of threads to be executed \n" +
                " <site file>: list of HindIII restriction sites, one line per chromosome\n" +
                " <stats file>: output file containing total reads, for library complexity\n" +
                " <infile>: file in intermediate format to calculate statistics on, can be stream\n" +
                " <genome ID>: file to create chromosome handler\n" +
                " [stats file 2]: output file containing total reads for second mapping quality threshold\n" +
                " [outfile]: output, results of fragment search\n";
    }

    public void setMndIndex() {
        if (mndIndexFile != null && mndIndexFile.length() > 1) {
            mndIndex = readMndIndex(mndIndexFile);
        }
    }

    public Map<Integer, Long> readMndIndex(String mndIndexFile) {
        int counter = 0;
        FileInputStream is;
        Map<Integer, Long> mndIndex = new ConcurrentHashMap<>();
        try {
            is = new FileInputStream(mndIndexFile);
            BufferedReader reader = new BufferedReader(new InputStreamReader(is), HiCGlobals.bufferSize);
            String nextLine;
            while ((nextLine = reader.readLine()) != null) {
                String[] nextEntry = nextLine.split(",");
                if (nextEntry.length != 2) {
                    System.err.println("Improperly formatted merged nodups index");
                    System.exit(70);
                } else {
                    mndIndex.put(counter++, Long.parseLong(nextEntry[1]));
                }
            }
        } catch (Exception e) {
            System.err.println("Unable to read merged nodups index");
            System.exit(70);
        }
        mndIndexSize = counter;
        return mndIndex;
    }

    public void readSiteFile() {
        //read in restriction site file and store as multidimensional array q
        if (!siteFile.contains("none")) {
            //if restriction enzyme exists, find the RE distance//
            try {
                chromosomes = FragmentCalculation.readFragments(siteFile, localHandler);
            } catch (IOException error) {
                error.printStackTrace();
            }
        }
    }

    private final AtomicInteger threadCounter = new AtomicInteger();

    public void runIndividualStatistics(List<StatisticsContainer> statisticsResults) {
        long mndIndexStart;
        while (threadCounter.get() < mndIndexSize) {
            if (mndIndex != null) {
                int currentCount = threadCounter.getAndIncrement();
                if (!mndIndex.containsKey(currentCount)) {
                    System.out.println("Index position does not exist");
                } else {
                    mndIndexStart = mndIndex.get(currentCount);
                    try {
                        StatisticsWorker runner = new StatisticsWorker(siteFile, statsFile, statsFile2, mapqThreshold,
                                mapqThreshold2, ligationJunction, inFile, localHandler, mndIndexStart, chromosomes);
                        runner.infileStatistics();
                        statisticsResults.add(runner.getResultsContainer());
                    } catch (Exception e2) {
                        e2.printStackTrace();
                    }
                }
            }
        }
    }

    @Override
    public void readArguments(String[] args, CommandLineParser parser) {
        if (args.length != 6 && args.length != 5) {
            printUsageAndExit();
        }
        //set required arguments to variables
        siteFile = args[1];
        statsFile = args[2];
        if (args.length == 6) {// two map q values,input text files
            statsFile2 = args[3];
            inFile = args[4];
            localHandler = HiCFileTools.loadChromosomes(args[5]); //genomeID
        } else {//only one mapq value
            inFile = args[3];
            localHandler = HiCFileTools.loadChromosomes(args[4]);
        }
        //check for flags, else use default values
        List<Integer> mapQT = parser.getMultipleMapQOptions();
        if (mapQT != null && (mapQT.size() == 1 || mapQT.size() == 2)) { //only one or two mapq values
            mapqThreshold = mapQT.get(0) > 0 ? mapQT.get(0) : 1;
            if (mapQT.size() == 2) {
                mapqThreshold2 = mapQT.get(1) > 0 ? mapQT.get(1) : 30;
            }
        }
        String ligJunc = parser.getLigationOption();
        if (ligJunc != null) {
            ligationJunction = ligJunc;
        }
        //multithreading flags
        numThreads = Math.max(parser.getNumThreads(), 1);
        mndIndexFile = parser.getMndIndexOption();
    }

    public StatisticsContainer merge(List<StatisticsContainer> statisticsResults) {
        StatisticsContainer mergedResults = new StatisticsContainer();
        for (StatisticsContainer sc : statisticsResults) {
            mergedResults.add(sc);
        }
        return mergedResults;
    }

    @Override
    public void run() {
        setMndIndex();
        readSiteFile();
        if (mndIndex == null || numThreads == 1) {
            StatisticsWorker runner = new StatisticsWorker(siteFile, statsFile, statsFile2,
                    mapqThreshold, mapqThreshold2, ligationJunction, inFile, localHandler, chromosomes);
            runner.infileStatistics();
            outputStatsFile(runner.getResultsContainer());
            writeHistFile(runner.getResultsContainer());
        } else {
            List<StatisticsContainer> statisticsResults = Collections.synchronizedList(new ArrayList<>());
            ExecutorService executor = Executors.newFixedThreadPool(numThreads);
            for (int l = 0; l < numThreads; l++) {
                Runnable worker = () -> runIndividualStatistics(statisticsResults);
                executor.execute(worker);
            }
            executor.shutdown();
            // Wait until all threads finish
            while (!executor.isTerminated()) {
            }
            outputStatsFile(merge(statisticsResults));
            writeHistFile(merge(statisticsResults));
        }
    }

    private String commify(int value) {
        return NumberFormat.getNumberInstance(Locale.US).format(value);
    }

    public void outputStatsFile(StatisticsContainer mergedResults) {
        boolean seq = false;
        int reads = 1;
        File statFile = new File(statsFile);
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
                if (mergedResults.unique == 0) {
                    mergedResults.unique++;
                }
                statsOut.write("Intra-fragment Reads: " + commify(mergedResults.intraFragment) + " (");
                if (seq) {
                    statsOut.write(String.format("%.2f", (float) mergedResults.intraFragment * 100 / reads) + "%");
                } else {
                    statsOut.write("(");
                }
                statsOut.write(" / " + String.format("%.2f", (float) mergedResults.intraFragment * 100 / mergedResults.unique) + "%)" + "\n");

                statsOut.write("Below MAPQ Threshold: " + commify(mergedResults.underMapQ) + " (");
                if (seq) {
                    statsOut.write(String.format("%.2f", (float) mergedResults.underMapQ * 100 / reads) + "%");
                } else {
                    statsOut.write("(");
                }
                statsOut.write(" / " + String.format("%.2f", (float) mergedResults.underMapQ * 100 / mergedResults.unique) + "%)" + "\n");

                statsOut.write("Hi-C Contacts: " + commify(mergedResults.totalCurrent) + " (");
                if (seq) {
                    statsOut.write(String.format("%.2f", (float) mergedResults.totalCurrent * 100 / reads) + "%");
                } else {
                    statsOut.write("(");
                }
                statsOut.write(" / " + String.format("%.2f", (float) mergedResults.totalCurrent * 100 / mergedResults.unique) + "%)" + "\n");

                statsOut.write(" Ligation Motif Present: " + commify(mergedResults.ligation) + " (");
                if (seq) {
                    statsOut.write(String.format("%.2f", (float) mergedResults.ligation * 100 / reads) + "%");
                } else {
                    statsOut.write("(");
                }
                statsOut.write(" / " + String.format("%.2f", (float) mergedResults.ligation * 100 / mergedResults.unique) + "%)" + "\n");

                if ((mergedResults.fivePrimeEnd + mergedResults.threePrimeEnd) > 0) {
                    float f1 = (float) mergedResults.threePrimeEnd * 100f / (mergedResults.threePrimeEnd + mergedResults.fivePrimeEnd);
                    float f2 = (float) mergedResults.fivePrimeEnd * 100f / (mergedResults.threePrimeEnd + mergedResults.fivePrimeEnd);
                    statsOut.write(" 3' Bias (Long Range): " + (String.format("%.0f", f1)) + "%");
                    statsOut.write(" - " + (String.format("%.0f", f2)) + "%" + "\n");
                } else {
                    statsOut.write(" 3' Bias (Long Range): 0\\% \\- 0\\%\n");
                }
                if (mergedResults.large > 0) {
                    statsOut.write(" Pair Type %(L-I-O-R): " + (String.format("%.0f", (float) mergedResults.left * 100 / mergedResults.large)) + "%");
                    statsOut.write(" - " + (String.format("%.0f", (float) mergedResults.inner * 100 / mergedResults.large)) + "%");
                    statsOut.write(" - " + (String.format("%.0f", (float) mergedResults.outer * 100 / mergedResults.large)) + "%");
                    statsOut.write(" - " + (String.format("%.0f", (float) mergedResults.right * 100 / mergedResults.large)) + "%" + "\n");
                } else {
                    statsOut.write(" Pair Type %(L-I-O-R): 0\\% - 0\\% - 0\\% - 0\\%\n");
                }

                statsOut.write("Inter-chromosomal: " + commify(mergedResults.inter) + " (");
                if (seq) {
                    statsOut.write(String.format("%.2f", (float) mergedResults.inter * 100 / reads) + "%");
                } else {
                    statsOut.write("(");
                }
                statsOut.write(" / " + String.format("%.2f", (float) mergedResults.inter * 100 / mergedResults.unique) + "%)" + "\n");

                statsOut.write("Intra-chromosomal: %s " + commify(mergedResults.intra) + " (");
                if (seq) {
                    statsOut.write(String.format("%.2f", (float) mergedResults.intra * 100 / reads) + "%");
                } else {
                    statsOut.write("(");
                }
                statsOut.write(" / " + String.format("%.2f", (float) mergedResults.intra * 100 / mergedResults.unique) + "%)" + "\n");

                statsOut.write("Short Range (<20Kb): " + commify(mergedResults.small) + " (");
                if (seq) {
                    statsOut.write(String.format("%.2f", (float) mergedResults.small * 100 / reads) + "%");
                } else {
                    statsOut.write("(");
                }
                statsOut.write(" / " + String.format("%.2f", (float) mergedResults.small * 100 / mergedResults.unique) + "%)" + "\n");

                statsOut.write("Long Range (<20Kb): " + commify(mergedResults.large) + " (");
                if (seq) {
                    statsOut.write(String.format("%.2f", (float) mergedResults.large * 100 / reads) + "%");
                } else {
                    statsOut.write("(");
                }
                statsOut.write(" / " + String.format("%.2f", (float) mergedResults.large * 100 / mergedResults.unique) + "%)" + "\n");
                statsOut.close();
            } catch (IOException error) {
                error.printStackTrace();
            }
        }
        seq = false;
        reads = 1;
        //output statistics file for first mapq calculation
        if (statsFile2 != null && new File(statsFile2).exists()) {
            File statFile2 = new File(statsFile2);
            try {
                BufferedReader stats = new BufferedReader(new FileReader(statFile2));
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
        if (statsFile2 != null && new File(statsFile2).exists()) {
            File statFile2 = new File(statsFile2);
            try {
                BufferedWriter statsOut = new BufferedWriter(new FileWriter(statFile2, true));
                if (mergedResults.unique == 0) {
                    mergedResults.unique++;
                }
                statsOut.write("Intra-fragment Reads: " + commify(mergedResults.intraFragment) + " (");
                if (seq) {
                    statsOut.write(String.format("%.2f", (float) mergedResults.intraFragment * 100 / reads) + "%");
                } else {
                    statsOut.write("(");
                }
                statsOut.write(" / " + String.format("%.2f", (float) mergedResults.intraFragment * 100 / mergedResults.unique) + "%)" + "\n");

                statsOut.write("Below MAPQ Threshold: " + commify(mergedResults.underMapQ2) + " (");
                if (seq) {
                    statsOut.write(String.format("%.2f", (float) mergedResults.underMapQ2 * 100 / reads) + "%");
                } else {
                    statsOut.write("(");
                }
                statsOut.write(" / " + String.format("%.2f", (float) mergedResults.underMapQ2 * 100 / mergedResults.unique) + "%)" + "\n");

                statsOut.write("Hi-C Contacts: " + commify(mergedResults.totalCurrent2) + " (");
                if (seq) {
                    statsOut.write(String.format("%.2f", (float) mergedResults.totalCurrent2 * 100 / reads) + "%");
                } else {
                    statsOut.write("(");
                }
                statsOut.write(" / " + String.format("%.2f", (float) mergedResults.totalCurrent2 * 100 / mergedResults.unique) + "%)" + "\n");

                statsOut.write(" Ligation Motif Present: " + commify(mergedResults.ligation2) + " (");
                if (seq) {
                    statsOut.write(String.format("%.2f", (float) mergedResults.ligation2 * 100 / reads) + "%");
                } else {
                    statsOut.write("(");
                }
                statsOut.write(" / " + String.format("%.2f", (float) mergedResults.ligation2 * 100 / mergedResults.unique) + "%)" + "\n");

                if ((mergedResults.fivePrimeEnd2 + mergedResults.threePrimeEnd2) > 0) {
                    float f1 = (float) mergedResults.threePrimeEnd2 * 100f / (mergedResults.threePrimeEnd2 + mergedResults.fivePrimeEnd2);
                    float f2 = (float) mergedResults.fivePrimeEnd2 * 100f / (mergedResults.threePrimeEnd2 + mergedResults.fivePrimeEnd2);
                    statsOut.write(" 3' Bias (Long Range): " + (String.format("%.0f", f1)) + "%");
                    statsOut.write(" - " + (String.format("%.0f", f2)) + "%" + "\n");
                } else {
                    statsOut.write(" 3' Bias (Long Range): 0\\% \\- 0\\%\n");
                }
                if (mergedResults.large > 0) {
                    statsOut.write(" Pair Type %(L-I-O-R): " + (String.format("%.0f", (float) mergedResults.left2 * 100 / mergedResults.large2)) + "%");
                    statsOut.write(" - " + (String.format("%.0f", (float) mergedResults.inner2 * 100 / mergedResults.large2)) + "%");
                    statsOut.write(" - " + (String.format("%.0f", (float) mergedResults.outer2 * 100 / mergedResults.large2)) + "%");
                    statsOut.write(" - " + (String.format("%.0f", (float) mergedResults.right2 * 100 / mergedResults.large2)) + "%" + "\n");
                } else {
                    statsOut.write(" Pair Type %(L-I-O-R): 0\\% - 0\\% - 0\\% - 0\\%\n");
                }

                statsOut.write("Inter-chromosomal: " + commify(mergedResults.inter2) + " (");
                if (seq) {
                    statsOut.write(String.format("%.2f", (float) mergedResults.inter2 * 100 / reads) + "%");
                } else {
                    statsOut.write("(");
                }
                statsOut.write(" / " + String.format("%.2f", (float) mergedResults.inter2 * 100 / mergedResults.unique) + "%)" + "\n");

                statsOut.write("Intra-chromosomal: %s " + commify(mergedResults.intra2) + " (");
                if (seq) {
                    statsOut.write(String.format("%.2f", (float) mergedResults.intra2 * 100 / reads) + "%");
                } else {
                    statsOut.write("(");
                }
                statsOut.write(" / " + String.format("%.2f", (float) mergedResults.intra2 * 100 / mergedResults.unique) + "%)" + "\n");

                statsOut.write("Short Range (<20Kb): " + commify(mergedResults.small2) + " (");
                if (seq) {
                    statsOut.write(String.format("%.2f", (float) mergedResults.small2 * 100 / reads) + "%");
                } else {
                    statsOut.write("(");
                }
                statsOut.write(" / " + String.format("%.2f", (float) mergedResults.small2 * 100 / mergedResults.unique) + "%)" + "\n");

                statsOut.write("Long Range (<20Kb): " + commify(mergedResults.large2) + " (");
                if (seq) {
                    statsOut.write(String.format("%.2f", (float) mergedResults.large2 * 100 / reads) + "%");
                } else {
                    statsOut.write("(");
                }
                statsOut.write(" / " + String.format("%.2f", (float) mergedResults.large2 * 100 / mergedResults.unique) + "%)" + "\n");
                statsOut.close();
            } catch (IOException error) {
                error.printStackTrace();
            }
        }
    }

    public void writeHistFile(StatisticsContainer mergedResults) {
        //write for first mapq if file exists
        if (new File(statsFile).exists()) {
            //separate stats file name
            int index = statsFile.lastIndexOf("\\");
            String statsFilePath = statsFile.substring(0, index + 1); //directories
            String statsFileName = statsFile.substring(index + 1).replaceAll(".txt", ""); //filename
            String histsFile = statsFilePath + statsFileName + "_hists.m";
            try {
                BufferedWriter hist = new BufferedWriter(new FileWriter(histsFile, StandardCharsets.UTF_8, false));
                hist.write("A = [\n");
                for (int i = 1; i <= 2000; i++) {
                    int tmp = mergedResults.hindIII.getOrDefault(i, 0);
                    hist.write(tmp + " ");
                }
                hist.write("\n];\n");

                hist.write("B = [\n");
                for (int i = 1; i <= 200; i++) {
                    int tmp = mergedResults.mapQ.getOrDefault(i, 0);
                    int tmp2 = mergedResults.mapQIntra.getOrDefault(i, 0);
                    int tmp3 = mergedResults.mapQInter.getOrDefault(i, 0);
                    hist.write(tmp + " " + tmp2 + " " + tmp3 + "\n");
                }
                hist.write("\n];\n");

                hist.write("D = [\n");
                for (int i = 0; i < bins.length; i++) {
                    int tmp = mergedResults.innerM.getOrDefault(i, 0);
                    int tmp2 = mergedResults.outerM.getOrDefault(i, 0);
                    int tmp3 = mergedResults.rightM.getOrDefault(i, 0);
                    int tmp4 = mergedResults.leftM.getOrDefault(i, 0);
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
        //write to second file if it exists
        if (statsFile2 != null && new File(statsFile2).exists()) {
            //separate statsfile name
            int index = statsFile.lastIndexOf("\\");
            String statsFilePath = statsFile.substring(0, index + 1); //directories
            String statsFileName = statsFile.substring(index + 1).replaceAll(".txt", ""); //filename
            String histsFile = statsFilePath + statsFileName + "_hists.m";
            try {
                BufferedWriter hist = new BufferedWriter(new FileWriter(histsFile, StandardCharsets.UTF_8, false));
                hist.write("A = [\n");
                for (int i = 1; i <= 2000; i++) {
                    int tmp = mergedResults.hindIII2.getOrDefault(i, 0);
                    hist.write(tmp + " ");
                }
                hist.write("\n];\n");

                hist.write("B = [\n");
                for (int i = 1; i <= 200; i++) {
                    int tmp = mergedResults.mapQ2.getOrDefault(i, 0);
                    int tmp2 = mergedResults.mapQIntra2.getOrDefault(i, 0);
                    int tmp3 = mergedResults.mapQInter2.getOrDefault(i, 0);
                    hist.write(tmp + " " + tmp2 + " " + tmp3 + "\n");
                }
                hist.write("\n];\n");

                hist.write("D = [\n");
                for (int i = 0; i < bins.length; i++) {
                    int tmp = mergedResults.innerM2.getOrDefault(i, 0);
                    int tmp2 = mergedResults.outerM2.getOrDefault(i, 0);
                    int tmp3 = mergedResults.rightM2.getOrDefault(i, 0);
                    int tmp4 = mergedResults.leftM2.getOrDefault(i, 0);
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
