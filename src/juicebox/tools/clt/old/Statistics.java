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
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import juicebox.data.ChromosomeHandler;
import juicebox.tools.utils.original.FragmentCalculation;

public class Statistics extends JuiceboxCLT {

    private int numThreads;
    private int mndIndexSize;
    private String siteFile;
    private String ligationJunction;
    private String inFile;
    private String mndIndexFile;
    private ChromosomeHandler localHandler;
    private Map<Integer, Long> mndIndex;
    private FragmentCalculation chromosomes;
    private final List<String> statsFiles = new ArrayList<>();
    private final List<Integer> mapqThresholds = new ArrayList<>();

    public Statistics() {
        //constructor
        super(getUsage());
    }

    public static String getUsage() {
        return " Usage: statistics [--ligation NNNN] [--mapqs mapq1,maqp2] [--mndindex mndindex.txt] [--threads numthreads]\n " +
                "                   <site_file> <stats_file> [stats_file_2] <infile> <genomeID> [outfile]\n" +
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
                        StatisticsWorker runner = new StatisticsWorker(siteFile, statsFiles, mapqThresholds,
                                ligationJunction, inFile, localHandler, mndIndexStart, chromosomes);
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
        statsFiles.add(args[2]);
        if (args.length == 6) {// two map q values,input text files
            statsFiles.add(args[3]);
            inFile = args[4];
            localHandler = HiCFileTools.loadChromosomes(args[5]); //genomeID
        } else {//only one mapq value
            inFile = args[3];
            localHandler = HiCFileTools.loadChromosomes(args[4]);
        }
        //check for flags, else use default values
        List<Integer> mapQT = parser.getMultipleMapQOptions();
        if (mapQT != null && (mapQT.size() == 1 || mapQT.size() == 2)) { //only one or two mapq values
            int mapqThreshold = mapQT.get(0) > 0 ? mapQT.get(0) : 1;
            mapqThresholds.add(mapqThreshold);
            mapqThreshold = 30;
            if (mapQT.size() == 2) {
                mapqThreshold = mapQT.get(1) > 0 ? mapQT.get(1) : 30;
            }
            mapqThresholds.add(mapqThreshold);
        }
        else {
            mapqThresholds.add(1);
            if (statsFiles.size() == 2) {
                mapqThresholds.add(30);
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
        mergedResults.initMaps();
        for (StatisticsContainer sc : statisticsResults) {
            mergedResults.add(sc,statsFiles.size());
        }
        return mergedResults;
    }

    @Override
    public void run() {
        setMndIndex();
        readSiteFile();
        if (mndIndex == null || numThreads == 1) {
            StatisticsWorker runner = new StatisticsWorker(siteFile, statsFiles, mapqThresholds,
                    ligationJunction, inFile, localHandler, chromosomes);
            runner.infileStatistics();
            runner.getResultsContainer().outputStatsFile(statsFiles);
            runner.getResultsContainer().writeHistFile(statsFiles);

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
            merge(statisticsResults).outputStatsFile(statsFiles);
            merge(statisticsResults).writeHistFile(statsFiles);
        }
    }
}
