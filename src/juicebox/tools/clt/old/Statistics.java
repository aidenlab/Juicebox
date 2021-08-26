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

package juicebox.tools.clt.old;

import juicebox.data.ChromosomeHandler;
import juicebox.data.HiCFileTools;
import juicebox.tools.clt.CommandLineParser;
import juicebox.tools.clt.JuiceboxCLT;
import juicebox.tools.utils.original.Chunk;
import juicebox.tools.utils.original.FragmentCalculation;
import juicebox.tools.utils.original.MTIndexHandler;
import juicebox.tools.utils.original.stats.LoneStatisticsWorker;
import juicebox.tools.utils.original.stats.ParallelStatistics;
import juicebox.tools.utils.original.stats.StatisticsContainer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class Statistics extends JuiceboxCLT {

    private String siteFile;
    private String ligationJunction = "none";
    private String inFile;
    private String mndIndexFile;
    private ChromosomeHandler localHandler = null;
    private final List<Chunk> mndChunks = new ArrayList<>();
    private final List<String> statsFiles = new ArrayList<>();
    private final List<Integer> mapqThresholds = new ArrayList<>();

    public Statistics() {
        super(getUsage());
    }

    public static String getUsage() {
        return " Usage: statistics [--ligation NNNN] [--mapqs mapq1,maqp2] [--mndindex mndindex.txt] [--threads numthreads]\n " +
                "                   <site_file> <stats_file> [stats_file_2] <infile> <genomeID>\n" +
                " --ligation: ligation junction\n" +
                " --mapqs: mapping quality threshold(s), do not consider reads < threshold\n" +
                " --mndindex: file of indices for merged nodups to read from\n" +
                " --threads: number of threads to be executed \n" +
                " <site file>: list of HindIII restriction sites, one line per chromosome\n" +
                " <stats file>: output file containing total reads, for library complexity\n" +
                " <infile>: file in intermediate format to calculate statistics on, can be stream\n" +
                " <genome ID>: file to create chromosome handler\n" +
                " [stats file 2]: output file containing total reads for second mapping quality threshold\n";
    }

    public void setMndIndex() {
        if (localHandler != null && mndIndexFile != null && mndIndexFile.length() > 1) {
            Map<Integer, String> chromosomePairIndexes = new ConcurrentHashMap<>();
            MTIndexHandler.populateChromosomePairIndexes(localHandler,
                    chromosomePairIndexes, new HashMap<>(),
                    new HashMap<>(), new HashMap<>());
            Map<Integer, List<Chunk>> mndIndex = MTIndexHandler.readMndIndex(mndIndexFile, chromosomePairIndexes);
            for (List<Chunk> values : mndIndex.values()) {
                mndChunks.addAll(values);
            }
        }
    }

    private FragmentCalculation readSiteFile(String siteFile) {
        //read in restriction site file and store as multidimensional array q
        if (!siteFile.contains("none")) {
            //if restriction enzyme exists, find the RE distance//
            return FragmentCalculation.readFragments(siteFile, localHandler, "Stats");
        }
        return null;
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
            tryToReadLocalHandler(args[5]);
        } else {//only one mapq value
            inFile = args[3];
            tryToReadLocalHandler(args[4]);
        }
        //check for flags, else use default values
        List<Integer> mapQT = parser.getMultipleMapQOptions();
        if (mapQT != null && (mapQT.size() == 1 || mapQT.size() == 2)) { //only one or two mapq values
            int mapqThreshold = mapQT.get(0) > 0 ? mapQT.get(0) : 1;
            mapqThresholds.add(mapqThreshold);

            if (statsFiles.size() == 2) {
                mapqThreshold = 30;
                if (mapQT.size() == 2) {
                    mapqThreshold = mapQT.get(1) > 0 ? mapQT.get(1) : 30;
                }
                mapqThresholds.add(mapqThreshold);
            }
        }
        else {
            mapqThresholds.add(1);
            if (statsFiles.size() == 2) {
                mapqThresholds.add(30);
            }
        }
        String ligJunc = parser.getLigationOption();
        if (ligJunc != null && ligJunc.length() > 1) {
            ligationJunction = ligJunc;
        }
        //multithreading flags
        updateNumberOfCPUThreads(parser, 1);
        mndIndexFile = parser.getMndIndexOption();
    }

    private void tryToReadLocalHandler(String genomeID) {
        if (genomeID.equalsIgnoreCase("na")
                || genomeID.equalsIgnoreCase("null")
                || genomeID.equalsIgnoreCase("none")) {
            localHandler = null;
        } else {
            localHandler = HiCFileTools.loadChromosomes(genomeID); //genomeID
        }
    }

    @Override
    public void run() {
        setMndIndex();
        FragmentCalculation fragmentCalculation = readSiteFile(siteFile);
        StatisticsContainer container;
        if (localHandler == null || mndChunks.size() < 2 || numCPUThreads == 1) {
            LoneStatisticsWorker runner = new LoneStatisticsWorker(siteFile, statsFiles, mapqThresholds,
                    ligationJunction, inFile, fragmentCalculation);
            runner.infileStatistics();
            container = runner.getResultsContainer();
        } else {
            container = new StatisticsContainer();
            ParallelStatistics pStats = new ParallelStatistics(numCPUThreads, container,
                    mndChunks, siteFile, statsFiles, mapqThresholds,
                    ligationJunction, inFile, localHandler, fragmentCalculation);
            pStats.launchThreads();
        }
        container.calculateConvergence(statsFiles.size());
        container.outputStatsFile(statsFiles);
        container.writeHistFile(statsFiles);
    }
}
