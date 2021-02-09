/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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

package juicebox.tools.utils;

import juicebox.data.ChromosomeHandler;
import juicebox.data.basics.Chromosome;
import juicebox.tools.clt.CommandLineParser;
import juicebox.tools.clt.JuiceboxCLT;
import juicebox.tools.clt.old.Dump;
import org.broad.igv.Globals;

import java.util.Random;

/**
 * Created by Neva Durand on 8/4/16 for benchmark testing for DCIC.
 */

public class Benchmark extends JuiceboxCLT {

    private final int NUM_QUERIES = 1000;
    // Query 10,000 times at 256x256 and 2048x2048
    private int QUERY_SIZE = 256;
    private Dump dump;
    
    public Benchmark() {
        super(getUsage());
    }

    private static String getUsage() {
        return "benchmark <hicFile> <norm>";
    }

    @Override
    public void readArguments(String[] argv, CommandLineParser parser) {
        Globals.setHeadless(true);


        if (argv.length != 3) {
            printUsageAndExit();
        }

        dump = new Dump();

        // dump will read in the index of the .hic file and output the observed matrix with no normalization
        // change "NONE" to "KR" or "VC" for different normalizations
        // change outputfile if needed
        // the other values are dummy and will be reset
        String[] args = {"dump", "observed", argv[2], argv[1], "X", "X", "BP", "1000000", "/Users/nchernia/Downloads/output2.txt"};
        dump.readArguments(args, parser);

    }


    @Override
    public void run() {

        // will use to make sure we're not off the end of the chromosome
        ChromosomeHandler handler = dump.getChromosomeHandler();

        Random random = new Random();

        // chromosomes in this dataset, so we query them
        String[] chrs = new String[handler.size() - 1];
        int ind=0;

        for (Chromosome chr : handler.getChromosomeArray()) {
            if (!chr.getName().equalsIgnoreCase("All")) chrs[ind++] = chr.getName();
        }

        // BP bin sizes in this dataset
        int[] bpBinSizes = dump.getBpBinSizes();


        long sum=0;
        for (int i=0; i<NUM_QUERIES; i++) {
            // Randomly choose chromosome and resolution to query
            String chr1 = chrs[random.nextInt(chrs.length)];
            int binSize = bpBinSizes[random.nextInt(bpBinSizes.length)];
    
            long end1 = random.nextInt((int) handler.getChromosomeFromName(chr1).getLength()); // endpoint between 0 and end of chromosome
            long start1 = end1 - binSize * QUERY_SIZE; // QUERY_SIZE number of bins earlier
            if (start1 < 0) start1 = 0;
    
            dump.setQuery(chr1 + ":" + start1 + ":" + end1, chr1 + ":" + start1 + ":" + end1, binSize);
            long currentTime = System.currentTimeMillis();
            dump.run();
            long totalTime = System.currentTimeMillis() - currentTime;
            sum += totalTime;
        }
        System.err.println("Average time to query " + QUERY_SIZE + "x" + QUERY_SIZE +": " + sum/NUM_QUERIES + " milliseconds");

        QUERY_SIZE=2048;
        sum=0;
        for (int i=0; i<NUM_QUERIES; i++) {
            // Randomly choose chromosome and resolution to query
            String chr1 = chrs[random.nextInt(chrs.length)];
            int binSize = bpBinSizes[random.nextInt(bpBinSizes.length)];
    
            long end1 = random.nextInt((int) handler.getChromosomeFromName(chr1).getLength()); // endpoint between 0 and end of chromosome
            long start1 = end1 - binSize * QUERY_SIZE; // QUERY_SIZE number of bins earlier
            if (start1 < 0) start1 = 0;
    
            dump.setQuery(chr1 + ":" + start1 + ":" + end1, chr1 + ":" + start1 + ":" + end1, binSize);
            long currentTime = System.currentTimeMillis();
            dump.run();
            long totalTime = System.currentTimeMillis() - currentTime;
            sum += totalTime;
        }
        System.err.println("Average time to query " + QUERY_SIZE + "x" + QUERY_SIZE +": " + sum/NUM_QUERIES + " milliseconds");
    }
}
