/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2016 Broad Institute, Aiden Lab
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

package juicebox.tools.clt;

import juicebox.tools.clt.juicer.*;
import juicebox.tools.clt.old.*;

/**
 * Factory for command line tools to call different functions
 *
 * @author Muhammad Shamim
 * @since 1/30/2015
 */
public class CLTFactory {

    // Commenting some out because we're not going to release all these when we release CLT
    private final static String[] nameToCommandLineTool = {
            //        "addGWNorm",    "addGWNorm <input_HiC_file> <min resolution>",
            //        "addNorm",      "addNorm <input_HiC_file> [0 for no frag, 1 for no single frag]",
            //        "bigWig",       "bigWig <bigWig path or URL> <window size in bp> [chr] [start base] [end base]",
            //        "binToPairs",   "binToPairs <input_HiC_file> <output_HiC_file>",
            //        "bpToFrag",     "bpToFrag <fragmentFile> <inputBedFile> <outputFile>",
            //        "calcKR",       "calcKR <input_HiC_file>",
            "dump", "dump <observed/oe/pearson/norm> <NONE/VC/VC_SQRT/KR> <hicFile(s)> <chr1> <chr2> <BP/FRAG> <binsize>",
            //        "fragmentToBed","fragmentToBed <fragmentFile>",
            //        "pairsToBin",   "pairsToBin <input_HiC_file> <output_HiC_file> <genomeID>",
            //        "db",           "db <frag|annot|update> [items]",
            "pre", "pre <options> <infile> <outfile> <genomeID>",
            "apa", "apa <HiC file(s)> <PeaksFile> <SaveFolder>",
            "arrowhead", "arrowhead <HiC file(s)> <outfile>",
            "hiccups", "hiccups <HiC file(s)> <finalLoopsList>",
            "hiccupsdiff", "hiccupsdiff <HiC file1> <HiC file2> <peak list1> <peak list2> <output directory>"
    };

    public static void generalUsage() {

        System.out.println("Juicebox Command Line Tools Usage:");
        for (int i = 0; i < nameToCommandLineTool.length; i += 2) {
            System.out.println("       juicebox " + nameToCommandLineTool[i + 1]);
        }
        System.out.println("Type juicebox <command name> for further usage instructions");

    }

    public static JuiceboxCLT getCLTCommand(String cmd) {

        cmd = cmd.toLowerCase();

        if (cmd.equals("pre")) {
            return new PreProcessing();
        } else if (cmd.equals("dump")) {
            return new Dump();
        } else if (cmd.equals("addGWNorm".toLowerCase())) {
            return new AddGWNorm();
        } else if (cmd.equals("addNorm".toLowerCase())) {
            return new AddNorm();
        } else if (cmd.equals("apa")) {
            return new APA();
        } else if (cmd.equals("compare")) {
            return new CompareLists();
        } else if (cmd.equals("arrowhead")) {
            return new Arrowhead();
        } else if (cmd.equals("bigWig".toLowerCase())) {
            return new BigWig();
        } else if (cmd.equals("binToPairs".toLowerCase())) {
            return new BinToPairs();
        } else if (cmd.equals("bpToFrag".toLowerCase())) {
            return new BPToFragment();
        } else if (cmd.equals("calcKR".toLowerCase())) {
            return new CalcKR();
        } else if (cmd.equals("fragmentToBed".toLowerCase())) {
            return new FragmentToBed();
        } else if (cmd.equals("hiccups")) {
            return new HiCCUPS();
        } else if (cmd.equals("loop_domains")) {
            return new LoopDomains();
        } else if (cmd.equals("motifs")) {
            return new MotifFinder();
        } else if (cmd.equals("pairsToBin".toLowerCase())) {
            return new PairsToBin();
        } else if (cmd.equals("db")) {
            return new SQLDatabase();
        } else if (cmd.equals("hiccupsdiff")) {
            return new HiCCUPSDiff();
        } else if (cmd.equals("ab_compdiff")) {
            return new ABCompartmentsDiff();
        }


        return null;
    }
}
