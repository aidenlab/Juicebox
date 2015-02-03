/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2015 Broad Institute, Aiden Lab
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

/**
 * Created by muhammadsaadshamim on 1/30/15.
 */
public class CLTFactory {

    private final static String[] nameToCommandLineTool = {
            "addGWNorm",    "juicebox.tools.clt.AddGWNorm",         "addGWNorm <input_HiC_file> <min resolution>",
            "addNorm",      "juicebox.tools.clt.AddNorm",           "addNorm <input_HiC_file> [0 for no frag, 1 for no single frag]",
            "apa",          "juicebox.tools.clt.APA",               "apa <minval maxval window  resolution> CountsFolder PeaksFile/PeaksFolder SaveFolder SavePrefix",
            "arrowhead",    "juicebox.tools.Arrowhead",             "arrowhead <input_HiC_file> <resolution>",
            "bigWig",       "juicebox.tools.clt.BigWig",            "bigWig <bigWig path or URL> <window size in bp> [chr] [start base] [end base]",
            "binToPairs",   "juicebox.tools.clt.BinToPairs",        "binToPairs <input_HiC_file> <output_HiC_file>",
            "bpToFrag",     "juicebox.tools.clt.BPToFragment",      "bpToFrag <fragmentFile> <inputBedFile> <outputFile>",
            "calcKR",       "juicebox.tools.clt.CalcKR",            "calcKR <input_HiC_file>",
            "dump",         "juicebox.tools.clt.Dump",              "dump <observed/oe/pearson/norm/expected/eigenvector> <NONE/VC/VC_SQRT/KR/GW_VC/GW_KR/INTER_VC/INTER_KR> <hicFile(s)> <chr1> <chr2> <BP/FRAG> <binsize>",
            "fragmentToBed","juicebox.tools.clt.FragmentToBed",     "fragmentToBed <fragmentFile>",
            "hiccups",      "juicebox.tools.clt.HiCCUPS",           "",
            "pairsToBin",   "juicebox.tools.clt.PairsToBin",        "pairsToBin <input_HiC_file> <output_HiC_file> <genomeID>",
            "db",           "juicebox.tools.clt.SQLDatabase",       "db <frag|annot|update> [items]",
            "pre",          "juicebox.tools.clt.PreProcessing",     "pre <options> <infile> <outfile> <genomeID>"
    };

    public static void usage() {

        System.out.println("Juicebox Command Line Tools Usage:");
        for (int i = 0; i < nameToCommandLineTool.length; i += 3) {
            System.out.println("       juicebox " + nameToCommandLineTool[i + 2]);
        }

        /*
        System.out.println("Usage: juicebox db <frag|annot|update> [items]");
        System.out.println("       juicebox binToPairs <infile> <outfile>");
        System.out.println("       juicebox dump <observed/oe/pearson/norm/expected/eigenvector> <NONE/VC/VC_SQRT/KR/GW_VC/GW_KR/INTER_VC/INTER_KR> <hicFile(s)> <chr1> <chr2> <BP/FRAG> <binsize> [binary outfile]");
        System.out.println("       juicebox addNorm <hicFile> [0 for no frag, 1 for no single frag]");
        System.out.println("       juicebox addGWNorm <hicFile> <min resolution>");
        System.out.println("       juicebox bigWig <bigWig path or URL> <window size in bp> [chr] [start base] [end base]");
        System.out.println("       juicebox calcKR <infile>");
        System.out.println("       juicebox arrowhead <hicfile> <resolution>");
        System.out.println("       juicebox pre <options> <infile> <outfile> <genomeID>");
        */

        System.out.println("   <options>: -d only calculate intra chromosome (diagonal) [false]");
        System.out.println("           : -f <restriction site file> calculate fragment map");
        System.out.println("           : -m <int> only write cells with count above threshold m [0]");
        System.out.println("           : -q <int> filter by MAPQ score greater than or equal to q");
        System.out.println("           : -c <chromosome ID> only calculate map on specific chromosome");
        System.out.println("           : -s <statsFile> include text statistics file");
        System.out.println("           : -g <graphFile> include graph file");
        System.out.println("           : -t, --tmpdir <temporary file directory>");
        System.out.println("           : -h print help");
    }

    public static JuiceboxCLT getCLTCommand(String cmd) {

        cmd = cmd.toLowerCase();

        if(cmd.equals("addGWNorm")){
            return new AddGWNorm();
        }
        else if(cmd.equals("addNorm".toLowerCase())){
            return new AddNorm();
        }
        else if(cmd.equals("apa")){
            return new APA();
        }
        else if(cmd.equals("arrowhead")){
            return new Arrowhead();
        }
        else if(cmd.equals("bigWig".toLowerCase())){
            return new BigWig();
        }else if(cmd.equals("binToPairs".toLowerCase())){
            return new BinToPairs();
        }
        else if(cmd.equals("bpToFrag".toLowerCase())){
            return new BPToFragment();
        }
        else if(cmd.equals("calcKR".toLowerCase())){
            return new CalcKR();
        }
        else if(cmd.equals("dump")){
            return new Dump();
        }
        else if(cmd.equals("fragmentToBed".toLowerCase())){
            return new FragmentToBed();
        }
        else if(cmd.equals("hiccups")){
            return new HiCCUPS();
        }
        else if(cmd.equals("pairsToBin".toLowerCase())){
            return new PairsToBin();
        }
        else if(cmd.equals("db")){
            return new SQLDatabase();
        }
        else if(cmd.equals("pre")){
            return new PreProcessing();
        }

        return null;
    }
}
