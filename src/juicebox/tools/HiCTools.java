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

package juicebox.tools;

import jargs.gnu.CmdLineParser;
import juicebox.tools.clt.JuiceboxCLT;
import org.broad.igv.Globals;

import java.io.IOException;
import java.lang.reflect.Constructor;
import java.util.*;


/**
 * @author Muhammad Shamim
 * @date 1/20/2015
 * <p/>
 * MSS - Comments
 * <p/>
 * I've tested the new CLT changes with AddGWNorm, AddNorm, BinToPairs, CalcKR, Dump,
 * and PairsToBin and verified that the outputs are identical to the previous ones.
 * <p/>
 * using python:
 * {
 * import filecmp
 * print filecmp.cmp('output1.hic', 'output2.hic') # byte by byte comparison of output files
 * }
 * <p/>
 * Have not tested BigWig, BPToFrag, FragToBed, Pre, or DB.
 * They should still work, as I haven't changed their internal code.
 * If some one could let me know where to find frag/bed files,
 * and how the genomeIDs match up with HiC files, I can test the remaining commands.
 * <p/>
 * I'm going to go ahead and commit the changes.
 * The old file will be renamed HiCToolsOld and I'll leave a TODO comment explaining to remove it after testing.
 * <p/>
 * Some of the commands used
 * addGWNorm /Users/muhammadsaadshamim/Desktop/testing/test2.hic 100000000
 * addNorm /Users/muhammadsaadshamim/Desktop/testing/test2.hic 100000000
 * binToPairs /Users/muhammadsaadshamim/Desktop/testing/a1.hic /Users/muhammadsaadshamim/Desktop/testing/b1.hic
 * calcKR /Users/muhammadsaadshamim/Desktop/testing/a1.hic
 * dump observed NONE /Users/muhammadsaadshamim/Desktop/testing/a1.hic 1 1 BP 1000000 /Users/muhammadsaadshamim/Desktop/testing/b1.hic
 * pairsToBin /Users/muhammadsaadshamim/Desktop/testing/a1.hic /Users/muhammadsaadshamim/Desktop/testing/b1.hic hg19
 */
public class HiCTools {

    private static void usage() {

        System.out.println("Juicebox Command Line Tools Usage:");
        for (int i = 0; i < nameToCommandLineTool.length; i += 3) {
            System.out.println("       juicebox "+ nameToCommandLineTool[i+2] );
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

    public static void main(String[] argv) throws IOException, CmdLineParser.UnknownOptionException, CmdLineParser.IllegalOptionValueException {

        Map<String, String> argToClass = new HashMap<String, String>();
        Map<String, String> argToUsage = new HashMap<String, String>();
        for (int i = 0; i < nameToCommandLineTool.length; i += 3) {
            argToClass.put(nameToCommandLineTool[i].toLowerCase(), nameToCommandLineTool[i + 1]);
            argToUsage.put(nameToCommandLineTool[i].toLowerCase(), nameToCommandLineTool[i + 2]);
        }

        Globals.setHeadless(true);

        CommandLineParser parser = new CommandLineParser();
        parser.parse(argv);
        String[] args = parser.getRemainingArgs();

        if (parser.getHelpOption()) {
            usage();
            System.exit(0);
        }

        String cmd = args[0].toLowerCase();

        try {
            if (argToClass.containsKey(cmd)) {
                Class c = Class.forName(argToClass.get(cmd));
                Constructor constructor = c.getConstructor();
                JuiceboxCLT instanceOfCLT = (JuiceboxCLT) constructor.newInstance();
                instanceOfCLT.setUsage(argToUsage.get(cmd));

                try {
                    instanceOfCLT.readArguments(args, parser);
                } catch (IOException e) {
                    instanceOfCLT.printUsage(); // error reading arguments, print specific usage help
                    System.exit(Integer.parseInt(e.getMessage()));
                }

                try {
                    instanceOfCLT.run();
                } catch (Exception e) {
                    System.out.println(e.getMessage());
                    e.printStackTrace();
                    System.exit(-7); // error running the code, these shouldn't occur (error checking should be added)
                }
            } else {
                usage();
                System.exit(2);
            }
        } catch (Exception e) {
            //System.out.println(e.getMessage());
            usage();
            e.printStackTrace();
            System.exit(2);
        }
    }

    public static class CommandLineParser extends CmdLineParser {
        private Option diagonalsOption = null;
        private Option chromosomeOption = null;
        private Option countThresholdOption = null;
        private Option helpOption = null;
        private Option fragmentOption = null;
        private Option tmpDirOption = null;
        private Option statsOption = null;
        private Option graphOption = null;
        private Option mapqOption = null;

        CommandLineParser() {
            diagonalsOption = addBooleanOption('d', "diagonals");
            chromosomeOption = addStringOption('c', "chromosomes");
            countThresholdOption = addIntegerOption('m', "minCountThreshold");
            fragmentOption = addStringOption('f', "restriction fragment site file");
            tmpDirOption = addStringOption('t', "tmpDir");
            helpOption = addBooleanOption('h', "help");
            statsOption = addStringOption('s', "statistics text file");
            graphOption = addStringOption('g', "graph text file");
            mapqOption = addIntegerOption('q', "mapping quality threshold");
        }

        public boolean getHelpOption() {
            Object opt = getOptionValue(helpOption);
            return opt != null && (Boolean) opt;
        }

        public boolean getDiagonalsOption() {
            Object opt = getOptionValue(diagonalsOption);
            return opt != null && (Boolean) opt;
        }

        public Set<String> getChromosomeOption() {
            Object opt = getOptionValue(chromosomeOption);
            return opt == null ? null : new HashSet<String>(Arrays.asList(opt.toString().split(",")));
        }

        public String getFragmentOption() {
            Object opt = getOptionValue(fragmentOption);
            return opt == null ? null : opt.toString();
        }

        public String getStatsOption() {
            Object opt = getOptionValue(statsOption);
            return opt == null ? null : opt.toString();
        }

        public String getGraphOption() {
            Object opt = getOptionValue(graphOption);
            return opt == null ? null : opt.toString();
        }

        public String getTmpdirOption() {
            Object opt = getOptionValue(tmpDirOption);
            return opt == null ? null : opt.toString();
        }

        public int getCountThresholdOption() {
            Object opt = getOptionValue(countThresholdOption);
            return opt == null ? 0 : ((Number) opt).intValue();
        }

        public int getMapqThresholdOption() {
            Object opt = getOptionValue(mapqOption);
            return opt == null ? 0 : ((Number) opt).intValue();
        }
    }
}
