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
import juicebox.tools.clt.CLTFactory;
import juicebox.tools.clt.JuiceboxCLT;
import org.broad.igv.Globals;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;


/**
 * Command line tool handling through factory model
 *
 * @author Muhammad Shamim
 * @date 1/20/2015
 */
public class HiCTools {

    public static void main(String[] argv) throws IOException, CmdLineParser.UnknownOptionException, CmdLineParser.IllegalOptionValueException {


        Globals.setHeadless(true);

        CommandLineParser parser = new CommandLineParser();
        parser.parse(argv);
        String[] args = parser.getRemainingArgs();

        if (parser.getHelpOption()) {
            CLTFactory.usage();
            System.exit(0);
        }

        try {
            String cmd = args[0].toLowerCase();
            JuiceboxCLT instanceOfCLT = CLTFactory.getCLTCommand(cmd);

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
                System.exit(-7);
                // error running the code, these shouldn't occur i.e. error checking
                // should be added within each CLT for better error tracing
            }
        } catch (Exception e) {
            CLTFactory.usage();
            System.exit(2);
        }
    }

    public static class CommandLineParser extends CmdLineParser {

        // boolean
        private Option diagonalsOption = null;
        private Option helpOption = null;
        private Option removeCacheMemoryOption = null;

        // String
        private Option fragmentOption = null;
        private Option tmpDirOption = null;
        private Option statsOption = null;
        private Option graphOption = null;

        // ints
        private Option apaWindowOption = null;
        private Option countThresholdOption = null;
        private Option mapqOption = null;
        private Option matrixSizeOption = null;

        // doubles
        private Option apaMinValOption = null;
        private Option apaMaxValOption = null;

        // sets of strings
        private Option multipleChromosomesOption = null;
        private Option multipleResolutionsOption = null;

        CommandLineParser() {

            diagonalsOption = addBooleanOption('d', "diagonals");
            helpOption = addBooleanOption('h', "help");
            removeCacheMemoryOption = addBooleanOption('x', "remove memory cache");

            fragmentOption = addStringOption('f', "restriction fragment site file");
            tmpDirOption = addStringOption('t', "tmpDir");
            statsOption = addStringOption('s', "statistics text file");
            graphOption = addStringOption('g', "graph text file");

            apaWindowOption = addIntegerOption('w', "window");
            countThresholdOption = addIntegerOption('m', "minCountThreshold");
            mapqOption = addIntegerOption('q', "mapping quality threshold");
            matrixSizeOption = addIntegerOption('m', "minCountThreshold (hiccups)");

            apaMinValOption = addDoubleOption('n', "minimum value");
            apaMaxValOption = addDoubleOption('x', "maximum value");

            multipleChromosomesOption = addStringOption('c', "chromosomes");
            multipleResolutionsOption = addStringOption('r', "multiple resolutions separated by ','");
        }

        /**
         * boolean flags
         */
        private boolean optionToBoolean(Option option) {
            Object opt = getOptionValue(option);
            return opt != null && (Boolean) opt;
        }

        public boolean getHelpOption() {
            return optionToBoolean(helpOption);
        }

        public boolean getDiagonalsOption() {
            return optionToBoolean(diagonalsOption);
        }

        public boolean useCacheMemory() {
            return optionToBoolean(removeCacheMemoryOption);
        }

        /**
         * String flags
         */
        private String optionToString(Option option) {
            Object opt = getOptionValue(option);
            return opt == null ? null : opt.toString();
        }

        public String getFragmentOption() {
            return optionToString(fragmentOption);
        }

        public String getStatsOption() {
            return optionToString(statsOption);
        }

        public String getGraphOption() {
            return optionToString(graphOption);
        }

        public String getTmpdirOption() {
            return optionToString(tmpDirOption);
        }

        /**
         * int flags
         */
        private int optionToInt(Option option) {
            Object opt = getOptionValue(option);
            return opt == null ? 0 : ((Number) opt).intValue();
        }

        public int getAPAWindowSizeOption() {
            return optionToInt(apaWindowOption);
        }

        public int getCountThresholdOption() {
            return optionToInt(countThresholdOption);
        }

        public int getMapqThresholdOption() {
            return optionToInt(mapqOption);
        }

        public int getMatrixSizeOption() {
            return optionToInt(matrixSizeOption);
        }

        /**
         * double flags
         */
        private double optionToDouble(Option option) {
            Object opt = getOptionValue(option);
            return opt == null ? 0 : ((Number) opt).doubleValue();
        }

        public double getAPAMinVal() {
            return optionToInt(apaMinValOption);
        }

        public double getAPAMaxVal() {
            return optionToInt(apaMaxValOption);
        }

        /**
         * String Set flags
         */
        private Set<String> optionToStringSet(Option option) {
            Object opt = getOptionValue(option);
            return opt == null ? null : new HashSet<String>(Arrays.asList(opt.toString().split(",")));
        }

        public Set<String> getChromosomeOption() {
            return optionToStringSet(multipleChromosomesOption);
        }

        public Set<String> getMultipleResolutionOptions() {
            return optionToStringSet(multipleResolutionsOption);
        }
    }
}
