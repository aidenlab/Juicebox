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
import java.util.*;


/**
 * Command line tool handling through factory model
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
                    System.exit(-7); // error running the code, these shouldn't occur (error checking should be added)
                }
        } catch (Exception e) {
            CLTFactory.usage();
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

        private Option resolutionOption = null;
        private Option multipleResolutionsOption = null;

        // APA options
        private Option apaMinValOption = null;
        private Option apaMaxValOption = null;
        private Option apaWindowOption = null;
        private Set<Integer> multipleResolutionOptions;


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

            resolutionOption = addIntegerOption('r', "resolution");
            multipleResolutionsOption = addStringOption('r', "multiple resolutions separated by ','");

            //apa <-m minval> <-x maxval> <-w window>  <-r resolution>
            apaMinValOption = addDoubleOption('n', "minimum value");
            apaMaxValOption = addDoubleOption('x', "maximum value");
            apaWindowOption = addIntegerOption('w', "window");

        }

        public Number[] getAPAOptions(){
            Number[] apaFlagValues = new Number[4];

            Object[] apaOptions = {getOptionValue(apaMinValOption), getOptionValue(apaMaxValOption),
                    getOptionValue(apaWindowOption), getOptionValue(resolutionOption)};

            for(int i = 0; i < apaOptions.length; i++) {
                if (apaOptions[i] != null)
                    apaFlagValues[i] = ((Number) apaOptions[i]);
                else
                    apaFlagValues[i] = null;
            }
            return apaFlagValues;
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

        public int getResolutionOption() {
            Object opt = getOptionValue(resolutionOption);
            return opt == null ? 0 : ((Number) opt).intValue();
        }

        public Set<String> getMultipleResolutionOptions() {
            Object opt = getOptionValue(multipleResolutionsOption);
            return opt == null ? null : new HashSet<String>(Arrays.asList(opt.toString().split(",")));
        }
    }
}
