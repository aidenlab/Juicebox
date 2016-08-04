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

import jargs.gnu.CmdLineParser;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * Created by muhammadsaadshamim on 9/4/15.
 */
public class CommandLineParser extends CmdLineParser {

    // boolean
    private static Option diagonalsOption = null;
    private static Option helpOption = null;
    private static Option removeCacheMemoryOption = null;
    private static Option verboseOption = null;

    // String
    private static Option fragmentOption = null;
    private static Option tmpDirOption = null;
    private static Option statsOption = null;
    private static Option graphOption = null;

    // ints
    private static Option countThresholdOption = null;
    private static Option mapqOption = null;

    // sets of strings
    private static Option multipleChromosomesOption = null;
    private static Option resolutionOption = null;

    public CommandLineParser() {

        diagonalsOption = addBooleanOption('d', "diagonals");
        helpOption = addBooleanOption('h', "help");
        removeCacheMemoryOption = addBooleanOption('x', "remove memory cache");
        verboseOption = addBooleanOption('v', "verbose");

        fragmentOption = addStringOption('f', "restriction fragment site file");
        tmpDirOption = addStringOption('t', "tmpDir");
        statsOption = addStringOption('s', "statistics text file");
        graphOption = addStringOption('g', "graph text file");

        countThresholdOption = addIntegerOption('m', "minCountThreshold");
        mapqOption = addIntegerOption('q', "mapping quality threshold");

        multipleChromosomesOption = addStringOption('c', "chromosomes");
        resolutionOption = addStringOption('r', "resolutions");
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

    public boolean getVerboseOption() {
        return optionToBoolean(verboseOption);
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

    public int getCountThresholdOption() {
        return optionToInt(countThresholdOption);
    }

    public int getMapqThresholdOption() {
        return optionToInt(mapqOption);
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

    public Set<String> getResolutionOption() { return optionToStringSet(resolutionOption);}
}
