/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2019 Broad Institute, Aiden Lab
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
 * Command Line Parser for original (Pre/Dump) calls. Created by muhammadsaadshamim on 9/4/15.
 */
public class CommandLineParser extends CmdLineParser {

    // boolean
    private static Option diagonalsOption = null;
    private static Option helpOption = null;
    private static Option removeCacheMemoryOption = null;
    private static Option verboseOption = null;
    private static Option noNormOption = null;
    private static Option allPearsonsOption = null;
    private static Option versionOption = null;

    // String
    private static Option fragmentOption = null;
    private static Option tmpDirOption = null;
    private static Option statsOption = null;
    private static Option graphOption = null;
    private static Option expectedVectorOption = null;

    // ints
    private static Option countThresholdOption = null;
    private static Option mapqOption = null;
    private static Option noFragNormOption = null;
    private static Option genomeWideOption = null;
    private static Option hicFileScalingOption = null;

    // sets of strings
    private static Option multipleChromosomesOption = null;
    private static Option resolutionOption = null;
    private static Option randomizePositionMapsOption = null;


    //filter option based on directionality
    private static Option alignmentFilterOption = null;

    private static Option randomizePositionOption = null;
    private static Option randomSeedOption = null;

    public CommandLineParser() {

        // available
        // beijklouy

        // used
        // d h x v n p F V f t s g m q w c r z a

        diagonalsOption = addBooleanOption('d', "diagonals");
        helpOption = addBooleanOption('h', "help");
        removeCacheMemoryOption = addBooleanOption('x', "remove_memory_cache");
        verboseOption = addBooleanOption('v', "verbose");
        noNormOption = addBooleanOption('n', "no_normalization");
        allPearsonsOption = addBooleanOption('p', "pearsons_all_resolutions");
        noFragNormOption = addBooleanOption('F', "no_fragment_normalization");
        versionOption = addBooleanOption('V', "version");

        fragmentOption = addStringOption('f', "restriction_fragment_site_file");
        tmpDirOption = addStringOption('t', "tmpDir");
        statsOption = addStringOption('s', "statistics");
        graphOption = addStringOption('g', "graphs");

        countThresholdOption = addIntegerOption('m', "min_count");
        mapqOption = addIntegerOption('q', "mapq");

        genomeWideOption = addIntegerOption('w', "genome_wide");

        multipleChromosomesOption = addStringOption('c', "chromosomes");
        resolutionOption = addStringOption('r', "resolutions");

        expectedVectorOption = addStringOption('e', "expected_vector_file");
        hicFileScalingOption = addDoubleOption('z', "scale");

        alignmentFilterOption = addIntegerOption('a', "alignment");
        randomizePositionOption = addBooleanOption("randomize_position");
        randomSeedOption = addLongOption("random_seed");
        randomizePositionMapsOption = addStringOption("randomize_pos_maps");

    }


    /**
     * boolean flags
     */
    private boolean optionToBoolean(Option option) {
        Object opt = getOptionValue(option);
        return opt != null && (Boolean) opt;
    }

    public boolean getHelpOption() { return optionToBoolean(helpOption);}

    public boolean getDiagonalsOption() {
        return optionToBoolean(diagonalsOption);
    }

    public boolean useCacheMemory() {
        return optionToBoolean(removeCacheMemoryOption);
    }

    public boolean getVerboseOption() {
        return optionToBoolean(verboseOption);
    }

    public boolean getNoNormOption() { return optionToBoolean(noNormOption); }

    public boolean getAllPearsonsOption() {return optionToBoolean(allPearsonsOption);}

    public boolean getNoFragNormOption() { return optionToBoolean(noFragNormOption); }

    public boolean getVersionOption() { return optionToBoolean(versionOption); }

    public boolean getRandomizePositionsOption() {
        return optionToBoolean(randomizePositionOption);
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

    public String getExpectedVectorOption() {
        return optionToString(expectedVectorOption);
    }

    public Alignment getAlignmentOption() {
        int alignmentInt = optionToInt(alignmentFilterOption);

        if (alignmentInt == 0) {
            return null;
        }
        if (alignmentInt == 1) {
            return Alignment.INNER;
        } else if (alignmentInt == 2) {
            return Alignment.OUTER;
        } else if (alignmentInt == 3) {
            return Alignment.LL;
        } else if (alignmentInt == 4) {
            return Alignment.RR;
        } else {
            throw new IllegalArgumentException(String.format("alignment option %d not supported", alignmentInt));
        }
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

    public int getMapqThresholdOption() { return optionToInt(mapqOption); }

    public int getGenomeWideOption() { return optionToInt(genomeWideOption); }

    private long optionToLong(Option option) {
        Object opt = getOptionValue(option);
        return opt == null ? 0 : ((Number) opt).longValue();
    }

    public long getRandomPositionSeedOption() {
        return optionToLong(randomSeedOption);
    }

    public enum Alignment {
        INNER, OUTER, LL, RR;
    }

    /**
     * double flags
     */
    private double optionToDouble(Option option) {
        Object opt = getOptionValue(option);
        return opt == null ? 0 : ((Number) opt).doubleValue();
    }

    public double getScalingOption() {
        return optionToDouble(hicFileScalingOption);
    }

    /**
     * String Set flags
     */
    private Set<String> optionToStringSet(Option option) {
        Object opt = getOptionValue(option);
        return opt == null ? null : new HashSet<>(Arrays.asList(opt.toString().split(",")));
    }

    public Set<String> getChromosomeOption() {
        return optionToStringSet(multipleChromosomesOption);
    }

    public Set<String> getResolutionOption() { return optionToStringSet(resolutionOption);}

    public Set<String> getRandomizePositionMaps() {return optionToStringSet(randomizePositionMapsOption);}
}
