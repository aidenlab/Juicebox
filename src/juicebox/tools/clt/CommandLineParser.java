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
import juicebox.windowui.NormalizationHandler;
import juicebox.windowui.NormalizationType;

import java.util.*;

/**
 * Command Line Parser for original (Pre/Dump) calls. Created by muhammadsaadshamim on 9/4/15.
 */
public class CommandLineParser extends CmdLineParser {

    // available
    // bijlou
    // used
    // d h x v n p k F V f t s g m q w c r z a y

    // universal
    protected final Option verboseOption = addBooleanOption('v', "verbose");
    protected final Option helpOption = addBooleanOption('h', "help");
    protected final Option versionOption = addBooleanOption('V', "version");

    // boolean
    private final Option diagonalsOption = addBooleanOption('d', "diagonals");
    private final Option removeCacheMemoryOption = addBooleanOption('x', "remove-memory-cache");
    private final Option noNormOption = addBooleanOption('n', "no_normalization");
    private final Option allPearsonsOption = addBooleanOption('p', "pearsons-all-resolutions");
    private final Option noFragNormOption = addBooleanOption('F', "no_fragment-normalization");
    private final Option randomizePositionOption = addBooleanOption("randomize_position");

    // String
    private final Option fragmentOption = addStringOption('f', "restriction-fragment-site-file");
    private final Option tmpDirOption = addStringOption('t', "tmpdir");
    private final Option statsOption = addStringOption('s', "statistics");
    private final Option graphOption = addStringOption('g', "graphs");
    private final Option genomeIDOption = addStringOption('y', "genomeid");
    private final Option expectedVectorOption = addStringOption('e', "expected-vector-file");
    protected final Option normalizationTypeOption = addStringOption('k', "normalization");

    // ints
    private final Option countThresholdOption = addIntegerOption('m', "min-count");
    private final Option mapqOption = addIntegerOption('q', "mapq");
    private final Option genomeWideOption = addIntegerOption('w', "genomewide");
    private final Option alignmentFilterOption = addIntegerOption('a', "alignment");

    // sets of strings
    private final Option multipleChromosomesOption = addStringOption('c', "chromosomes");
    private final Option resolutionOption = addStringOption('r', "resolutions");
    private final Option randomizePositionMapsOption = addStringOption("frag-site-maps");


    //filter optrectionalion based on diity
    private final Option hicFileScalingOption = addDoubleOption('z', "scale");
    private final Option randomSeedOption = addLongOption("random-seed");


    public CommandLineParser() {
    }


    /**
     * boolean flags
     */
    protected boolean optionToBoolean(Option option) {
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
    protected String optionToString(Option option) {
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

    public String getGenomeOption() { return optionToString(genomeIDOption); }

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
    protected int optionToInt(Option option) {
        Object opt = getOptionValue(option);
        return opt == null ? 0 : ((Number) opt).intValue();
    }

    public int getCountThresholdOption() {
        return optionToInt(countThresholdOption);
    }

    public int getMapqThresholdOption() { return optionToInt(mapqOption); }

    public int getGenomeWideOption() { return optionToInt(genomeWideOption); }

    protected long optionToLong(Option option) {
        Object opt = getOptionValue(option);
        return opt == null ? 0 : ((Number) opt).longValue();
    }

    public long getRandomPositionSeedOption() {
        return optionToLong(randomSeedOption);
    }

    public enum Alignment {
        INNER, OUTER, LL, RR
    }

    /**
     * double flags
     */
    protected double optionToDouble(Option option) {
        Object opt = getOptionValue(option);
        return opt == null ? 0 : ((Number) opt).doubleValue();
    }

    public double getScalingOption() {
        return optionToDouble(hicFileScalingOption);
    }

    /**
     * String Set flags
     */
    protected Set<String> optionToStringSet(Option option) {
        Object opt = getOptionValue(option);
        return opt == null ? null : new HashSet<>(Arrays.asList(opt.toString().split(",")));
    }

    protected List<String> optionToStringList(Option option) {
        Object opt = getOptionValue(option);
        return opt == null ? null : new ArrayList<>(Arrays.asList(opt.toString().split(",")));
    }

    public Set<String> getChromosomeSetOption() {
        return optionToStringSet(multipleChromosomesOption);
    }

    public Set<String> getResolutionOption() { return optionToStringSet(resolutionOption);}

    public Set<String> getRandomizePositionMaps() {return optionToStringSet(randomizePositionMapsOption);}

    public List<NormalizationType> getAllNormalizationTypesOption() {
        NormalizationHandler normalizationHandler = new NormalizationHandler();
        List<String> normStrings = optionToStringList(normalizationTypeOption);
        if (normStrings == null || normStrings.size() < 1) {
            return normalizationHandler.getDefaultSetForHiCFileBuilding();
        }

        List<NormalizationType> normalizationTypes = new ArrayList<>();
        for (String normString : normStrings) {
            normalizationTypes.add(retrieveNormalization(normString, normalizationHandler));
        }

        return normalizationTypes;
    }

    protected NormalizationType retrieveNormalization(String norm, NormalizationHandler normalizationHandler) {
        if (norm == null || norm.length() < 1)
            return null;

        try {
            return normalizationHandler.getNormTypeFromString(norm);
        } catch (IllegalArgumentException error) {
            System.err.println("Normalization must be one of \"NONE\", \"VC\", \"VC_SQRT\", \"KR\", \"GW_KR\", \"GW_VC\", \"INTER_KR\", or \"INTER_VC\".");
            System.exit(7);
        }
        return null;
    }
}
