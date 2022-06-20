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
    // blou
    // used
    // d h x v n p k F V f t s g m q w c r z a y j i

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
    private final Option throwIntraFragOption = addBooleanOption("skip-intra-frag");
    private final Option useMinRAM = addBooleanOption("conserve-ram");
    private final Option checkMemory = addBooleanOption("check-ram-usage");
    private final Option fromHIC = addBooleanOption("from-hic");

    // String
    private final Option fragmentOption = addStringOption('f', "restriction-fragment-site-file");
    private final Option tmpDirOption = addStringOption('t', "tmpdir");
    private final Option statsOption = addStringOption('s', "statistics");
    private final Option graphOption = addStringOption('g', "graphs");
    private final Option genomeIDOption = addStringOption('y', "genomeid");
    private final Option expectedVectorOption = addStringOption('e', "expected-vector-file");
    protected final Option normalizationTypeOption = addStringOption('k', "normalization");
    private final Option mndIndexOption = addStringOption('i', "mndindex");
    private final Option ligationOption = addStringOption("ligation");
    private final Option shellOption = addStringOption("shell");

    // ints
    private final Option blockCapacityOption = addIntegerOption("block-capacity");
    private final Option countThresholdOption = addIntegerOption('m', "min-count");
    private final Option mapqOption = addIntegerOption('q', "mapq");
    private final Option genomeWideOption = addIntegerOption('w', "genomewide");
    private final Option alignmentFilterOption = addIntegerOption('a', "alignment");
    private final Option threadNumOption = addIntegerOption('j', "threads");
    private final Option matrixThreadNumOption = addIntegerOption("mthreads");
    private final Option v9DepthBaseOption = addIntegerOption("v9-depth-base");

    // double
    private final Option subsampleOption = addDoubleOption("subsample");

    // sets of strings
    private final Option multipleChromosomesOption = addStringOption('c', "chromosomes");
    private final Option resolutionOption = addStringOption('r', "resolutions");
    private final Option randomizePositionMapsOption = addStringOption("frag-site-maps");

    //set of ints
    private final Option multipleMapQOption = addStringOption("mapqs");

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

    public boolean getAllPearsonsOption() {
        return optionToBoolean(allPearsonsOption);
    }

    public boolean getNoFragNormOption() {
        return optionToBoolean(noFragNormOption);
    }

    public boolean getVersionOption() {
        return optionToBoolean(versionOption);
    }

    public boolean getRandomizePositionsOption() {
        return optionToBoolean(randomizePositionOption);
    }

    public boolean getThrowIntraFragOption() {
        return optionToBoolean(throwIntraFragOption);
    }

    public boolean getDontPutAllContactsIntoRAM() {
        return optionToBoolean(useMinRAM);
    }

    public boolean shouldCheckRAMUsage() {
        return optionToBoolean(checkMemory);
    }

    public boolean getFromHICOption() { return optionToBoolean(fromHIC); }

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

    public String getMndIndexOption() {
        return optionToString(mndIndexOption);
    }

    public String getLigationOption() {
        return optionToString(ligationOption);
    }

    public String getShellOption() {
        return optionToString(shellOption);
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
        } else if (alignmentInt == 5) {
            return Alignment.TANDEM;
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

    public int getBlockCapacityOption() {
        return optionToInt(blockCapacityOption);
    }

    public int getCountThresholdOption() {
        return optionToInt(countThresholdOption);
    }

    public int getMapqThresholdOption() {
        return optionToInt(mapqOption);
    }

    public int getGenomeWideOption() {
        return optionToInt(genomeWideOption);
    }

    protected long optionToLong(Option option) {
        Object opt = getOptionValue(option);
        return opt == null ? 0 : ((Number) opt).longValue();
    }

    public long getRandomPositionSeedOption() {
        return optionToLong(randomSeedOption);
    }

    public enum Alignment {INNER, OUTER, LL, RR, TANDEM}

    public int getNumThreads() {
        return optionToInt(threadNumOption);
    }

    public int getNumMatrixOperationThreads() {
        return optionToInt(matrixThreadNumOption);
    }

    public int getV9DepthBase() {
        return optionToInt(v9DepthBaseOption);
    }

    /**
     * double flags
     */
    protected double optionToDouble(Option option) {
        Object opt = getOptionValue(option);
        return opt == null ? -1 : ((Number) opt).doubleValue();
    }

    public double getScalingOption() {
        double opt = optionToDouble(hicFileScalingOption);
        if (opt > -1) return opt;
        return 1;
    }

    public double getSubsampleOption() { return optionToDouble(subsampleOption); }

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

    public List<String> getResolutionOption() {
        return optionToStringList(resolutionOption);
    }

    public Set<String> getRandomizePositionMaps() {return optionToStringSet(randomizePositionMapsOption);}

    /**
     * Int Set flags
     */
    protected List<Integer> optionToIntList(Option option) {
        Object opt = getOptionValue(option);
        if(opt == null){
            return null;
        }
        String[] temp = opt.toString().split(",");
        List<Integer> options = new ArrayList<>();
        for(String s : temp){
            options.add(Integer.parseInt(s));
        }
        return options;
    }

    public List<Integer> getMultipleMapQOptions() {return optionToIntList(multipleMapQOption);}

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
