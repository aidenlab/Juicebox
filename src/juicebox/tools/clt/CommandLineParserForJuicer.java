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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Command Line Parser for Juicer commands (hiccups, arrowhead, apa)
 * @author Muhammad Shamim
 */
public class CommandLineParserForJuicer extends CmdLineParser {

    // General
    private static Option matrixSizeOption = null;
    private static Option multipleChromosomesOption = null;
    private static Option multipleResolutionsOption = null;
    private static Option normalizationTypeOption = null;
    private static Option bypassMinimumMapCountCheckOption = null;
    private static Option verboseOption = null;
    private static Option legacyOutputOption = null;
    private static Option helpOption = null;
    private static Option versionOption = null;
    private static Option threadNumOption = null;


    // APA
    private static Option apaWindowOption = null;
    private static Option apaMinValOption = null;
    private static Option apaMaxValOption = null;
    private static Option multipleCornerRegionDimensionsOption = null;
    private static Option includeInterChromosomalOption = null;
    private static Option apaSaveAllData = null;

    // for HiCCUPS
    private static Option cpuVersionHiCCUPSOption = null;
    private static Option restrictSearchRegionsOption = null;
    private static Option fdrOption = null;
    private static Option windowOption = null;
    private static Option peakOption = null;
    private static Option clusterRadiusOption = null;
    private static Option thresholdOption = null;

    // previously for AFA
    private static Option relativeLocationOption = null;
    private static Option multipleAttributesOption = null;

    public CommandLineParserForJuicer() {
        // used flags
        // wmnxcrplafdptkqbvuhgjyz

        // available flags
        // oes

        // General
        matrixSizeOption = addIntegerOption('m', "matrix_window_width");
        multipleChromosomesOption = addStringOption('c', "chromosomes");
        multipleResolutionsOption = addStringOption('r', "resolutions");
        normalizationTypeOption = addStringOption('k', "normalization");
        bypassMinimumMapCountCheckOption = addBooleanOption('b', "ignore_sparsity");
        verboseOption = addBooleanOption('v', "verbose");
        legacyOutputOption = addBooleanOption('g', "legacy");
        helpOption = addBooleanOption('h', "help");
        versionOption = addBooleanOption('V', "version");
        threadNumOption = addIntegerOption('z', "threads");

        // APA
        apaWindowOption = addIntegerOption('w', "window");
        apaMinValOption = addDoubleOption('n', "min_dist");
        apaMaxValOption = addDoubleOption('x', "max_dist");
        multipleCornerRegionDimensionsOption = addStringOption('q', "corner_width");
        includeInterChromosomalOption = addBooleanOption('e', "include_inter_chr");
        apaSaveAllData = addBooleanOption('u', "all_data");

        // HICCUPS
        fdrOption = addStringOption('f', "fdr_thresholds");
        windowOption = addStringOption('i', "window_width");
        peakOption = addStringOption('p', "peak_width");
        clusterRadiusOption = addStringOption('d', "centroid_radii");
        thresholdOption = addStringOption('t', "postprocessing_thresholds");
        cpuVersionHiCCUPSOption = addBooleanOption('j', "cpu");
        restrictSearchRegionsOption = addBooleanOption('y', "restrict");

        // previously for AFA
        relativeLocationOption = addStringOption('l', "location_type");
        multipleAttributesOption = addStringOption('a', "attributes");
    }

    public static boolean isJuicerCommand(String cmd) {
        return cmd.equals("hiccups") || cmd.equals("apa") || cmd.equals("arrowhead") || cmd.equals("motifs")
                || cmd.equals("cluster") || cmd.equals("compare") || cmd.equals("loop_domains") ||
                cmd.equals("hiccupsdiff") || cmd.equals("ab_compdiff") || cmd.equals("genes")
                || cmd.equals("apa_vs_distance") || cmd.equals("drink");
    }

    public boolean getBypassMinimumMapCountCheckOption() {
        Object opt = getOptionValue(bypassMinimumMapCountCheckOption);
        return opt != null;
    }

    public boolean getVerboseOption() {
        Object opt = getOptionValue(verboseOption);
        return opt != null;
    }

    public boolean getLegacyOutputOption() {
        Object opt = getOptionValue(legacyOutputOption);
        return opt != null;
    }

    public boolean getIncludeInterChromosomal() {
        Object opt = getOptionValue(includeInterChromosomalOption);
        return opt != null;
    }


    public boolean getAPASaveAllData() {
        Object opt = getOptionValue(apaSaveAllData);
        return opt != null;
    }

    public boolean getHelpOption() {
        Object opt = getOptionValue(helpOption);
        return opt != null;
    }

    public boolean getVersionOption() {
        Object opt = getOptionValue(versionOption);
        return opt != null;
    }

    /**
     * String flags
     */
    private String optionToString(Option option) {
        Object opt = getOptionValue(option);
        return opt == null ? null : opt.toString();
    }

    public String getRelativeLocationOption() {
        return optionToString(relativeLocationOption);
    }

    public NormalizationType getNormalizationTypeOption(NormalizationHandler normalizationHandler) {
        return retrieveNormalization(optionToString(normalizationTypeOption), normalizationHandler);
    }

    private NormalizationType retrieveNormalization(String norm, NormalizationHandler normalizationHandler) {
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

    /**
     * int flags
     */
    private int optionToInt(Option option) {
        Object opt = getOptionValue(option);
        return opt == null ? -1 : ((Number) opt).intValue();
    }

    public int getAPAWindowSizeOption() {
        return optionToInt(apaWindowOption);
    }

    public int getMatrixSizeOption() {
        return optionToInt(matrixSizeOption);
    }

    public int getNumThreads() {
        return optionToInt(threadNumOption);
    }

    /**
     * double flags
     */
    private double optionToDouble(Option option) {
        Object opt = getOptionValue(option);
        return opt == null ? -1 : ((Number) opt).doubleValue();
    }

    public double getAPAMinVal() {
        return optionToDouble(apaMinValOption);
    }

    public double getAPAMaxVal() {
        return optionToDouble(apaMaxValOption);
    }

    /**
     * String Set flags
     */
    private List<String> optionToStringList(Option option) {
        Object opt = getOptionValue(option);
        return opt == null ? null : new ArrayList<>(Arrays.asList(opt.toString().split(",")));
    }

    List<String> getChromosomeOption() {
        return optionToStringList(multipleChromosomesOption);
    }

    public List<String> getMultipleResolutionOptions() {
        return optionToStringList(multipleResolutionsOption);
    }

    public List<String> getAPACornerRegionDimensionOptions() {
        return optionToStringList(multipleCornerRegionDimensionsOption);
    }

    public List<String> getAttributeOption() {
        return optionToStringList(multipleAttributesOption);
    }

    public List<String> getFDROptions() {
        return optionToStringList(fdrOption);
    }

    public List<String> getPeakOptions() {
        return optionToStringList(peakOption);
    }

    public List<String> getWindowOptions() {
        return optionToStringList(windowOption);
    }

    public List<String> getClusterRadiusOptions() {
        return optionToStringList(clusterRadiusOption);
    }

    public List<String> getThresholdOptions() {
        return optionToStringList(thresholdOption);
    }

    public boolean getCPUVersionOfHiCCUPSOptions() {
        Object opt = getOptionValue(cpuVersionHiCCUPSOption);
        return opt != null;
    }

    public boolean restrictSearchRegionsOptions() {
        Object opt = getOptionValue(restrictSearchRegionsOption);
        return opt != null;
    }
}