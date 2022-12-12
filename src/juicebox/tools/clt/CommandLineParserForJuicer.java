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

import juicebox.windowui.NormalizationHandler;
import juicebox.windowui.NormalizationType;

import java.util.List;

/**
 * Command Line Parser for Juicer commands (hiccups, arrowhead, apa)
 * @author Muhammad Shamim
 */
public class CommandLineParserForJuicer extends CommandLineParser {

    // used flags
    // wmnxcrplafdptkqbvuhgjyzoe

    // available flags
    // s

    // General
    private final Option matrixSizeOption = addIntegerOption('m', "matrix-width");
    private final Option multipleChromosomesOption = addStringOption('c', "chromosomes");
    private final Option multipleResolutionsOption = addStringOption('r', "resolutions");
    private final Option legacyOutputOption = addBooleanOption('g', "legacy");

    // APA
    private final Option apaWindowOption = addIntegerOption('w', "window");
    private final Option apaMinValOption = addDoubleOption('n', "min-dist");
    private final Option apaMaxValOption = addDoubleOption('x', "max-dist");
    private final Option multipleCornerRegionDimensionsOption = addStringOption('q', "corner-width");
    private final Option includeInterChromosomalOption = addBooleanOption('e', "include-inter-chr");
    private final Option apaSaveAllData = addBooleanOption('u', "save-all");
    private final Option apaDontIncludePlots = addBooleanOption('o', "no-plots");

    // HICCUPS
    private final Option fdrOption = addStringOption('f', "fdr-thresholds");
    private final Option windowOption = addStringOption('i', "window-width");
    private final Option peakOption = addStringOption('p', "peak-width");
    private final Option clusterRadiusOption = addStringOption('d', "centroid-radii");
    private final Option thresholdOption = addStringOption('t', "postprocessing-thresholds");
    private final Option cpuVersionHiCCUPSOption = addBooleanOption('j', "cpu");
    private final Option restrictSearchRegionsOption = addBooleanOption('y', "restrict");

    //Localizer
    private final Option expandSize = addIntegerOption('e', "expand");
    private final Option numPeaks = addIntegerOption("num-peaks");
    private final Option relativeLocationOption = addStringOption('l', "location-type");
    private final Option multipleAttributesOption = addStringOption('a', "attributes");

    public CommandLineParserForJuicer() {
    }

    public static boolean isJuicerCommand(String cmd) {
        return cmd.equals("hiccups") || cmd.equals("hiccups2") || cmd.equals("hiccups-scores") || cmd.equals("apa") || cmd.equals("arrowhead") || cmd.equals("motifs")
                || cmd.equals("cluster") || cmd.equals("compare") || cmd.equals("loop_domains") || cmd.equals("localizer")
                || cmd.equals("hiccupsdiff") || cmd.equals("ab_compdiff") || cmd.equals("genes")
                || cmd.equals("apa_vs_distance");
    }

    public boolean getLegacyOutputOption() {
        return optionToBoolean(legacyOutputOption);
    }

    public boolean getIncludeInterChromosomal() {
        return optionToBoolean(includeInterChromosomalOption);
    }


    public boolean getAPASaveAllData() {
        return optionToBoolean(apaSaveAllData);
    }

    public boolean getAPADontIncludePlots() {
        return optionToBoolean(apaDontIncludePlots);
    }

    /**
     * String flags
     */
    public NormalizationType getNormalizationTypeOption(NormalizationHandler normalizationHandler) {
        return retrieveNormalization(optionToString(normalizationTypeOption), normalizationHandler);
    }

    public NormalizationType[] getBothNormalizationTypeOption(NormalizationHandler normHandler1,
                                                              NormalizationHandler normHandler2) {
        NormalizationType[] normalizationTypes = new NormalizationType[2];
        String normStrings = optionToString(normalizationTypeOption);
        if (normStrings != null) {
            String[] bothNorms = normStrings.split(",");
            if (bothNorms.length > 2 || bothNorms.length < 1) {
                System.err.println("Invalid norm syntax: " + normStrings);
                return null;
            } else if (bothNorms.length == 2) {
                normalizationTypes[0] = retrieveNormalization(bothNorms[0], normHandler1);
                normalizationTypes[1] = retrieveNormalization(bothNorms[1], normHandler2);
            } else if (bothNorms.length == 1) {
                normalizationTypes[0] = retrieveNormalization(bothNorms[0], normHandler1);
                normalizationTypes[1] = retrieveNormalization(bothNorms[0], normHandler2);
            }
            return normalizationTypes;
        }
        return null;
    }

    /**
     * int flags
     */
    public int getAPAWindowSizeOption() {
        return optionToInt(apaWindowOption);
    }

    public int getMatrixSizeOption() {
        return optionToInt(matrixSizeOption);
    }

    public int getExpandSize() { return optionToInt(expandSize);}

    public int getNumPeaks() { return optionToInt(numPeaks);}

    /**
     * double flags
     */

    public double getAPAMinVal() {
        return optionToDouble(apaMinValOption);
    }

    public double getAPAMaxVal() {
        return optionToDouble(apaMaxValOption);
    }

    /**
     * String Set flags
     */

    List<String> getChromosomeListOption() {
        return optionToStringList(multipleChromosomesOption);
    }

    // todo fix to return list of ints
    public List<String> getMultipleResolutionOptions() {
        return optionToStringList(multipleResolutionsOption);
    }

    public List<String> getAPACornerRegionDimensionOptions() {
        return optionToStringList(multipleCornerRegionDimensionsOption);
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
        return optionToBoolean(restrictSearchRegionsOption);
    }
}