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

import jargs.gnu.CmdLineParser;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * Created by muhammadsaadshamim on 9/4/15.
 */
public class CommandLineParserForJuicer extends CmdLineParser {

    // ints
    private static Option apaWindowOption = null;
    private static Option matrixSizeOption = null;
    // doubles
    private static Option apaMinValOption = null;
    private static Option apaMaxValOption = null;
    // sets of strings
    private static Option multipleChromosomesOption = null;
    private static Option multipleResolutionsOption = null;
    // for motif finder
    private static Option ctcfOption = null;
    private static Option ctcfCollapsedOption = null;
    private static Option rad21Option = null;
    private static Option smc3Option = null;
    public CommandLineParserForJuicer() {

        apaWindowOption = addIntegerOption('w', "window");
        matrixSizeOption = addIntegerOption('m', "minCountThreshold (hiccups)");

        apaMinValOption = addDoubleOption('n', "minimum value");
        apaMaxValOption = addDoubleOption('x', "maximum value");

        multipleChromosomesOption = addStringOption('c', "chromosomes");
        multipleResolutionsOption = addStringOption('r', "multiple resolutions separated by ','");

        ctcfOption = addStringOption('c', "CTCF_input_file");
        ctcfCollapsedOption = addStringOption('a', "CTCF_collapsed_input_file");
        rad21Option = addStringOption('r', "RAD21_input_file");
        smc3Option = addStringOption('s', "SMC3_input_file");
    }

    public static boolean isJuicerCommand(String cmd) {
        return cmd.equals("hiccups") || cmd.equals("apa") || cmd.equals("arrowhead") || cmd.equals("motif_finder");
    }

    /**
     * String flags
     */
    private String optionToString(Option option) {
        Object opt = getOptionValue(option);
        return opt == null ? null : opt.toString();
    }

    public String getCTCFOption() {
        return optionToString(ctcfOption);
    }

    public String getCTCFCollapsedOption() {
        return optionToString(ctcfCollapsedOption);
    }

    public String getRAD21Option() {
        return optionToString(rad21Option);
    }

    public String getSMC3Option() {
        return optionToString(smc3Option);
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

    public int getMatrixSizeOption() {
        return optionToInt(matrixSizeOption);
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