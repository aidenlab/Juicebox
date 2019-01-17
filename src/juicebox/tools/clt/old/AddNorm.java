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

package juicebox.tools.clt.old;

import jargs.gnu.CmdLineParser;
import juicebox.tools.clt.CommandLineParser;
import juicebox.tools.clt.JuiceboxCLT;
import juicebox.tools.utils.original.norm.CustomNormVectorFileHandler;
import juicebox.tools.utils.original.norm.NormalizationVectorUpdater;


public class AddNorm extends JuiceboxCLT {

    private boolean noFragNorm = false;

    private String inputVectorFile = null;

    private int genomeWideResolution = -100;

    private String file;

    public AddNorm() {
        super(getBasicUsage()+"\n"
                + "           : -d use intra chromosome (diagonal) [false]\n"
                + "           : -F don't calculate normalization for fragment-delimited maps [false]\n"
                + "           : -w <int> calculate genome-wide resolution on all resolutions >= input resolution [not set]\n"
                + " Above options ignored if input_vector_file present\n"
        );
    }

    public static String getBasicUsage() {
        return "addNorm <input_HiC_file> [input_vector_file]";
    }

    @Override
    public void readArguments(String[] args, CmdLineParser parser) {
        CommandLineParser parser1 = (CommandLineParser) parser;
        if (parser1.getHelpOption()) {
            printUsageAndExit();
        }

        if (args.length == 3) {
            inputVectorFile = args[2];
        }
        else if (args.length != 2) {
            printUsageAndExit();
        }
        noFragNorm = parser1.getNoFragNormOption();
        genomeWideResolution = parser1.getGenomeWideOption();
        file = args[1];

    }

    @Override
    public void run() {
        try {
            if (inputVectorFile != null) {
                CustomNormVectorFileHandler.updateHicFile(file, inputVectorFile);
            }
            else {
                boolean useGenomeWideResolution = genomeWideResolution != -100;
                if (useGenomeWideResolution)
                    NormalizationVectorUpdater.updateHicFile(file, genomeWideResolution, noFragNorm);
                else
                    NormalizationVectorUpdater.updateHicFile(file, 0, noFragNorm);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}