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
import juicebox.data.Dataset;
import juicebox.data.HiCFileTools;
import juicebox.windowui.NormalizationType;

import java.util.Arrays;

/**
 * All command line tools should extend from this class
 */
public abstract class JuiceboxCLT {

    private static String usage;
    protected Dataset dataset = null;
    protected NormalizationType norm = null;

    protected JuiceboxCLT(String usage) {
        setUsage(usage);
    }

    public abstract void readArguments(String[] args, CmdLineParser parser);

    public abstract void run();

    private void setUsage(String newUsage) {
        usage = newUsage;
    }

    public void printUsageAndExit() {
        System.out.println("Usage:   juicer_tools " + usage);
        System.exit(0);
    }

    public void printUsageAndExit(int exitcode) {
        System.out.println("Usage:   juicer_tools " + usage);
        System.exit(exitcode);
    }

    protected void setDatasetAndNorm(String files, String normType, boolean allowPrinting) {
        dataset = HiCFileTools.extractDatasetForCLT(Arrays.asList(files.split("\\+")), allowPrinting);

        norm = dataset.getNormalizationHandler().getNormTypeFromString(normType);
        if (norm == null) {
            System.err.println("Normalization type " + norm + " unrecognized.  Normalization type must be one of \n" +
                    "\"NONE\", \"VC\", \"VC_SQRT\", \"KR\", \"GW_KR\"," +
                    " \"GW_VC\", \"INTER_KR\", or \"INTER_VC\".");
            System.exit(16);
        }
    }
}

