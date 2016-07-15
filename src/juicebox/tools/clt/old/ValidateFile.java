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

package juicebox.tools.clt.old;

import jargs.gnu.CmdLineParser;
import juicebox.data.Dataset;
import juicebox.data.HiCFileTools;
import juicebox.tools.clt.JuiceboxCLT;

import java.util.Arrays;

/**
 * Created by muhammadsaadshamim on 6/2/16.
 */
public class ValidateFile extends JuiceboxCLT {

    private String filePath;

    public ValidateFile() {
        super(getUsage());
    }

    public static String getUsage() {
        return "validate <hicFile>";
    }

    @Override
    public void readArguments(String[] args, CmdLineParser parser) {
        if (args.length != 2) {
            printUsageAndExit();
        }
        filePath = args[1];
    }

    @Override
    public void run() {
        Dataset ds = HiCFileTools.extractDatasetForCLT(Arrays.asList(filePath.split("\\+")), true);
        assert ds.getGenomeId() != null;
        assert ds.getChromosomes().size() > 0;
        System.exit(0);
    }
}