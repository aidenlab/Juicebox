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
import juicebox.tools.clt.JuiceboxCLT;
import juicebox.tools.utils.original.norm.GenomeWideNormalizationVectorUpdater;


public class AddGWNorm extends JuiceboxCLT {

    private String file;
    private int genomeWideResolution = -100;

    public AddGWNorm() {
        super("addGWNorm <input_HiC_file> <min resolution>");
    }

    @Override
    public void readArguments(String[] args, CmdLineParser parser) {
        //setUsage("juicebox addGWNorm hicFile <max genome-wide resolution>");
        if (args.length != 3) {
            printUsageAndExit();
        }
        file = args[1];

        try {
            genomeWideResolution = Integer.valueOf(args[2]);
        } catch (NumberFormatException error) {
            printUsageAndExit();
        }
    }

    @Override
    public void run() {
        try {
            GenomeWideNormalizationVectorUpdater.addGWNorm(file, genomeWideResolution);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
