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

import juicebox.tools.HiCTools;
import juicebox.tools.NormalizationVectorUpdater;

import java.io.IOException;


public class AddNorm extends JuiceboxCLT {

    private boolean useGenomeWideResolution = false;

    private int genomeWideResolution = -100;

    private String file;

    @Override
    public void readArguments(String[] args, HiCTools.CommandLineParser parser) throws IOException {
        //setUsage("juicebox addNorm hicFile <max genome-wide resolution>");
        if (args.length < 2 || args.length > 3) {
            throw new IOException("1");
        }
        file = args[1];
        if (args.length > 2) {
            try {
                genomeWideResolution = Integer.valueOf(args[2]);
            } catch (NumberFormatException error) {
                throw new IOException("1");
            }
            useGenomeWideResolution = true;
        }
    }

    @Override
    public void run() throws IOException {
        if (useGenomeWideResolution)
            NormalizationVectorUpdater.updateHicFile(file, genomeWideResolution);
        else
            NormalizationVectorUpdater.updateHicFile(file);
    }
}