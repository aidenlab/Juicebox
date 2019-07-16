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

package juicebox.tools.dev;

import juicebox.data.Dataset;
import juicebox.data.HiCFileTools;
import juicebox.tools.clt.CommandLineParserForJuicer;
import juicebox.tools.clt.JuicerCLT;
import juicebox.windowui.NormalizationType;

import java.io.File;
import java.util.Arrays;
import java.util.List;

/**
 * Generating Regions of Interest for Network Discovery
 */

public class Grind extends JuicerCLT {

    private int x, y, z;
    private boolean useObservedOverExpected = false;
    private boolean denseMatrix = false;
    private String chromosomes;
    private int resolution;
    private File outputDirectory;

    protected Grind(String usage) {
        super("grind [hic file] [x,y,z] [directory]");
    }

    @Override
    protected void readJuicerArguments(String[] args, CommandLineParserForJuicer juicerParser) {
        if (args.length != 3) {
            printUsageAndExit();
        }

        Dataset ds = HiCFileTools.extractDatasetForCLT(Arrays.asList(args[1].split("\\+")), true);

        // split on commas
      

        useObservedOverExpected = juicerParser.getUseObservedOverExpectedOption();
        denseMatrix = juicerParser.getDenseMatrixOption();

        outputDirectory = HiCFileTools.createValidDirectory(args[3]);

        NormalizationType preferredNorm = juicerParser.getNormalizationTypeOption(ds.getNormalizationHandler());
        if (preferredNorm != null) norm = preferredNorm;

        List<String> possibleResolutions = juicerParser.getMultipleResolutionOptions();
        if (possibleResolutions != null) {
            if (possibleResolutions.size() > 1)
                System.err.println("Only one resolution can be specified for Drink\nUsing " + possibleResolutions.get(0));
            int resolution = Integer.parseInt(possibleResolutions.get(0));
        }
    }

    @Override
    public void run() {

    }
}
