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
import juicebox.tools.utils.juicer.grind.DomainFinder;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.Feature2DParser;
import juicebox.windowui.NormalizationType;

import java.io.File;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Generating Regions of Interest for Network Discovery
 */

public class Grind extends JuicerCLT {

    private int x, y, z;
    private boolean useObservedOverExpected = false;
    Dataset ds;
    private boolean useDenseLabels = false;
    private Set<String> chromosome = null;
    private boolean wholeGenome = false;
    private File outputDirectory;
    private Set<Integer> resolutions = new HashSet<>();

    protected Grind(String usage) {
        super("grind [hic file] [bedpe positions] [x,y,z] [directory]");
    }

    @Override
    protected void readJuicerArguments(String[] args, CommandLineParserForJuicer juicerParser) {
        if (args.length != 3) {
            printUsageAndExit();
        }

        ds = HiCFileTools.extractDatasetForCLT(Arrays.asList(args[1].split("\\+")), true);


        // split on commas
        // save the dimensions
        String[] dimensions = args[3].split(",");
        x = Integer.parseInt(dimensions[0]);
        y = Integer.parseInt(dimensions[1]);
        z = Integer.parseInt(dimensions[2]);


        useObservedOverExpected = juicerParser.getUseObservedOverExpectedOption();
        useDenseLabels = juicerParser.getDenseLabelsOption();

        outputDirectory = HiCFileTools.createValidDirectory(args[4]);

        NormalizationType preferredNorm = juicerParser.getNormalizationTypeOption(ds.getNormalizationHandler());
        if (preferredNorm != null) norm = preferredNorm;

        List<String> possibleResolutions = juicerParser.getMultipleResolutionOptions();
        if (possibleResolutions != null) {
            for (String num : possibleResolutions) {
                resolutions.add(Integer.parseInt(num));
            }
        } else {
            resolutions.add(10000);
        }
    }

    @Override
    public void run() {

        Feature2DList features = Feature2DParser.loadFeatures("loopListPath", ds.getChromosomeHandler(), false, null, false);

        // use these as inputs
        DomainFinder domainFinder = new DomainFinder(ds, features, outputDirectory, givenChromosomes, norm, useObservedOverExpected, useDenseLabels, resolutions);

        // read in any additional data required


        // iterate over regions of interest and save them to a directory


        // oe

        //RealMatrix localizedRegionData = ExtractingOEDataUtils.extractLocalThresholdedLogOEBoundedRegion(zd, 0, maxBin,
        //        0, maxBin, maxSize, maxSize, norm, true, df, chromosome.getIndex(), logThreshold);





    }
}
