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

import juicebox.data.ChromosomeHandler;
import juicebox.data.Dataset;
import juicebox.data.HiCFileTools;
import juicebox.tools.clt.CommandLineParserForJuicer;
import juicebox.tools.clt.JuicerCLT;
import juicebox.tools.utils.juicer.grind.*;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.Feature2DParser;
import juicebox.windowui.NormalizationType;

import java.io.File;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Generating Regions of Interest for Network Discovery
 */

public class Grind extends JuicerCLT {

    private int x, y, z;
    private int sliceTypeOption = 0;
    private boolean useObservedOverExpected = false;
    private Dataset ds;
    private boolean useDenseLabels = false;
    private boolean onlyMakePositiveExamples = false;
    private boolean ignoreDirectionOrientation = false;
    private boolean wholeGenome = false;
    private File outputDirectory;
    private Set<Integer> resolutions = new HashSet<>();
    private String featureListPath;
    private int offsetOfCornerFromDiagonal = 0;
    private int stride = 1;
    private String imgFileType = "";
    private boolean useAmorphicPixelLabeling = false;
    private boolean useDiagonal = false;

    public Grind() {
        super("grind [-k NONE/KR/VC/VC_SQRT] [-r resolution] [--stride increment] " +
                "[--off-from-diagonal max-dist-from-diag] " +
                "--observed-over-expected --dense-labels --ignore-feature-orientation --only-make-positives " + //--whole-genome --distort
                "<--loops --domains --stripes> <hic file> <bedpe positions> <x,y,z> <directory>");
    }

    @Override
    protected void readJuicerArguments(String[] args, CommandLineParserForJuicer juicerParser) {
        if (args.length != 5) {
            printUsageAndExit();
        }

        ds = HiCFileTools.
                extractDatasetForCLT(Arrays.asList(args[1].split("\\+")), true);

        featureListPath = args[2];

        // split on commas
        // save the dimensions
        String[] dimensions = args[3].split(",");
        x = Integer.parseInt(dimensions[0]);
        y = Integer.parseInt(dimensions[1]);
        z = Integer.parseInt(dimensions[2]);

        useObservedOverExpected = juicerParser.getUseObservedOverExpectedOption();
        ignoreDirectionOrientation = juicerParser.getUseIgnoreDirectionOrientationOption();
        useAmorphicPixelLabeling = juicerParser.getUseAmorphicLabelingOption();
        onlyMakePositiveExamples = juicerParser.getUseOnlyMakePositiveExamplesOption();
        useDenseLabels = juicerParser.getDenseLabelsOption();
        wholeGenome = juicerParser.getUseWholeGenome();
        offsetOfCornerFromDiagonal = juicerParser.getCornerOffBy();
        stride = juicerParser.getStride();
        outputDirectory = HiCFileTools.createValidDirectory(args[4]);
        useDiagonal = juicerParser.getUseGenomeDiagonal();

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

        sliceTypeOption = juicerParser.getGrindDataSliceOption();
        imgFileType = juicerParser.getGenerateImageFormatPicturesOption();
    }

    @Override
    public void run() {

        Feature2DList feature2DList = null;
        try {
            feature2DList = Feature2DParser.loadFeatures(featureListPath, ds.getChromosomeHandler(), false, null, false);
        } catch (Exception e) {
            if (sliceTypeOption != 4) {
                System.err.println("Feature list failed to load");
                e.printStackTrace();
                System.exit(-9);
            }
        }

        ChromosomeHandler chromosomeHandler = ds.getChromosomeHandler();
        if (givenChromosomes != null)
            chromosomeHandler = HiCFileTools.stringToChromosomes(givenChromosomes, chromosomeHandler);

        RegionFinder finder = null;
        if (sliceTypeOption == 1) {
            finder = new LoopFinder(x, y, z, stride, ds, feature2DList, outputDirectory, chromosomeHandler, norm,
                    useObservedOverExpected, useDenseLabels, resolutions, onlyMakePositiveExamples, useAmorphicPixelLabeling);
        } else if (sliceTypeOption == 2) {
            finder = new DomainFinder(x, y, z, ds, feature2DList, outputDirectory, chromosomeHandler, norm,
                    useObservedOverExpected, useDenseLabels, resolutions);
        } else if (sliceTypeOption == 4) {
            ExecutorService executor = Executors.newFixedThreadPool(resolutions.size());
            for (final int resolution : resolutions) {
                Runnable worker = new Runnable() {
                    @Override
                    public void run() {
                        ChromosomeHandler chromosomeHandler = ds.getChromosomeHandler();
                        if (givenChromosomes != null)
                            chromosomeHandler = HiCFileTools.stringToChromosomes(givenChromosomes, chromosomeHandler);

                        RegionFinder finder = new DistortionFinder(x, y, z, ds, outputDirectory, chromosomeHandler, norm,
                                useObservedOverExpected, useDenseLabels, resolution, stride, imgFileType, useDiagonal);
                        finder.makeExamples();
                    }
                };
                executor.execute(worker);
            }
            executor.shutdown();

            // Wait until all threads finish
            while (!executor.isTerminated()) {
            }
        } else {
            finder = new StripeFinder(x, y, z, ds, feature2DList, outputDirectory, chromosomeHandler, norm, useObservedOverExpected,
                    useDenseLabels, resolutions, offsetOfCornerFromDiagonal, stride, onlyMakePositiveExamples,
                    ignoreDirectionOrientation, useAmorphicPixelLabeling);
        }
        if (finder != null) {
            finder.makeExamples();
        }
    }
}
