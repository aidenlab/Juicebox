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

import juicebox.data.HiCFileTools;
import juicebox.tools.clt.CommandLineParserForJuicer;
import juicebox.tools.clt.JuicerCLT;
import juicebox.tools.utils.juicer.grind.*;
import juicebox.track.feature.Feature2DParser;
import juicebox.windowui.NormalizationType;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Generating Regions of Interest for Network Discovery
 */

public class Grind extends JuicerCLT {

    public static final int LIST_ITERATION_OPTION = 1;
    public static final int DOMAIN_OPTION = 2;
    public static final int DOWN_DIAGONAL_OPTION = 3;
    public static final int DISTORTION_OPTION = 4;
    private ParameterConfigurationContainer container = new ParameterConfigurationContainer();

    public Grind() {
        super("grind [-k NONE/KR/VC/VC_SQRT] [-r resolution] [--stride increment] " +
                "[--off-from-diagonal max-dist-from-diag] " +
                "--observed-over-expected --dense-labels --ignore-feature-orientation --only-make-positives " + //--whole-genome --distort
                "<mode> <hic file> <bedpe positions> <x,y,z> <directory>" +
                "     \n" +
                "     mode: --iterate-down-diagonal --iterate-on-list --iterate-distortions --iterate-domains");
    }

    @Override
    protected void readJuicerArguments(String[] args, CommandLineParserForJuicer juicerParser) {
        if (args.length != 5) {
            printUsageAndExit();
        }

        container.ds = HiCFileTools.
                extractDatasetForCLT(Arrays.asList(args[1].split("\\+")), true);

        container.featureListPath = args[2];

        // split on commas
        // save the dimensions
        String[] dimensions = args[3].split(",");
        container.x = Integer.parseInt(dimensions[0]);
        container.y = Integer.parseInt(dimensions[1]);
        container.z = Integer.parseInt(dimensions[2]);

        container.useObservedOverExpected = juicerParser.getUseObservedOverExpectedOption();
        container.featureDirectionOrientationIsImportant = juicerParser.getDontIgnoreDirectionOrientationOption();
        container.useAmorphicPixelLabeling = juicerParser.getUseAmorphicLabelingOption();
        container.onlyMakePositiveExamples = juicerParser.getUseOnlyMakePositiveExamplesOption();
        container.useDenseLabelsNotBinary = juicerParser.getDenseLabelsOption();
        container.wholeGenome = juicerParser.getUseWholeGenome();
        container.offsetOfCornerFromDiagonal = juicerParser.getCornerOffBy();
        container.stride = juicerParser.getStride();
        container.outputDirectory = HiCFileTools.createValidDirectory(args[4]);
        container.useDiagonal = juicerParser.getUseGenomeDiagonal();
        container.useTxtInsteadOfNPY = juicerParser.getUseTxtInsteadOfNPY();

        NormalizationType preferredNorm = juicerParser.getNormalizationTypeOption(container.ds.getNormalizationHandler());
        if (preferredNorm != null) norm = preferredNorm;
        container.norm = norm;

        List<String> possibleResolutions = juicerParser.getMultipleResolutionOptions();
        if (possibleResolutions != null) {
            for (String num : possibleResolutions) {
                container.resolutions.add(Integer.parseInt(num));
            }
        } else {
            container.resolutions.add(10000);
        }

        container.grindIterationTypeOption = juicerParser.getGrindDataSliceOption();
        container.imgFileType = juicerParser.getGenerateImageFormatPicturesOption();
    }

    @Override
    public void run() {

        container.chromosomeHandler = container.ds.getChromosomeHandler();
        container.feature2DList = null;
        try {
            container.feature2DList = Feature2DParser.loadFeatures(container.featureListPath, container.chromosomeHandler, false, null, false);
        } catch (Exception e) {
            if (container.grindIterationTypeOption != 4) {
                System.err.println("Feature list failed to load");
                e.printStackTrace();
                System.exit(-9);
            }
        }

        if (givenChromosomes != null)
            container.chromosomeHandler = HiCFileTools.stringToChromosomes(givenChromosomes, container.chromosomeHandler);

        RegionFinder finder = null;
        if (container.grindIterationTypeOption == LIST_ITERATION_OPTION) {
            finder = new IterateOnFeatureListFinder(container);
        } else if (container.grindIterationTypeOption == DOMAIN_OPTION) {
            finder = new DomainFinder(container);
        } else if (container.grindIterationTypeOption == DISTORTION_OPTION) {
            runDistortionTypeOfIteration();
        } else {
            finder = new IterateDownDiagonalFinder(container);
        }

        if (finder != null) {
            finder.makeExamples();
        }
    }

    private void runDistortionTypeOfIteration() {
        ExecutorService executor = Executors.newFixedThreadPool(container.resolutions.size());
        for (final int resolution : container.resolutions) {
            Runnable worker = new Runnable() {
                @Override
                public void run() {
                    RegionFinder finder = new DistortionFinder(resolution, container);
                    finder.makeExamples();
                }
            };
            executor.execute(worker);
        }
        executor.shutdown();

        // Wait until all threads finish
        while (!executor.isTerminated()) {
        }
    }

}
