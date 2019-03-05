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

package juicebox.tools.clt.juicer;

import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.data.*;
import juicebox.tools.clt.CommandLineParserForJuicer;
import juicebox.tools.clt.JuicerCLT;
import juicebox.tools.utils.juicer.arrowhead.ArrowheadScoreList;
import juicebox.tools.utils.juicer.arrowhead.BlockBuster;
import juicebox.tools.utils.juicer.hiccups.HiCCUPSUtils;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.Feature2DParser;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationHandler;
import juicebox.windowui.NormalizationType;
import org.broad.igv.feature.Chromosome;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Arrowhead
 * <p/>
 * Developed by Miriam Huntley and Neva Durand
 * Implemented by Muhammad Shamim
 * <p/>
 * -------
 * Arrowhead
 * -------
 * <p/>
 * arrowhead [-c chromosome(s)] [-m matrix size] <NONE/VC/VC_SQRT/KR> <input_HiC_file(s)> <output_file>
 * <resolution> [feature_list] [control_list]
 * *
 * The required arguments are:
 * <p/>
 * <NONE/VC/VC_SQRT/KR> One of the normalizations must be selected (case sensitive). Generally, KR (Knight-Ruiz)
 * balancing should be used.
 * <p/>
 * <input_HiC_file(s)>: Address of HiC File(s) which should end with .hic.  This is the file you will
 * load into Juicebox. URLs or local addresses may be used. To sum multiple HiC Files together,
 * use the '+' symbol between the addresses (no whitespace between addresses).
 * <p/>
 * <output_file>: Final list of all contact domains found by Arrowhead. Can be visualized directly in Juicebox
 * as a 2D annotation.
 * <p/>
 * <resolution>: Integer resolution for which Arrowhead will be run. Generally, 5kB (5000) or 10kB (10000)
 * resolution is used depending on the depth of sequencing in the hic file(s).
 * <p/>
 * -- NOTE -- If you want to find scores for a feature and control list, both must be provided:
 * <p/>
 * [feature_list]: Feature list of loops/domains for which block scores are to be calculated
 * <p/>
 * [control_list]: Control list of loops/domains for which block scores are to be calculated
 * <p/>
 * <p/>
 * The optional arguments are:
 * <p/>
 * -m <int> Size of the sliding window along the diagonal in which contact domains will be found. Must be an even
 * number as (m/2) is used as the increment for the sliding window. (Default 2000)
 * <p/>
 * -c <String(s)> Chromosome(s) on which Arrowhead will be run. The number/letter for the chromosome can be used with or
 * without appending the "chr" string. Multiple chromosomes can be specified using commas (e.g. 1,chr2,X,chrY)
 * <p/>
 * <p/>
 * ----------------
 * Arrowhead Examples
 * ----------------
 * <p/>
 * arrowhead -m 2000 KR ch12-lx-b-lymphoblasts_mapq_30.hic contact_domains_list 10000
 * This command will run Arrowhead on a mouse cell line HiC map and save all contact domains to the
 * contact_domains_list file. These are the settings used to generate the official contact domain list on the
 * ch12-lx-b-lymphoblast cell line.
 * <p/>
 * arrowhead KR GM12878_mapq_30.hic contact_domains_list 5000
 * This command will run Arrowhead on the GM12878 HiC map and save all contact domains to the contact_domains_list
 * file. These are the settings used to generate the official GM12878 contact domain list.
 */
public class Arrowhead extends JuicerCLT {

    private static int matrixSize = 2000;
    private File outputDirectory;
    private boolean configurationsSetByUser = false;
    private boolean controlAndListProvided = false;
    private String featureList, controlList;
    // must be passed via command line
    private int resolution = 10000;
    private Dataset ds;
    private boolean checkMapDensityThreshold = true;
    private static int numCPUThreads = 4;

    public Arrowhead() {
        super("arrowhead [-c chromosome(s)] [-m matrix size] [-r resolution] [-k normalization (NONE/VC/VC_SQRT/KR)] " +
                "[--ignore_sparsity flag] <hicFile(s)> <output_file> [feature_list] [control_list]");
        HiCGlobals.useCache = false;
    }

    public static String getBasicUsage() {
        return "arrowhead <hicFile(s)> <output_file>";
    }

    @Override
    protected void readJuicerArguments(String[] args, CommandLineParserForJuicer juicerParser) {
        if (args.length != 3 && args.length != 5) {
            // 3 - standard, 5 - when list/control provided
            printUsageAndExit();  // this will exit
        }

        ds = HiCFileTools.extractDatasetForCLT(Arrays.asList(args[1].split("\\+")), true);
        outputDirectory = HiCFileTools.createValidDirectory(args[2]);


        NormalizationType preferredNorm = juicerParser.getNormalizationTypeOption(ds.getNormalizationHandler());
        if (preferredNorm != null)
            norm = preferredNorm;

        List<String> potentialResolution = juicerParser.getMultipleResolutionOptions();
        if (potentialResolution != null) {
            resolution = Integer.parseInt(potentialResolution.get(0));
            configurationsSetByUser = true;
        }

        if (args.length == 5) {
            controlAndListProvided = true;
            featureList = args[3];
            controlList = args[4];
        }

        int specifiedMatrixSize = juicerParser.getMatrixSizeOption();
        if (specifiedMatrixSize > 1) {
            matrixSize = specifiedMatrixSize;
        }

        if (juicerParser.getBypassMinimumMapCountCheckOption()) {
            checkMapDensityThreshold = false;
        }

        int numThreads = juicerParser.getNumThreads();
        if (numThreads > 0) {
            numCPUThreads = numThreads;
        } else {
            numCPUThreads = Runtime.getRuntime().availableProcessors();
        }
        System.out.println("Using " + numCPUThreads + " CPU threads");

        List<String> t = juicerParser.getThresholdOptions();
        if (t != null && t.size() == 6) {
            double[] thresholds = HiCCUPSUtils.extractDoubleValues(t, 6, Double.NaN);
            if (thresholds.length == 6) {
                if (!Double.isNaN(thresholds[0])) BlockBuster.varThreshold = thresholds[0];
                if (!Double.isNaN(thresholds[1])) BlockBuster.highSignThreshold = thresholds[1];
                if (!Double.isNaN(thresholds[2])) BlockBuster.maxLowSignThreshold = thresholds[2];
                if (!Double.isNaN(thresholds[3])) BlockBuster.minLowSignThreshold = thresholds[3];
                if (!Double.isNaN(thresholds[4])) BlockBuster.decrementLowSignThreshold = thresholds[4];
                if (!Double.isNaN(thresholds[5])) BlockBuster.minBlockSize = (int) thresholds[5];
            }
        }
    }

    @Override
    public void run() {
        try {
            final ExpectedValueFunction df = ds.getExpectedValues(new HiCZoom(HiC.Unit.BP, 2500000), NormalizationHandler.NONE);
            double firstExpected = df.getExpectedValues()[0]; // expected value on diagonal
            // From empirical testing, if the expected value on diagonal at 2.5Mb is >= 100,000
            // then the map had more than 300M contacts.
            // If map has less than 300M contacts, we will not run Arrowhead or HiCCUPs
            if (HiCGlobals.printVerboseComments) {
                System.err.println("First expected is " + firstExpected);
            }

            if (firstExpected < 100000) {
                System.err.println("Warning: Hi-C map is too sparse to find many domains via Arrowhead.");
                if (checkMapDensityThreshold) {
                    System.err.println("Exiting. To disable sparsity check, use the --ignore_sparsity flag.");
                    System.exit(0);
                }
            }

            // high quality (IMR90, GM12878) maps have different settings
            if (!configurationsSetByUser) {
                matrixSize = 2000;
                if (firstExpected > 250000) {
                    resolution = 5000;
                    System.out.println("Default settings for 5kb being used");
                } else {
                    resolution = 10000;
                    System.out.println("Default settings for 10kb being used");
                }
            }
        } catch (Exception e) {
            System.err.println("Unable to assess map sparsity; continuing with Arrowhead");
            if (!configurationsSetByUser) {
                matrixSize = 2000;
                resolution = 10000;
                System.out.println("Default settings for 10kb being used");
            }
        }

        ChromosomeHandler chromosomeHandler = ds.getChromosomeHandler();

        final Feature2DList contactDomainsGenomeWide = new Feature2DList();
        final Feature2DList contactDomainListScoresGenomeWide = new Feature2DList();
        final Feature2DList contactDomainControlScoresGenomeWide = new Feature2DList();

        final Feature2DList inputList = new Feature2DList();
        final Feature2DList inputControl = new Feature2DList();
        if (controlAndListProvided) {
            inputList.add(Feature2DParser.loadFeatures(featureList, chromosomeHandler, true, null, false));
            inputControl.add(Feature2DParser.loadFeatures(controlList, chromosomeHandler, true, null, false));
        }

        File outputBlockFile = new File(outputDirectory, resolution + "_blocks.bedpe");
        File outputListFile = null;
        File outputControlFile = null;
        if (controlAndListProvided) {
            outputListFile = new File(outputDirectory, resolution + "_list_scores.bedpe");
            outputControlFile = new File(outputDirectory, resolution + "_control_scores.bedpe");
        }

        // chromosome filtering must be done after input/control created
        // because full set of chromosomes required to parse lists
        if (givenChromosomes != null)
            chromosomeHandler = HiCFileTools.stringToChromosomes(givenChromosomes, chromosomeHandler);

        final HiCZoom zoom = new HiCZoom(HiC.Unit.BP, resolution);

        final double maxProgressStatus = determineHowManyChromosomesWillActuallyRun(ds, chromosomeHandler);
        final AtomicInteger currentProgressStatus = new AtomicInteger(0);
        System.out.println("max " + maxProgressStatus);

        ExecutorService executor = Executors.newFixedThreadPool(numCPUThreads);

        for (final Chromosome chr : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {

            Runnable worker = new Runnable() {
                @Override
                public void run() {

                    Matrix matrix = ds.getMatrix(chr, chr);
                    if (matrix != null) {

                        ArrowheadScoreList list = new ArrowheadScoreList(inputList, chr, resolution);
                        ArrowheadScoreList control = new ArrowheadScoreList(inputControl, chr, resolution);

                        if (HiCGlobals.printVerboseComments) {
                            System.out.println("\nProcessing " + chr.getName());
                        }

                        // actual Arrowhead algorithm
                        BlockBuster.run(chr.getIndex(), chr.getName(), chr.getLength(), resolution, matrixSize,
                                matrix.getZoomData(zoom), norm, list, control, contactDomainsGenomeWide,
                                contactDomainListScoresGenomeWide, contactDomainControlScoresGenomeWide);

                        //todo should this be inside if? But the wouldn't increment for skipped chr;s?
                        int currProg = currentProgressStatus.incrementAndGet();
                        System.out.println(((int) Math.floor((100.0 * currProg) / maxProgressStatus)) + "% ");
                    }
                }
            };
            executor.execute(worker);
        }

        executor.shutdown();
        // Wait until all threads finish
        while (!executor.isTerminated()) {
        }

        // save the data on local machine
        contactDomainsGenomeWide.exportFeatureList(outputBlockFile, true, Feature2DList.ListFormat.ARROWHEAD);
        System.out.println(contactDomainsGenomeWide.getNumTotalFeatures() + " domains written to file: " +
                outputBlockFile.getAbsolutePath());
        if (controlAndListProvided) {
            contactDomainListScoresGenomeWide.exportFeatureList(outputListFile, false, Feature2DList.ListFormat.NA);
            contactDomainControlScoresGenomeWide.exportFeatureList(outputControlFile, false, Feature2DList.ListFormat.NA);
        }
        System.out.println("Arrowhead complete");
    }
}