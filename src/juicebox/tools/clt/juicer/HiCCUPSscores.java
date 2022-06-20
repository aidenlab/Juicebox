/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab
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
import juicebox.data.ChromosomeHandler;
import juicebox.data.Dataset;
import juicebox.data.ExpectedValueFunction;
import juicebox.data.HiCFileTools;
import juicebox.data.basics.Chromosome;
import juicebox.mapcolorui.Feature2DHandler;
import juicebox.tools.clt.CommandLineParserForJuicer;
import juicebox.tools.clt.JuicerCLT;
import juicebox.tools.utils.common.ArrayTools;
import juicebox.tools.utils.juicer.hiccups.*;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationHandler;
import juicebox.windowui.NormalizationType;

import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

/**
 *
 */
public class HiCCUPSscores extends JuicerCLT {

    public static final int regionMargin = 20;
    public static final int krNeighborhood = 5;
    public static final Color defaultPeakColor = Color.cyan;
    public static final boolean shouldColorBeScaledByFDR = false;
    public static final String CPU_VERSION_WARNING = "WARNING - You are using the CPU version of HiCCUPS.\n" +
            "The GPU version of HiCCUPS is the official version and has been tested extensively.\n" +
            "The CPU version only searches for loops within 8MB (by default) of the diagonal and is still experimental.";
    private static final int totalMargin = 2 * regionMargin;
    public static final int w1 = 50;      // TODO dimension should be variably set
    private static final int w2 = 10000;   // TODO dimension should be variably set
    private static final boolean dataShouldBePostProcessed = true;
    public static double fdrsum1 = 0.00001;
    public static double fdrsum2 = 0.00003;
    public static double oeThreshold1 = 1.5;
    public static double oeThreshold2 = 1.75;
    public static double oeThreshold3 = 2;
    public static double oeThreshold4 = 40;
    private static int matrixSize = 512;// 540 original
    private static int regionWidth = matrixSize - totalMargin;
    private boolean configurationsSetByUser = false;
    private String featureListPath;
    private boolean listGiven = false;
    private boolean checkMapDensityThreshold = true;
    private ChromosomeHandler directlyInitializedChromosomeHandler = null;

    /*
     * Reasonable Commands
     *
     * fdr = 0.10 for all resolutions
     * peak width = 1 for 25kb, 2 for 10kb, 4 for 5kb
     * window = 3 for 25kb, 5 for 10kb, 7 for 5kb
     *
     * cluster radius is 20kb for 5kb and 10kb res and 50kb for 25kb res
     * fdrsumthreshold is 0.02 for all resolutions
     * oeThreshold1 = 1.5 for all res
     * oeThreshold2 = 1.75 for all res
     * oeThreshold3 = 2 for all res
     *
     * published GM12878 looplist was only generated with 5kb and 10kb resolutions
     * same with published IMR90 looplist
     * published CH12 looplist only generated with 10kb
     */
    private File outputDirectory;
    private List<HiCCUPSConfiguration> configurations;
    private Dataset ds;
    private boolean useCPUVersionHiCCUPS = false, restrictSearchRegions = true;

    public HiCCUPSscores() {
        super("hiccups-scores [-m matrixSize] [-k normalization (NONE/VC/VC_SQRT/KR)] [-c chromosome(s)] [-r resolution(s)] " +
                "[-p peak width] [-i window]" +
                "<hicFile> <outputDirectory> <specified_loop_list>");
        Feature2D.allowHiCCUPSOrdering = true;
    }

    public static String getBasicUsage() {
        return "hiccups2 <hicFile> <outputDirectory> <specified_loop_list>";
    }

    @Override
    protected void readJuicerArguments(String[] args, CommandLineParserForJuicer juicerParser) {
        if (args.length != 4) {
            printUsageAndExit();
        }
        // TODO: add code here to check for CUDA/GPU installation. The below is not ideal.

        ds = HiCFileTools.extractDatasetForCLT(Arrays.asList(args[1].split("\\+")), true);
        outputDirectory = HiCFileTools.createValidDirectory(args[2]);

        listGiven = true;
        featureListPath = args[3];


        NormalizationType preferredNorm = juicerParser.getNormalizationTypeOption(ds.getNormalizationHandler());
        if (preferredNorm != null)
            norm = preferredNorm;

        determineValidMatrixSize(juicerParser);
        determineValidConfigurations(juicerParser, ds.getBpZooms());

        if (juicerParser.restrictSearchRegionsOptions()) {
            restrictSearchRegions = true;
            System.out.println("WARNING - You are restricting the regions the HiCCUPS will explore.");
        }

        if (juicerParser.getCPUVersionOfHiCCUPSOptions()) {
            useCPUVersionHiCCUPS = true;
            restrictSearchRegions = true;
            System.out.println(CPU_VERSION_WARNING);
        }

        updateNumberOfCPUThreads(juicerParser, 1);

    }

    /**
     * Used by hiccups diff to set the properties of hiccups directly without resorting to command line usage
     *
     * @param dataset
     * @param outputDirectoryPath
     * @param featureListPath
     * @param preferredNorm
     * @param matrixSize
     * @param providedCommonChromosomeHandler
     * @param configurations
     * @param thresholds
     */
    public void initializeDirectly(Dataset dataset, String outputDirectoryPath,
                                   String featureListPath, NormalizationType preferredNorm, int matrixSize,
                                   ChromosomeHandler providedCommonChromosomeHandler,
                                   List<HiCCUPSConfiguration> configurations, double[] thresholds,
                                   boolean usingCPUVersion, boolean restrictSearchRegions) {
        this.ds = dataset;
        outputDirectory = HiCFileTools.createValidDirectory(outputDirectoryPath);

        if (featureListPath != null) {
            listGiven = true;
            this.featureListPath = featureListPath;
        }

        directlyInitializedChromosomeHandler = providedCommonChromosomeHandler;

        if (preferredNorm != null) norm = preferredNorm;

        // will just confirm matrix size is large enough
        determineValidMatrixSize(matrixSize);

        // configurations should have been passed in
        if (configurations != null && configurations.size() > 0) {
            configurationsSetByUser = true;
            this.configurations = configurations;
        }

        // fdr & oe thresholds directly sent in
        if (thresholds != null) setHiCCUPSFDROEThresholds(thresholds);

        // force hiccups to run
        checkMapDensityThreshold = false;

        this.restrictSearchRegions = restrictSearchRegions;
        if (usingCPUVersion) {
            useCPUVersionHiCCUPS = true;
            this.restrictSearchRegions = true;
        }
    }

    @Override
    public void run() {

        try {
            final ExpectedValueFunction df = ds.getExpectedValues(new HiCZoom(HiC.Unit.BP, 2500000),
                    NormalizationHandler.NONE);
            double firstExpected = df.getExpectedValuesNoNormalization().getFirstValue(); // expected value on diagonal
            // From empirical testing, if the expected value on diagonal at 2.5Mb is >= 100,000
            // then the map had more than 300M contacts.
            if (firstExpected < 100000) {
                System.err.println("Warning Hi-C map is too sparse to find many loops via HiCCUPS2.");
            }
        } catch (Exception e) {
            System.err.println("Unable to assess map sparsity; continuing with HiCCUPS2");
        }

        if (!configurationsSetByUser) {
            configurations = HiCCUPSConfiguration.getHiCCUPS2DefaultSetOfConfigsForUsers();
        }

        ChromosomeHandler commonChromosomesHandler = ds.getChromosomeHandler();
        if (directlyInitializedChromosomeHandler != null && directlyInitializedChromosomeHandler.size() > 0) {
            commonChromosomesHandler = directlyInitializedChromosomeHandler;
        } else if (givenChromosomes != null && givenChromosomes.size() > 0) {
            commonChromosomesHandler = HiCFileTools.stringToChromosomes(givenChromosomes, commonChromosomesHandler);
        }

        Map<Integer, Feature2DList> givenLoopLists = new HashMap<>();

        File outputMergedGivenFile = new File(outputDirectory, HiCCUPSUtils.getMergedRequestedLoopsFileName());

        Feature2DHandler inputListFeature2DHandler = new Feature2DHandler();

        inputListFeature2DHandler.setLoopList(featureListPath, commonChromosomesHandler);


        for (HiCCUPSConfiguration conf : configurations) {
            System.out.println("Running HiCCUPS2 for resolution " + conf.getResolution());
            runHiccupsScoresProcessing(ds, conf, commonChromosomesHandler, inputListFeature2DHandler, givenLoopLists);
        }

        if (dataShouldBePostProcessed) {
            HiCCUPSUtils.postProcess(givenLoopLists, ds, commonChromosomesHandler,
                        configurations, norm, outputDirectory, true, outputMergedGivenFile, 2);
        }
        System.out.println("HiCCUPS2 complete");
        // else the thresholds and raw pixels were already saved when hiccups was run
    }

    /**
     * Actual run of the HiCCUPS algorithm
     *
     * @param ds                dataset from hic file
     * @param conf              configuration of hiccups inputs
     * @param chromosomeHandler list of chromosomes to run hiccups on
     * @param givenLoopLists
     * @return list of enriched pixels
     */
    private void runHiccupsScoresProcessing(Dataset ds, final HiCCUPSConfiguration conf, ChromosomeHandler chromosomeHandler,
                                               final Feature2DHandler inputListFeature2DHandler, Map<Integer, Feature2DList> givenLoopLists) {

        long begin_time = System.currentTimeMillis();

        HiCZoom zoom = ds.getZoomForBPResolution(conf.getResolution());
        if (zoom == null) {
            System.err.println("Data not available at " + conf.getResolution() + " resolution");
            return;
        }


        final float[] thresholdBL = ArrayTools.newValueInitializedFloatArray(w1, (float) 0);
        final float[] thresholdDonut = ArrayTools.newValueInitializedFloatArray(w1, (float) 0);
        final float[] thresholdH = ArrayTools.newValueInitializedFloatArray(w1, (float) 0);
        final float[] thresholdV = ArrayTools.newValueInitializedFloatArray(w1, (float) 0);

        // to hold all enriched pixels found in second run
        final Feature2DList requestedList = new Feature2DList();


        // two runs, 1st to build histograms, 2nd to identify loops

        final HiCCUPSRegionHandler regionHandler = new HiCCUPSRegionHandler(ds, chromosomeHandler, zoom, norm, conf, regionWidth,
                regionMargin, restrictSearchRegions);



        final AtomicInteger currentProgressStatus = new AtomicInteger(0);
        final AtomicInteger indexOfHiCCUPSRegion = new AtomicInteger(0);

        ExecutorService executor = Executors.newFixedThreadPool(numCPUThreads);
        for (int l = 0; l < numCPUThreads; l++) {
            Runnable worker = new Runnable() {
                @Override
                public void run() {
                    runCoreCodeForHiCCUPS2(conf, indexOfHiCCUPSRegion, currentProgressStatus, regionHandler, matrixSize,
                                thresholdBL, thresholdDonut, thresholdH, thresholdV, norm, zoom,
                                inputListFeature2DHandler, requestedList);
                }
            };
            executor.execute(worker);
        }
        executor.shutdown();

        // Wait until all threads finish
        while (!executor.isTerminated()) {
        }



        requestedList.exportFeatureList(new File(outputDirectory, HiCCUPSUtils.getRequestedLoopsFileName(conf.getResolution())),
                    true, Feature2DList.ListFormat.ENRICHED);
        givenLoopLists.put(conf.getResolution(), requestedList);




        if (HiCGlobals.printVerboseComments) {
            long final_time = System.currentTimeMillis();
            System.out.println("Total time: " + (final_time - begin_time));
        }


    }

    private void runCoreCodeForHiCCUPS2(HiCCUPSConfiguration conf, AtomicInteger indexOfHiCCUPSRegion, AtomicInteger currentProgressStatus,
                                       HiCCUPSRegionHandler regionHandler, int matrixSize,
                                       float[] thresholdBL, float[] thresholdDonut, float[] thresholdH, float[] thresholdV,
                                       NormalizationType norm, HiCZoom zoom,
                                       Feature2DHandler inputListFeature2DHandler, Feature2DList requestedList) {


        int indexOfRegionForThread = indexOfHiCCUPSRegion.getAndIncrement();

        GPUController2 gpuController = buildGPUController(conf);

        while (indexOfRegionForThread < regionHandler.getSize()) {

            HiCCUPSRegionContainer regionContainer = regionHandler.getRegionFromIndex(indexOfRegionForThread);
            try {
                if (HiCGlobals.printVerboseComments) {
                    System.out.println();
                    System.out.println("GPU Run Details");
                    System.out.println("Row bounds " + Arrays.toString(regionContainer.getRowBounds()));
                    System.out.println("Col bounds " + Arrays.toString(regionContainer.getColumnBounds()));
                }

                int[] rowBounds = regionContainer.getRowBounds();
                int[] columnBounds = regionContainer.getColumnBounds();

                Chromosome chromosome = regionContainer.getChromosome();

                float rowBound1GenomeCoords = ((float) rowBounds[4]) * conf.getResolution();
                float columnBound1GenomeCoords = ((float) columnBounds[4]) * conf.getResolution();
                float rowBound2GenomeCoords = ((float) rowBounds[5] - 1) * conf.getResolution();
                float columnBound2GenomeCoords = ((float) columnBounds[5] - 1) * conf.getResolution();
                // System.out.println(chromosome.getIndex() + "\t" + rowBound1GenomeCoords + "\t" + rowBound2GenomeCoords + "\t" + columnBound1GenomeCoords + "\t" + columnBound2GenomeCoords);
                net.sf.jsi.Rectangle currentWindow = new net.sf.jsi.Rectangle(rowBound1GenomeCoords,
                        columnBound1GenomeCoords, rowBound2GenomeCoords, columnBound2GenomeCoords);
                List<Feature2D> inputListFoundFeatures = inputListFeature2DHandler.getContainedFeatures(chromosome.getIndex(), chromosome.getIndex(),
                        currentWindow);

                if (inputListFoundFeatures.size()>0) {
                    GPUOutputContainer2 gpuOutputs = gpuController.process(regionHandler, regionContainer, matrixSize,
                            thresholdBL, thresholdDonut, thresholdH, thresholdV, norm, zoom);

                    int diagonalCorrection = (rowBounds[4] - columnBounds[4]) + conf.getPeakWidth() + 2;
                    gpuOutputs.cleanUpPvalNans();
                    gpuOutputs.cleanUpPeakNaNs();
                    gpuOutputs.cleanUpPeakDiagonal(diagonalCorrection);

                    Feature2DList peaksRequestedList = gpuOutputs.extractPeaksListGiven(chromosome.getIndex(), chromosome.getName(),
                            w1, w2, rowBounds[4], columnBounds[4], conf.getResolution(), inputListFoundFeatures);
                    requestedList.add(peaksRequestedList);
                }


                int currProg = currentProgressStatus.incrementAndGet();
                int resonableDivisor = Math.max(regionHandler.getSize() / 20, 1);
                if (HiCGlobals.printVerboseComments || currProg % resonableDivisor == 0) {
                    DecimalFormat df = new DecimalFormat("#.####");
                    df.setRoundingMode(RoundingMode.FLOOR);
                    System.out.println(df.format(Math.floor((100.0 * currProg) / regionHandler.getSize())) + "% ");
                }

            } catch (IOException e) {
                System.err.println("No data in map region");
            }

            indexOfRegionForThread = indexOfHiCCUPSRegion.getAndIncrement();
        }
    }

    private GPUController2 buildGPUController(HiCCUPSConfiguration conf) {
        try {
            return new GPUController2(conf.getWindowWidth(), conf.getResolution(), matrixSize,
                    conf.getPeakWidth(), useCPUVersionHiCCUPS);
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("GPU/CUDA Installation Not Detected");
            System.err.println("Exiting HiCCUPS");
            System.exit(26);
            return null;
        }
    }

    /**
     * @param juicerParser  Parser to determine configurations
     * @param availableZooms
     */
    private void determineValidConfigurations(CommandLineParserForJuicer juicerParser, List<HiCZoom> availableZooms) {

        configurations = HiCCUPSConfiguration.extractConfigurationsFromCommandLine(juicerParser, availableZooms, 2);
        if (configurations == null) {
            System.out.println("No valid configurations specified, using default settings");
            configurationsSetByUser = false;
        }
        else {
            configurationsSetByUser = true;
        }

        try {
            List<String> t = juicerParser.getThresholdOptions();
            if (t != null && t.size() == 4) {
                double[] thresholds = HiCCUPSUtils.extractDoubleValues(t, 4, Double.NaN);
                setHiCCUPSFDROEThresholds(thresholds);
            } else if (t != null && t.size() == 5) {
                double[] thresholds = HiCCUPSUtils.extractDoubleValues(t, 5, Double.NaN);
                setHiCCUPSFDROEThresholds(thresholds);
            } else if (t != null && t.size() == 6) {
                double[] thresholds = HiCCUPSUtils.extractDoubleValues(t, 6, Double.NaN);
                setHiCCUPSFDROEThresholds(thresholds);
            }
        } catch (Exception e) {
            // do nothing - use default postprocessing thresholds
        }
        HiCCUPSscores.fdrsum1 = fdrsum1;
        HiCCUPSscores.fdrsum2 = fdrsum2;
        HiCCUPSscores.oeThreshold1 = oeThreshold1;
        HiCCUPSscores.oeThreshold2 = oeThreshold2;
        HiCCUPSscores.oeThreshold3 = oeThreshold3;
    }

    private void determineValidMatrixSize(CommandLineParserForJuicer juicerParser) {
        determineValidMatrixSize(juicerParser.getMatrixSizeOption());
    }

    private void determineValidMatrixSize(int specifiedMatrixSize) {
        if (specifiedMatrixSize > 2 * regionMargin) {
            matrixSize = specifiedMatrixSize;
            regionWidth = specifiedMatrixSize - totalMargin;
        }
        if (HiCGlobals.printVerboseComments) {
            System.out.println("Using Matrix Size " + matrixSize);
        }
    }

    private void setHiCCUPSFDROEThresholds(double[] thresholds) {
        if (thresholds != null && thresholds.length == 4) {
            if (!Double.isNaN(thresholds[0]) && thresholds[0] > 0) fdrsum1 = thresholds[0];
            if (!Double.isNaN(thresholds[0]) && thresholds[0] > 0) fdrsum2 = thresholds[0];
            if (!Double.isNaN(thresholds[1]) && thresholds[1] > 0) oeThreshold1 = thresholds[1];
            if (!Double.isNaN(thresholds[2]) && thresholds[2] > 0) oeThreshold2 = thresholds[2];
            if (!Double.isNaN(thresholds[3]) && thresholds[3] > 0) oeThreshold3 = thresholds[3];
        }
        if (thresholds != null && thresholds.length == 5) {
            if (!Double.isNaN(thresholds[0]) && thresholds[0] > 0) fdrsum1 = thresholds[0];
            if (!Double.isNaN(thresholds[1]) && thresholds[1] > 0) fdrsum2 = thresholds[1];
            if (!Double.isNaN(thresholds[2]) && thresholds[2] > 0) oeThreshold1 = thresholds[2];
            if (!Double.isNaN(thresholds[3]) && thresholds[3] > 0) oeThreshold2 = thresholds[3];
            if (!Double.isNaN(thresholds[4]) && thresholds[4] > 0) oeThreshold3 = thresholds[4];
        }
        if (thresholds != null && thresholds.length == 6) {
            if (!Double.isNaN(thresholds[0]) && thresholds[0] > 0) fdrsum1 = thresholds[0];
            if (!Double.isNaN(thresholds[1]) && thresholds[1] > 0) fdrsum2 = thresholds[1];
            if (!Double.isNaN(thresholds[2]) && thresholds[2] > 0) oeThreshold1 = thresholds[2];
            if (!Double.isNaN(thresholds[3]) && thresholds[3] > 0) oeThreshold2 = thresholds[3];
            if (!Double.isNaN(thresholds[4]) && thresholds[4] > 0) oeThreshold3 = thresholds[4];
            if (!Double.isNaN(thresholds[5]) && thresholds[5] > 0) oeThreshold3 = thresholds[5];
        }
    }

}