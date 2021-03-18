/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2021 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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
 *  FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.tools.clt.juicer;

import com.google.common.primitives.Ints;
import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.data.ChromosomeHandler;
import juicebox.data.Dataset;
import juicebox.data.HiCFileTools;
import juicebox.data.MatrixZoomData;
import juicebox.data.basics.Chromosome;
import juicebox.tools.clt.CommandLineParserForJuicer;
import juicebox.tools.clt.JuicerCLT;
import juicebox.tools.utils.juicer.apa.APADataStack;
import juicebox.tools.utils.juicer.apa.APARegionStatistics;
import juicebox.tools.utils.juicer.apa.APAUtils;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.Feature2DParser;
import juicebox.track.feature.FeatureFilter;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.RealMatrix;

import java.io.File;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Aggregate Peak Analysis developed by mhuntley
 * <p/>
 * Implemented in Juicer by mshamim
 * <p/>
 * ---
 * APA
 * ---
 * The "apa" command takes three required arguments and a number of optional
 * arguments.
 * <p/>
 * apa [-n minval] [-x maxval] [-w window]  [-r resolution(s)] [-c chromosome(s)]
 * [-k NONE/VC/VC_SQRT/KR] <HiC file(s)> <PeaksFile> <SaveFolder> [SavePrefix]
 * <p/>
 * The required arguments are:
 * <p/>
 * <hic file(s)>: Address of HiC File(s) which should end with ".hic". This is the file you will
 * load into Juicebox. URLs or local addresses may be used. To sum multiple HiC Files together,
 * use the '+' symbol between the addresses (no whitespace between addresses)
 * <PeaksFile>: List of peaks in standard 2D feature format (chr1 x1 x2 chr2 y1 y2 color ...)
 * <SaveFolder>: Working directory where outputs will be saved
 * <p/>
 * The optional arguments are:
 * -n <int> minimum distance away from the diagonal. Used to filter peaks too close to the diagonal.
 * Units are in terms of the provided resolution. (e.g. -n 30 @ resolution 5kB will filter loops
 * within 30*(5000/sqrt(2)) units of the diagonal)
 * -x <int> maximum distance away from the diagonal. Used to filter peaks too far from the diagonal.
 * Units are in terms of the provided resolution. (e.g. -n 30 @ resolution 5kB will filter loops
 * further than 30*(5000/sqrt(2)) units of the diagonal)
 * -r <int(s)> resolution for APA; multiple resolutions can be specified using commas (e.g. 5000,10000)
 * -c <String(s)> Chromosome(s) on which APA will be run. The number/letter for the chromosome can be
 * used with or without appending the "chr" string. Multiple chromosomes can be specified using
 * commas (e.g. 1,chr2,X,chrY)
 * -k <NONE/VC/VC_SQRT/KR> Normalizations (case sensitive) that can be selected. Generally,
 * KR (Knight-Ruiz) balancing should be used when available.
 * <p/>
 * Default settings of optional arguments:
 * -n 30
 * -x (infinity)
 * -r 25000,10000
 * -c (all_chromosomes)
 * -k KR
 * <p/>
 * ------------
 * APA Examples
 * ------------
 * <p/>
 * apa HIC006.hic all_loops.txt results1
 * > This command will run APA on HIC006 using loops from the all_loops files
 * > and save them under the results1 folder.
 * <p/>
 * apa https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic
 * all_loops.txt results1
 * > This command will run APA on the GM12878 mega map using loops from the all_loops
 * > files and save them under the results1 folder.
 * <p/>
 * apa -r 10000,5000 -c 17,18 HIC006.hic+HIC007.hic all_loops.txt results
 * > This command will run APA at 50 kB resolution on chromosomes 17 and 18 for the
 * > summed HiC maps (HIC006 and HIC007) using loops from the all_loops files
 * > and save them under the results folder
 */
public class APA extends JuicerCLT {
    private boolean saveAllData = false;
    private boolean dontIncludePlots = false;
    private String loopListPath;
    private File outputDirectory;
    private Dataset ds;

    //defaults
    // TODO right now these units are based on n*res/sqrt(2)
    // TODO the sqrt(2) scaling should be removed (i.e. handle scaling internally)
    private double minPeakDist = 30; // distance between two bins, can be changed in opts
    private double maxPeakDist = Double.POSITIVE_INFINITY;
    private int window = 10;
    private int[] resolutions = new int[]{25000, 10000, 5000};
    private int[] regionWidths = new int[]{6, 6, 3};
    private boolean includeInterChr = false;
    private final Object key = new Object();
    private boolean aggregateNormalization = false;

    /**
     * Usage for APA
     */
    public APA() {
        super("apa [-n minval] [-x maxval] [-w window] [-r resolution(s)] [-c chromosomes]" +
                " [-k NONE/VC/VC_SQRT/KR] [-q corner_width] [-e include_inter_chr] [-u save_all_data]" +
                " <hicFile(s)> <PeaksFile> <SaveFolder>");
        HiCGlobals.useCache = false;
    }

    public static String getBasicUsage() {
        return "apa <hicFile(s)> <PeaksFile> <SaveFolder>";
    }

    public void initializeDirectly(String inputHiCFileName, String inputPeaksFile, String outputDirectoryPath, int[] resolutions,double
            minPeakDist, double maxPeakDist){
        this.resolutions=resolutions;

        List<String> summedHiCFiles = Arrays.asList(inputHiCFileName.split("\\+"));
        ds = HiCFileTools.extractDatasetForCLT(summedHiCFiles, true);
        this.loopListPath=inputPeaksFile;
        outputDirectory = HiCFileTools.createValidDirectory(outputDirectoryPath);

        this.minPeakDist=minPeakDist;
        this.maxPeakDist=maxPeakDist;
    }

    @Override
    protected void readJuicerArguments(String[] args, CommandLineParserForJuicer juicerParser) {
        if (args.length != 4) {
            printUsageAndExit();
        }

        loopListPath = args[2];
        outputDirectory = HiCFileTools.createValidDirectory(args[3]);

        List<String> summedHiCFiles = Arrays.asList(args[1].split("\\+"));
        ds = HiCFileTools.extractDatasetForCLT(summedHiCFiles, true);

        NormalizationType preferredNorm = juicerParser.getNormalizationTypeOption(ds.getNormalizationHandler());
        if (preferredNorm != null)
            norm = preferredNorm;

        double potentialMinPeakDist = juicerParser.getAPAMinVal();
        if (potentialMinPeakDist > -1)
            minPeakDist = potentialMinPeakDist;

        double potentialMaxPeakDist = juicerParser.getAPAMaxVal();
        if (potentialMaxPeakDist > -1)
            maxPeakDist = potentialMaxPeakDist;

        int potentialWindow = juicerParser.getAPAWindowSizeOption();
        if (potentialWindow > 0)
            window = potentialWindow;

        includeInterChr = juicerParser.getIncludeInterChromosomal();

        saveAllData = juicerParser.getAPASaveAllData();

        dontIncludePlots = juicerParser.getAPADontIncludePlots();

        List<String> possibleRegionWidths = juicerParser.getAPACornerRegionDimensionOptions();
        if (possibleRegionWidths != null) {
            List<Integer> widths = new ArrayList<>();
            for (String res : possibleRegionWidths) {
                widths.add(Integer.parseInt(res));
            }
            regionWidths = Ints.toArray(widths);
        }

        List<String> possibleResolutions = juicerParser.getMultipleResolutionOptions();
        if (possibleResolutions != null) {
            List<Integer> intResolutions = new ArrayList<>();
            for (String res : possibleResolutions) {
                intResolutions.add(Integer.parseInt(res));
            }
            resolutions = Ints.toArray(intResolutions);
        }

        numCPUThreads = juicerParser.getNumThreads();

        aggregateNormalization = juicerParser.getAggregateNormalization();
    }

    @Override
    public void run() {
        runWithReturn();
    }


    public APARegionStatistics runWithReturn() {

        APARegionStatistics result = null;

        //Calculate parameters that will need later
        int L = 2 * window + 1;
        for (final int resolution : HiCFileTools.filterResolutions(ds.getBpZooms(), resolutions)) {

            AtomicInteger[] gwPeakNumbers = {new AtomicInteger(0), new AtomicInteger(0), new AtomicInteger(0)};
            //Arrays.fill(gwPeakNumbers, 0);

            // determine the region width corresponding to the resolution
            int currentRegionWidth = resolution == 5000 ? 3 : 6;
            try {
                if (regionWidths != null && regionWidths.length > 0) {
                    for (int i = 0; i < resolutions.length; i++) {
                        if (resolutions[i] == resolution) {
                            currentRegionWidth = regionWidths[i];
                        }
                    }
                }
            } catch (Exception e) {
                currentRegionWidth = resolution == 5000 ? 3 : 6;
            }
            final int finalCurrentRegionWidth = currentRegionWidth;

            System.out.println("Processing APA for resolution " + resolution);
            HiCZoom zoom = new HiCZoom(HiC.Unit.BP, resolution);

            ChromosomeHandler handler = ds.getChromosomeHandler();
            if (givenChromosomes != null)
                handler = HiCFileTools.stringToChromosomes(givenChromosomes, handler);

            // Metrics resulting from apa filtering
            final Map<String, Integer[]> filterMetrics = new HashMap<>();
            //looplist is empty here why??
            Feature2DList loopList = Feature2DParser.loadFeatures(loopListPath, handler, false,
                    new FeatureFilter() {
                        // Remove duplicates and filters by size
                        // also save internal metrics for these measures
                        @Override
                        public List<Feature2D> filter(String chr, List<Feature2D> features) {

                            List<Feature2D> uniqueFeatures = new ArrayList<>(new HashSet<>(features));
                            List<Feature2D> filteredUniqueFeatures = APAUtils.filterFeaturesBySize(uniqueFeatures,
                                    minPeakDist, maxPeakDist, resolution);

                            filterMetrics.put(chr,
                                    new Integer[]{filteredUniqueFeatures.size(), uniqueFeatures.size(), features.size()});

                            return filteredUniqueFeatures;
                        }
                    }, false);

            if (loopList.getNumTotalFeatures() > 0) {

                double maxProgressStatus = handler.size();
                final AtomicInteger currentProgressStatus = new AtomicInteger(0);
                Map<Integer,Chromosome[]> chromosomePairs = new ConcurrentHashMap<>();
                int pairCounter = 1;
                for (Chromosome chr1 : handler.getChromosomeArrayWithoutAllByAll()) {
                    for (Chromosome chr2 : handler.getChromosomeArrayWithoutAllByAll()) {
                        Chromosome[] chromosomePair = {chr1,chr2};
                        chromosomePairs.put(pairCounter, chromosomePair);
                        pairCounter++;
                    }
                }
                final int chromosomePairCounter = pairCounter;

                APADataStack.initializeDataSaveFolder(outputDirectory,"" + resolution);

                for (int chrPair = 1; chrPair < chromosomePairCounter; chrPair++) {
                    Chromosome chr1 = chromosomePairs.get(chrPair)[0];
                    Chromosome chr2 = chromosomePairs.get(chrPair)[1];
                    if ((chr2.getIndex() > chr1.getIndex() && includeInterChr) || (chr2.getIndex() == chr1.getIndex())) {
                        MatrixZoomData zd;
                        zd = HiCFileTools.getMatrixZoomData(ds, chr1, chr2, zoom);

                        if (zd == null) {
                            continue;
                        }
                        if (HiCGlobals.printVerboseComments) {
                            System.out.println("CHR " + chr1.getName() + " " + chr1.getIndex() + " CHR " + chr2.getName() + " " + chr2.getIndex());
                        }
                        List<Feature2D> loops = loopList.get(chr1.getIndex(), chr2.getIndex());
                        if (loops == null || loops.size() == 0) {
                            if (HiCGlobals.printVerboseComments) {
                                System.out.println("CHR " + chr1.getName() + " CHR " + chr2.getName() + " - no loops, check loop filtering constraints");
                            }
                            continue;
                        }

                        int numOfLoopChunks = (loops.size() / 2) + 1;
                        int numOfLoops = loops.size();
                        final AtomicInteger loopChunk = new AtomicInteger(0);
                        Integer[] peakNumbers = filterMetrics.get(Feature2DList.getKey(chr1, chr2));

                        if (loops.size() != peakNumbers[0]) {
                            System.err.println("Error reading statistics from " + chr1 + chr2);
                        }

                        for (int i = 0; i < peakNumbers.length; i++) {
                            gwPeakNumbers[i].addAndGet(peakNumbers[i]);
                        }

                        ExecutorService executor = Executors.newFixedThreadPool(numCPUThreads);
                        for (int l = 0; l < numCPUThreads; l++) {
                            final int threadChrPair = chrPair;
                            Runnable worker = new Runnable() {
                                @Override
                                public void run() {
                                    APADataStack apaDataStack = new APADataStack(L, chromosomePairCounter, aggregateNormalization);
                                    int threadChunk = loopChunk.getAndIncrement();
                                    while (threadChunk < numOfLoopChunks) {
                                        for (int loopIndex = threadChunk * 2; loopIndex < Math.min(numOfLoops, (threadChunk + 1) * 2); loopIndex++) {
                                            Feature2D loop = loops.get(loopIndex);
                                            try {
                                                RealMatrix newData;
                                                RealMatrix newExpectedData;
                                                newData = APAUtils.extractLocalizedData(zd, loop, L, resolution, window, norm);
                                                //newExpectedData = APAUtils.extractLocalizedExpectedData(df, chr1, loop, L, resolution, window);
                                                apaDataStack.addData(newData);
                                                if (aggregateNormalization) {
                                                    List<RealMatrix> newVectors = APAUtils.extractLocalizedRowSums(zd, loop, L, resolution, window, norm);
                                                    apaDataStack.addRowSums(newVectors);
                                                }
                                                //apaDataStack.addExpectedData(newExpectedData);
                                                //apaDataStack.addData(APAUtils.extractLocalizedData(zd, loop, L, resolution, window, norm));
                                            } catch (Exception e) {
                                                System.err.println(e);
                                                System.err.println("Unable to find data for loop: " + loop);
                                            }
                                        }
                                        threadChunk = loopChunk.getAndIncrement();
                                    }
                                    synchronized (ds) {
                                        apaDataStack.updateGenomeWideData();
                                    }

                                    apaDataStack.updateChromosomeWideData(threadChrPair);
                                }
                            };
                            executor.execute(worker);
                        }
                        executor.shutdown();

                        // Wait until all threads finish
                        while (!executor.isTerminated()) {
                        }

                        if (saveAllData) {
                            APADataStack.exportChromosomeData(chr1.getName() + 'v' + chr2.getName(), peakNumbers, finalCurrentRegionWidth, saveAllData, dontIncludePlots, chrPair);
                        }
                    }
                    if (chr2.getIndex() == chr1.getIndex()) {
                        System.out.print(((int) Math.floor((100.0 * currentProgressStatus.incrementAndGet()) / maxProgressStatus)) + "% ");
                    }
                }

                System.out.println("Exporting APA results...");
                //save data as int array
                result= APADataStack.retrieveDataStatistics(currentRegionWidth); //should retrieve data
                Integer[] gwPeakNumbersArray = {gwPeakNumbers[0].get(),gwPeakNumbers[1].get(),gwPeakNumbers[2].get()};
                APADataStack.exportGenomeWideData(gwPeakNumbersArray, currentRegionWidth, saveAllData, dontIncludePlots);
                APADataStack.clearAllData();
            } else {
                System.err.println("Loop list is empty or incorrect path provided.");
                System.exit(3);
            }
        }
        System.out.println("APA complete");
        return result;
        //if no data return null
    }
}