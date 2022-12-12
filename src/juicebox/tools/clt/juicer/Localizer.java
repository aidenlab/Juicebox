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

import com.google.common.primitives.Ints;
import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.data.ChromosomeHandler;
import juicebox.data.Dataset;
import juicebox.data.HiCFileTools;
import juicebox.data.MatrixZoomData;
import juicebox.data.anchor.GenericLocus;
import juicebox.data.anchor.GenericLocusTools;
import juicebox.data.anchor.MotifAnchorTools;
import juicebox.data.basics.Chromosome;
import juicebox.data.feature.GenomeWideList;
import juicebox.tools.clt.CommandLineParserForJuicer;
import juicebox.tools.clt.JuicerCLT;
import juicebox.tools.utils.juicer.apa.APARegionStatistics;
import juicebox.tools.utils.juicer.localizer.LocalizerUtils;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.Feature2DParser;
import juicebox.track.feature.FeatureFilter;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationHandler;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.RealMatrix;

import java.io.File;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

/**

 */
public class Localizer extends JuicerCLT {
    private String loopListPath;
    private File outputDirectory;
    private Dataset ds;
    private List<String> summedHiCFiles;

    //defaults
    // TODO right now these units are based on n*res/sqrt(2)
    // TODO the sqrt(2) scaling should be removed (i.e. handle scaling internally)
    private int window = 1;
    private int expandSize = 2500;
    private int numLocalizedPeaks = 1;
    private int[] resolutions = new int[]{100};
    private int[] regionWidths = new int[]{window};
    private boolean includeInterChr = false;
    private Feature2DList finalLoopList = new Feature2DList();
    private Feature2DList finalPrimaryLoopList = new Feature2DList();
    private GenomeWideList<GenericLocus> highResAnchorList = new GenomeWideList<>();
    private GenomeWideList<GenericLocus> highResAnchorPrimaryList = new GenomeWideList<>();
    private GenomeWideList<GenericLocus> upstreamAnchorList = new GenomeWideList<>();
    private GenomeWideList<GenericLocus> upstreamAnchorPrimaryList = new GenomeWideList<>();
    private GenomeWideList<GenericLocus> downstreamAnchorList = new GenomeWideList<>();
    private GenomeWideList<GenericLocus> downstreamAnchorPrimaryList = new GenomeWideList<>();

    /**
     * Usage for APA
     */
    public Localizer() {
        super("localizer [-n minval] [-x maxval] [-w window] [-r resolution(s)] [-c chromosomes]" +
                " [-k NONE/VC/VC_SQRT/KR] [-q corner_width] [-e include_inter_chr] [-u save_all_data]" +
                " <hicFile(s)> <PeaksFile> <SaveFolder>");
        HiCGlobals.useCache = false;
    }

    public static String getBasicUsage() {
        return "localizer <hicFile(s)> <PeaksFile> <SaveFolder>";
    }

    public void initializeDirectly(String inputHiCFileName, String inputPeaksFile, String outputDirectoryPath, int[] resolutions){
        this.resolutions=resolutions;

        List<String> summedHiCFiles = Arrays.asList(inputHiCFileName.split("\\+"));
        ds = HiCFileTools.extractDatasetForCLT(summedHiCFiles, true);
        this.loopListPath=inputPeaksFile;
        outputDirectory = HiCFileTools.createValidDirectory(outputDirectoryPath);

        //ds = HiCFileTools.extractDatasetForCLT(Arrays.asList(inputHiCFileName.split("\\+")), true);
        // outputDirectory = HiCFileTools.createValidDirectory(outputDirectoryPath);


    }

    @Override
    protected void readJuicerArguments(String[] args, CommandLineParserForJuicer juicerParser) {
        if (args.length != 4) {
            printUsageAndExit();
        }

        loopListPath = args[2];
        outputDirectory = HiCFileTools.createValidDirectory(args[3]);

        summedHiCFiles = Arrays.asList(args[1].split("\\+"));
        ds = HiCFileTools.extractDatasetForCLT(summedHiCFiles, true);

        NormalizationType preferredNorm = juicerParser.getNormalizationTypeOption(ds.getNormalizationHandler());
        if (preferredNorm != null)
            norm = preferredNorm;


        int potentialWindow = juicerParser.getAPAWindowSizeOption();
        if (potentialWindow > 0)
            window = potentialWindow;

        includeInterChr = juicerParser.getIncludeInterChromosomal();

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

        expandSize = juicerParser.getExpandSize();
        if (expandSize <= 0) {
            expandSize = 2500;
        }

        numLocalizedPeaks = juicerParser.getNumPeaks();
        if (numLocalizedPeaks <= 0) {
            numLocalizedPeaks = 1;
        }

    }

    @Override
    public void run() {

        //Calculate parameters that will need later
        for (final int resolution : HiCFileTools.filterResolutions(ds.getBpZooms(), resolutions)) {

            AtomicInteger[] gwPeakNumbers = {new AtomicInteger(0), new AtomicInteger(0), new AtomicInteger(0)};

            System.out.println("Processing Localizer for resolution " + resolution + " with window " + window);
            HiCZoom zoom = new HiCZoom(HiC.Unit.BP, resolution);

            ChromosomeHandler handler = ds.getChromosomeHandler();
            if (givenChromosomes != null)
                handler = HiCFileTools.stringToChromosomes(givenChromosomes, handler);

            //Load loop list for localizer
            Feature2DList loopList = Feature2DParser.loadFeatures(loopListPath, handler, true,
                    new FeatureFilter() {
                        // Remove duplicates and filters by size
                        // also save internal metrics for these measures
                        @Override
                        public List<Feature2D> filter(String chr, List<Feature2D> features) {

                            List<Feature2D> uniqueFeatures = new ArrayList<>(new HashSet<>(features));

                            return uniqueFeatures;
                        }
                    }, false);

            if (loopList.getNumTotalFeatures() > 0) {

                // output coarse anchors a la Rao & Huntley et al Cell 2014
                GenomeWideList<GenericLocus> featureAnchors = GenericLocusTools.extractAnchorsFromIntrachromosomalFeatures(loopList,
                        false, handler, expandSize);
                GenericLocusTools.updateOriginalFeatures(featureAnchors, "coarse");
                featureAnchors.simpleExport(new File(outputDirectory, "coarseLoopAnchors_"+expandSize+".bed"));


                double maxProgressStatus = handler.size();
                final AtomicInteger currentProgressStatus = new AtomicInteger(0);
                Map<Integer, Chromosome[]> chromosomePairs = new ConcurrentHashMap<>();
                int pairCounter = 1;
                for (Chromosome chr1 : handler.getChromosomeArrayWithoutAllByAll()) {
                    for (Chromosome chr2 : handler.getChromosomeArrayWithoutAllByAll()) {
                        Chromosome[] chromosomePair = {chr1,chr2};
                        chromosomePairs.put(pairCounter, chromosomePair);
                        pairCounter++;
                    }
                }
                final int chromosomePairCounter = pairCounter;

                // loop over chromosome pairs
                for (int i = 1; i < chromosomePairCounter; i++) {
                    Chromosome chr1 = chromosomePairs.get(i)[0];
                    Chromosome chr2 = chromosomePairs.get(i)[1];
                    if ((chr2.getIndex() > chr1.getIndex() && includeInterChr) || (chr2.getIndex() == chr1.getIndex())) {
                        System.out.println("processing: "+chr1.getName() + " " + chr2.getName());

                        //load zoom data
                        MatrixZoomData zd;
                        zd = HiCFileTools.getMatrixZoomData(ds, chr1, chr2, zoom);
                        if (zd == null) {
                            continue;
                        }
                        if (HiCGlobals.printVerboseComments) {
                            System.out.println("CHR " + chr1.getName() + " " + chr1.getIndex() + " CHR " + chr2.getName() + " " + chr2.getIndex());
                        }

                        //load loops for given chromosome pair
                        List<Feature2D> loops = loopList.get(chr1.getIndex(), chr2.getIndex());
                        if (loops == null || loops.size() == 0) {
                            if (HiCGlobals.printVerboseComments) {
                                System.out.println("CHR " + chr1.getName() + " CHR " + chr2.getName() + " - no loops, check loop filtering constraints");
                            }
                            continue;
                        }

                        //multithread loop processing, 100 loops per thread
                        int numOfLoopChunks = (loops.size() / 100) + 1;
                        int numOfLoops = loops.size();
                        final AtomicInteger loopChunk = new AtomicInteger(0);
                        ExecutorService executor = Executors.newFixedThreadPool(numCPUThreads);
                        for (int l = 0; l < numCPUThreads; l++) {
                            final int threadNum = l;
                            Runnable worker = new Runnable() {
                                @Override
                                public void run() {
                                    int threadChunk = loopChunk.getAndIncrement();
                                    while (threadChunk < numOfLoopChunks) {
                                        Instant A = Instant.now();
                                        // loop over loops in threadchunk, process one loop at a time
                                        for (int loopIndex = threadChunk * 100; loopIndex < Math.min(numOfLoops, (threadChunk + 1) * 100); loopIndex++) {
                                            Feature2D loop = loops.get(loopIndex);
                                            int newWindow = window;
                                            try {

                                                // set radius to search for localization, uses radius attribute from HiCCUPS, otherwise uses defaults
                                                int radius = 0;
                                                if (loop.containsAttributeKey("radius")) {
                                                    radius = (int) Float.parseFloat(loop.getAttribute("radius")) / resolution;
                                                }
                                                if (radius <= 0) {
                                                    radius = (int) (loop.getEnd1() - loop.getStart1()) / resolution;
                                                } else if (radius <= 2000 / resolution) {
                                                    radius = 2000 / resolution;
                                                }

                                                // load raw data and relevant portions of norm vector
                                                RealMatrix newData;
                                                List<List<Double>> normParts;
                                                newData = LocalizerUtils.extractLocalizedData(zd, loop, ((radius+newWindow) * 2) + 1, resolution, radius + newWindow, NormalizationHandler.NONE);
                                                synchronized(ds) {
                                                    normParts = LocalizerUtils.extractNormParts(ds, zd, loop, resolution, radius + newWindow, norm);
                                                }

                                                // find localization
                                                List<List<Double>> localizedPeaks = LocalizerUtils.localMax(newData, normParts.get(0), normParts.get(1), newWindow, numLocalizedPeaks, 0.05, outputDirectory);

                                                //code for expanding smoothing window in case localization not found
                                                //int counter = 1;
                                                //while (localizedPeaks.get(0).size() == 0 && counter < 2) {
                                                //    newWindow = window * (int) Math.pow(2,counter);
                                                //    newData = LocalizerUtils.extractLocalizedData(zd, loop, ((radius+newWindow) * 2) + 1, resolution, radius + newWindow, NormalizationHandler.NONE);
                                                //    synchronized(ds) {
                                                //        normParts = LocalizerUtils.extractNormParts(ds, zd, loop, resolution, radius + newWindow, norm);
                                                //    }
                                                //    localizedPeaks = LocalizerUtils.localMax(newData, normParts.get(0), normParts.get(1), newWindow, numLocalizedPeaks, 0.05, outputDirectory);
                                                //    counter++;
                                                //}

                                                // if localization not found, fill in localizer fields with NA
                                                if (localizedPeaks.get(0).size() == 0) {
                                                    Feature2D newLoop = loop.deepCopy();
                                                    newLoop.addStringAttribute("localX", "" + "NA");
                                                    newLoop.addStringAttribute("localY", "" + "NA");
                                                    newLoop.addStringAttribute("localPval", "" + "NA");
                                                    newLoop.addStringAttribute("localObserved", "" + "NA");
                                                    newLoop.addStringAttribute("localPeakID", "" + "NA");
                                                    newLoop.addStringAttribute("localPeakZ", "" + "NA");
                                                    newLoop.addStringAttribute("highRes_start_1", "" + "NA");
                                                    newLoop.addStringAttribute("highRes_end_1", "" + "NA");
                                                    newLoop.addStringAttribute("highRes_start_2", "" + "NA");
                                                    newLoop.addStringAttribute("highRes_end_2", "" + "NA");
                                                    newLoop.addStringAttribute("upstream_start_1", "" + "NA");
                                                    newLoop.addStringAttribute("upstream_end_1", "" + "NA");
                                                    newLoop.addStringAttribute("downstream_start_2", "" + "NA");
                                                    newLoop.addStringAttribute("downstream_end_2", "" + "NA");
                                                    Feature2D newPrimaryLoop = newLoop.deepCopy();
                                                    synchronized (finalPrimaryLoopList) {
                                                        finalPrimaryLoopList.add(chr1.getIndex(), chr2.getIndex(), newPrimaryLoop);
                                                    }
                                                    // if number of requested localized peaks is greater than 1, create a separate list given potential multiple entries
                                                    if (numLocalizedPeaks > 1) {
                                                        synchronized (finalLoopList) {
                                                            finalLoopList.add(chr1.getIndex(), chr2.getIndex(), newLoop);
                                                        }
                                                    }
                                                }

                                                // create features for localized peaks
                                                for (int peak = 0; peak < localizedPeaks.get(0).size(); peak++) {
                                                    Feature2D newLoop = loop.deepCopy();
                                                    long localPeakX = newLoop.getMidPt1() / resolution - radius - newWindow + localizedPeaks.get(0).get(peak).intValue();
                                                    long localPeakY = newLoop.getMidPt2() / resolution - radius - newWindow + localizedPeaks.get(1).get(peak).intValue();
                                                    double localPeakP = localizedPeaks.get(2).get(peak);
                                                    double localPeakO = localizedPeaks.get(3).get(peak);
                                                    double localPeakZ = localizedPeaks.get(4).get(peak);
                                                    newLoop.addStringAttribute("localX", "" + (localPeakX * resolution));
                                                    newLoop.addStringAttribute("localY", "" + (localPeakY * resolution));
                                                    newLoop.addStringAttribute("localPval", "" + localPeakP);
                                                    newLoop.addStringAttribute("localObserved", "" + localPeakO);
                                                    newLoop.addStringAttribute("localPeakZ", "" + localPeakZ);
                                                    newLoop.addStringAttribute("localPeakID", "" + peak);
                                                    Feature2D newPrimaryLoop = newLoop.deepCopy();
                                                    List<Feature2D> originalFeatures = new ArrayList<>();
                                                    originalFeatures.add(newLoop);
                                                    List<Feature2D> originalPrimaryFeatures = new ArrayList<>();
                                                    originalPrimaryFeatures.add(newPrimaryLoop);
                                                    List<Feature2D> emptyList = new ArrayList<>();
                                                    if (peak == 0) {
                                                        GenericLocus primaryAnchor1 = new GenericLocus(newPrimaryLoop.getChr1(), (long) (localPeakX-newWindow-0.5)*resolution, (long) (localPeakX+newWindow+0.5)*resolution, originalPrimaryFeatures, emptyList);
                                                        GenericLocus primaryAnchor2 = new GenericLocus(newPrimaryLoop.getChr2(), (long) (localPeakY-newWindow-0.5)*resolution, (long) (localPeakY+newWindow+0.5)*resolution, emptyList, originalPrimaryFeatures);
                                                        synchronized (finalPrimaryLoopList) {
                                                            finalPrimaryLoopList.add(chr1.getIndex(), chr2.getIndex(), newPrimaryLoop);
                                                        }
                                                        synchronized (highResAnchorPrimaryList) {
                                                            highResAnchorPrimaryList.addFeature(newLoop.getChr1(), primaryAnchor1);
                                                            highResAnchorPrimaryList.addFeature(newLoop.getChr2(), primaryAnchor2);
                                                        }
                                                        synchronized (upstreamAnchorPrimaryList) {
                                                            upstreamAnchorPrimaryList.addFeature(newLoop.getChr1(), primaryAnchor1);
                                                        }
                                                        synchronized (downstreamAnchorPrimaryList) {
                                                            downstreamAnchorPrimaryList.addFeature(newLoop.getChr2(), primaryAnchor2);
                                                        }
                                                    }
                                                    if (numLocalizedPeaks>1) {
                                                        GenericLocus anchor1 = new GenericLocus(newLoop.getChr1(), (long) (localPeakX-newWindow-0.5)*resolution, (long) (localPeakX+newWindow+0.5)*resolution, originalFeatures, emptyList);
                                                        GenericLocus anchor2 = new GenericLocus(newLoop.getChr2(), (long) (localPeakY-newWindow-0.5)*resolution, (long) (localPeakY+newWindow+0.5)*resolution, emptyList, originalFeatures);
                                                        synchronized (finalLoopList) {
                                                            finalLoopList.add(chr1.getIndex(), chr2.getIndex(), newLoop);
                                                        }
                                                        synchronized (highResAnchorList) {
                                                            highResAnchorList.addFeature(newLoop.getChr1(), anchor1);
                                                            highResAnchorList.addFeature(newLoop.getChr2(), anchor2);
                                                        }
                                                        synchronized (upstreamAnchorList) {
                                                            upstreamAnchorList.addFeature(newLoop.getChr1(), anchor1);
                                                        }
                                                        synchronized (downstreamAnchorList) {
                                                            downstreamAnchorList.addFeature(newLoop.getChr2(), anchor2);
                                                        }
                                                    }

                                                }

                                            } catch (Exception e) {
                                                System.err.println(e);
                                                e.printStackTrace();
                                                System.err.println("Unable to find data for loop: " + loop);
                                            }
                                        }

                                        int reasonableDivisor = Math.max(numOfLoopChunks / 20, 1);
                                        //if (HiCGlobals.printVerboseComments || threadChunk % reasonableDivisor == 0) {
                                        //    DecimalFormat df = new DecimalFormat("#.####");
                                        //    df.setRoundingMode(RoundingMode.FLOOR);
                                        //    System.out.println(df.format(Math.floor((100.0 * threadChunk) / numOfLoopChunks)) + "% ");
                                        //}
                                        threadChunk = loopChunk.getAndIncrement();
                                    }
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

                System.out.println("Exporting localizer results...");

                // output primary list
                GenericLocusTools.callMergeAnchors(highResAnchorPrimaryList);
                GenericLocusTools.updateOriginalFeatures(highResAnchorPrimaryList, "highRes");
                GenericLocusTools.callMergeAnchors(upstreamAnchorPrimaryList);
                GenericLocusTools.updateOriginalFeatures(upstreamAnchorPrimaryList, "upstream");
                GenericLocusTools.callMergeAnchors(downstreamAnchorPrimaryList);
                GenericLocusTools.updateOriginalFeatures(downstreamAnchorPrimaryList, "downstream");
                highResAnchorPrimaryList.simpleExport(new File(outputDirectory, "highRes_primary_loopAnchors.bed"));
                upstreamAnchorPrimaryList.simpleExport(new File(outputDirectory, "upstream_primary_loopAnchors.bed"));
                downstreamAnchorPrimaryList.simpleExport(new File(outputDirectory, "downstream_primary_loopAnchors.bed"));
                finalPrimaryLoopList.exportFeatureList(new File(outputDirectory, "localizedList_primary_"+resolution+".bedpe"), true, Feature2DList.ListFormat.LOCALIZED);

                // output secondary list if number of requested localized peaks > 1
                if (numLocalizedPeaks > 1) {
                    GenericLocusTools.callMergeAnchors(highResAnchorList);
                    GenericLocusTools.updateOriginalFeatures(highResAnchorList, "highRes");
                    GenericLocusTools.callMergeAnchors(upstreamAnchorList);
                    GenericLocusTools.updateOriginalFeatures(upstreamAnchorList, "upstream");
                    GenericLocusTools.callMergeAnchors(downstreamAnchorList);
                    GenericLocusTools.updateOriginalFeatures(downstreamAnchorList, "downstream");
                    highResAnchorList.simpleExport(new File(outputDirectory, "highRes_loopAnchors.bed"));
                    finalLoopList.exportFeatureList(new File(outputDirectory, "localizedList_" + resolution + ".bedpe"), true, Feature2DList.ListFormat.LOCALIZED);
                }
            } else {
                System.err.println("Loop list is empty or incorrect path provided.");
                System.exit(3);
            }
        }
        System.out.println("Localizer complete");
        //if no data return null
    }
}