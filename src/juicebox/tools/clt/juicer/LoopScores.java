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
import juicebox.data.*;
import juicebox.data.anchor.MotifAnchor;
import juicebox.data.anchor.MotifAnchorParser;
import juicebox.data.basics.Chromosome;
import juicebox.data.basics.ListOfDoubleArrays;
import juicebox.data.feature.GenomeWideList;
import juicebox.tools.clt.CommandLineParserForJuicer;
import juicebox.tools.clt.JuicerCLT;
import juicebox.tools.utils.juicer.apa.APAUtils;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.Feature2DParser;
import juicebox.track.feature.FeatureFilter;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.RealMatrix;

import java.awt.*;
import java.io.File;
import java.util.*;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

public class LoopScores extends JuicerCLT {
    private File outputDirectory;
    private String loopListPath;
    private Dataset ds;
    private ChromosomeHandler chromosomeHandler;
    private Set<String> includedChromosomes;
    private int chromosomePairCounter = 0;
    private Map<Integer, Chromosome[]> chromosomePairs = new ConcurrentHashMap<>();

    //defaults
    private double minDist = 20;
    private double maxDist = 8000;
    private long normDist = 10000;
    private double scaling = -1.0;
    private int[] resolutions = new int[]{1000};
    private boolean includeDistCorrection = false;
    private boolean generateLoopsFromAnchors = false;
    private int windowCPU = 10;
    private int matrixSizeCPU = 512;
    private int peakWidthCPU = 5;
    private Feature2DList finalLoopList = new Feature2DList();


    public LoopScores() {
        super("loopScores --anchors <hicFile(s)> <loopList> <SaveFolder>");
        HiCGlobals.useCache = false;
    }

    public static String getBasicUsage() {
        return "loopScores <hicFile(s)> <loopList> <SaveFolder>";
    }

    public void setIncludedChromosomes(List<String> includedChromosomes) {
        this.includedChromosomes = Collections.synchronizedSet(new HashSet<>());
        if (includedChromosomes != null && includedChromosomes.size() > 0) {
            for (String name : includedChromosomes) {
                this.includedChromosomes.add(chromosomeHandler.cleanUpName(name));
            }
        } else {
            for (Chromosome chr : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {
                this.includedChromosomes.add(chr.getName());
            }
        }
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
        chromosomeHandler = ds.getChromosomeHandler();

        setIncludedChromosomes(givenChromosomes);

        NormalizationType preferredNorm = juicerParser.getNormalizationTypeOption(ds.getNormalizationHandler());
        if (preferredNorm != null)
            norm = preferredNorm;

        double potentialMinDist = juicerParser.getAPAMinVal();
        if (potentialMinDist > -1)
            minDist = potentialMinDist;

        double potentialMaxDist = juicerParser.getAPAMaxVal();
        if (potentialMaxDist > -1)
            maxDist = potentialMaxDist;

        List<String> possibleResolutions = juicerParser.getMultipleResolutionOptions();
        if (possibleResolutions != null) {
            List<Integer> intResolutions = new ArrayList<>();
            for (String res : possibleResolutions) {
                intResolutions.add(Integer.parseInt(res));
            }
            resolutions = Ints.toArray(intResolutions);
        }

        updateNumberOfCPUThreads(juicerParser);

        generateLoopsFromAnchors = juicerParser.getLoopAnchorsOption();

    }

    @Override
    public void run() {
        int buffer_width = HiCCUPS.regionMargin;
        int L = buffer_width * 2 + 1;
        for (final int resolution : HiCFileTools.filterResolutions(ds.getBpZooms(), resolutions)) {
            HiCZoom zoom = new HiCZoom(HiC.Unit.BP, resolution);
            matricesToProcess(false);

            ChromosomeHandler handler = chromosomeHandler;
            if (givenChromosomes != null)
                handler = HiCFileTools.stringToChromosomes(givenChromosomes, handler);

            Feature2DList loopList;
            final Map<String, Integer[]> filterMetrics = new HashMap<>();
            if (!generateLoopsFromAnchors) {
                // Metrics resulting from apa filtering
                //looplist is empty here why??
                loopList = Feature2DParser.loadFeatures(loopListPath, handler, false,
                        new FeatureFilter() {
                            // Remove duplicates and filters by size
                            // also save internal metrics for these measures
                            @Override
                            public List<Feature2D> filter(String chr, List<Feature2D> features) {

                                List<Feature2D> uniqueFeatures = new ArrayList<>(new HashSet<>(features));
                                List<Feature2D> filteredUniqueFeatures = APAUtils.filterFeaturesBySize(uniqueFeatures,
                                        minDist, maxDist, resolution);

                                filterMetrics.put(chr,
                                        new Integer[]{filteredUniqueFeatures.size(), uniqueFeatures.size(), features.size()});

                                return filteredUniqueFeatures;
                            }
                        }, false);
            } else {
                GenomeWideList<MotifAnchor> anchorList = MotifAnchorParser.loadFromBEDFile(handler, loopListPath);
                Feature2DList tempList = new Feature2DList();
                for (Chromosome chr : handler.getChromosomeArray()) {
                    //System.out.println("test1");
                    if (anchorList.getFeatures(chr.getName()) != null) {
                        //System.out.println("test2");
                        for (MotifAnchor i : anchorList.getFeatures(chr.getName())) {
                            for (MotifAnchor j : anchorList.getFeatures(chr.getName())) {
                                //System.out.println("test3");
                                if (j.getX1() > i.getX1() && Math.abs(j.getX1() - i.getX1()) < (maxDist*resolution)) {
                                    //System.out.println("test4");
                                    tempList.add(chr.getIndex(), chr.getIndex(), new Feature2D(Feature2D.FeatureType.PEAK, i.getChr(), i.getX1(), i.getX2(), j.getChr(), j.getX1(), j.getX2(), Color.BLACK, new LinkedHashMap<>()));
                                }
                            }
                        }
                    }
                }
                tempList.filterLists(new FeatureFilter() {// Remove duplicates and filters by size
                    // also save internal metrics for these measures
                    @Override
                    public List<Feature2D> filter(String chr, List<Feature2D> features) {

                        List<Feature2D> uniqueFeatures = new ArrayList<>(new HashSet<>(features));
                        List<Feature2D> filteredUniqueFeatures = APAUtils.filterFeaturesBySize(uniqueFeatures,
                                minDist, maxDist, resolution);

                        filterMetrics.put(chr,
                                new Integer[]{filteredUniqueFeatures.size(), uniqueFeatures.size(), features.size()});

                        return filteredUniqueFeatures;
                    }
                });
                loopList = tempList;

            }

            if (loopList.getNumTotalFeatures() > 0) {

                double maxProgressStatus = handler.size();
                final AtomicInteger currentProgressStatus = new AtomicInteger(0);
                for (int chrPair = 1; chrPair < chromosomePairCounter; chrPair++) {
                    Chromosome chr1 = chromosomePairs.get(chrPair)[0];
                    Chromosome chr2 = chromosomePairs.get(chrPair)[1];
                    if (chr2.getIndex() == chr1.getIndex()) {
                        System.out.println(chr1.getName());
                        List<Feature2D> loops = loopList.get(chr1.getIndex(), chr2.getIndex());
                        if (loops == null || loops.size() == 0) {
                            if (HiCGlobals.printVerboseComments) {
                                System.out.println("CHR " + chr1.getName() + " CHR " + chr2.getName() + " - no loops, check loop filtering constraints");
                            }
                            continue;
                        }

                        MatrixZoomData zd;
                        zd = HiCFileTools.getMatrixZoomData(ds, chr1, chr2, zoom);

                        if (zd == null) {
                            continue;
                        }

                        NormalizationVector nv1 = ds.getNormalizationVector(chr1.getIndex(), zoom, norm);
                        NormalizationVector nv2 = ds.getNormalizationVector(chr2.getIndex(), zoom, norm);

                        ListOfDoubleArrays expected = HiCFileTools.extractChromosomeExpectedVector(ds, chr1.getIndex(), zoom, norm);

                        ListOfDoubleArrays distScaling = new ListOfDoubleArrays(chr1.getLength() / resolution - normDist / resolution);

                        initializeDistScaling(zd, distScaling, (int) normDist / resolution);

                        if (HiCGlobals.printVerboseComments) {
                            System.out.println("CHR " + chr1.getName() + " " + chr1.getIndex() + " CHR " + chr2.getName() + " " + chr2.getIndex());
                        }

                        int numOfLoopChunks = (loops.size() / 2) + 1;
                        int numOfLoops = loops.size();
                        final AtomicInteger loopChunk = new AtomicInteger(0);
                        Integer[] peakNumbers = filterMetrics.get(Feature2DList.getKey(chr1, chr2));

                        if (loops.size() != peakNumbers[0]) {
                            System.err.println("Error reading statistics from " + chr1 + chr2);
                        }

                        ExecutorService executor = Executors.newFixedThreadPool(numCPUThreads);
                        for (int l = 0; l < numCPUThreads; l++) {
                            final int threadChrPair = chrPair;
                            Runnable worker = new Runnable() {
                                @Override
                                public void run() {
                                    int threadChunk = loopChunk.getAndIncrement();
                                    while (threadChunk < numOfLoopChunks) {
                                        for (int loopIndex = threadChunk * 2; loopIndex < Math.min(numOfLoops, (threadChunk + 1) * 2); loopIndex++) {
                                            Feature2D loop = loops.get(loopIndex);
                                            try {
                                                RealMatrix newData;
                                                newData = APAUtils.extractLocalizedData(zd, loop, L, resolution, buffer_width, norm);
                                                long binX = loop.getMidPt1() / resolution;
                                                long binY = loop.getMidPt2() / resolution;
                                                long diff = binX - binY;
                                                float[] HiCCUPSScores = calculateHiCCUPSScores(newData, expected, (float) nv1.getData().get(binX), (float) nv2.getData().get(binY), (int) diff);
                                                float[] extrusionScores = calculateBottomLeftExtrusionScore(newData, expected, (float) nv1.getData().get(binX), (float) nv2.getData().get(binY), (int) diff, distScaling, binX, binY, resolution);
                                                Feature2D newLoop = loop.deepCopy();
                                                newLoop.addStringAttribute("observed", "" + (newData.getEntry(buffer_width,buffer_width)*nv1.getData().get(binX)*nv2.getData().get(binY)));
                                                newLoop.addStringAttribute("expectedBL", "" + HiCCUPSScores[0]);
                                                newLoop.addStringAttribute("expectedDonut", "" + HiCCUPSScores[1]);
                                                newLoop.addStringAttribute("expectedH", "" + HiCCUPSScores[2]);
                                                newLoop.addStringAttribute("expectedV", "" + HiCCUPSScores[3]);
                                                newLoop.addStringAttribute("centerSum", "" + extrusionScores[0]);
                                                newLoop.addStringAttribute("leftSum", "" + extrusionScores[1]);
                                                newLoop.addStringAttribute("downSum", "" + extrusionScores[2]);
                                                newLoop.addStringAttribute("centerSumDC", "" + extrusionScores[3]);
                                                newLoop.addStringAttribute("leftSumDC", "" + extrusionScores[4]);
                                                newLoop.addStringAttribute("downSumDC", "" + extrusionScores[5]);
                                                synchronized (finalLoopList) {
                                                    finalLoopList.add(chr1.getIndex(), chr2.getIndex(), newLoop);
                                                }
                                            } catch (Exception e) {
                                                System.err.println(e);
                                                System.err.println("Unable to find data for loop: " + loop);
                                            }

                                        }
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

                System.out.println("Exporting score results...");
                finalLoopList.exportFeatureList(new File(outputDirectory, "localizedList_"+resolution+".bedpe"), true, Feature2DList.ListFormat.SCORES);
            }
        }
    }

    private void matricesToProcess(boolean includeInter) {
        int pairCounter = 1;
        for (Chromosome chr1 : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {
            for (Chromosome chr2 : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {
                Chromosome[] chromosomePair = {chr1, chr2};
                if (chr1.equals(chr2) || (includeInter && chr1.getIndex() < chr2.getIndex())) {
                    if (includedChromosomes.contains(chr1.getName()) || includedChromosomes.contains(chr2.getName())) {
                        chromosomePairs.put(pairCounter, chromosomePair);
                        pairCounter++;
                    }
                }

            }
        }
        this.chromosomePairCounter = pairCounter;
    }

    private float[] calculateHiCCUPSScores(RealMatrix subMatrix, ListOfDoubleArrays expected, float norm1, float norm2, int diff) {
        float Evalue_bl = 0;
        float Edistvalue_bl = 0;
        float Evalue_donut = 0;
        float Edistvalue_donut = 0;
        float Evalue_h = 0;
        float Edistvalue_h = 0;
        float Evalue_v = 0;
        float Edistvalue_v = 0;
        float e_bl = 0;
        float e_donut = 0;
        float e_h = 0;
        float e_v = 0;
        float o = 0;

        int wsize = windowCPU;
        int msize = matrixSizeCPU;
        int pwidth = peakWidthCPU;
        int buffer_width = HiCCUPS.regionMargin;

        int diagDist = Math.abs(diff);
        int maxIndex = msize - buffer_width;

        wsize = Math.min(wsize, (diagDist - 1) / 2);
        if (wsize <= pwidth) {
            wsize = pwidth + 1;
        }
        wsize = Math.min(wsize, buffer_width);

        // calculate initial bottom left box
        for (int i = buffer_width + 1; i <= buffer_width + wsize; i++) {
            for (int j = buffer_width - wsize; j < buffer_width; j++) {
                if (!Double.isNaN(subMatrix.getEntry(i,j))) {
                    if (i + diff - j < 0) {
                        Evalue_bl += subMatrix.getEntry(i,j);
                        Edistvalue_bl += expected.get(Math.abs(i + diff - j));
                    }
                }
            }
        }
        //Subtract off the middle peak
        for (int i = buffer_width + 1; i <= buffer_width + pwidth; i++) {
            for (int j = buffer_width - pwidth; j < buffer_width; j++) {
                if (!Double.isNaN(subMatrix.getEntry(i,j))) {
                    if (i + diff - j < 0) {
                        Evalue_bl -= subMatrix.getEntry(i,j);
                        Edistvalue_bl -= expected.get(Math.abs(i + diff - j));
                    }
                }
            }
        }

        //fix box dimensions
        while (Evalue_bl < 16) {
            Evalue_bl = 0;
            Edistvalue_bl = 0;
            wsize += 1;
            //dvisor = powf(wsize,2.0) - powf(pwidth,2.0);
            for (int i = buffer_width + 1; i <= buffer_width + wsize; i++) {
                for (int j = buffer_width - wsize; j < buffer_width; j++) {
                    if (!Double.isNaN(subMatrix.getEntry(i,j))) {
                        if (i + diff - j < 0) {
                            Evalue_bl += subMatrix.getEntry(i,j);
                            int distVal = Math.abs(i + diff - j);
                            Edistvalue_bl += expected.get(distVal);
                            if (i >= buffer_width + 1) {
                                if (i < buffer_width + pwidth + 1) {
                                    if (j >= buffer_width - pwidth) {
                                        if (j < buffer_width) {
                                            Evalue_bl -= subMatrix.getEntry(i,j);
                                            Edistvalue_bl -= expected.get(distVal);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (wsize >= buffer_width) {
                break;
            }
            if (2 * wsize >= diagDist) {
                break;
            }
        }

        // calculate donut
        for (int i = buffer_width - wsize; i <= buffer_width + wsize; ++i) {
            for (int j = buffer_width - wsize; j <= buffer_width + wsize; ++j) {
                if (!Double.isNaN(subMatrix.getEntry(i,j))) {
                    if (i + diff - j < 0) {
                        Evalue_donut += subMatrix.getEntry(i,j);
                        Edistvalue_donut += expected.get(Math.abs(i + diff - j));
                    }
                }
            }
        }
        //Subtract off the middle peak
        for (int i = buffer_width - pwidth; i <= buffer_width + pwidth; ++i) {
            for (int j = buffer_width - pwidth; j <= buffer_width + pwidth; ++j) {
                if (!Double.isNaN(subMatrix.getEntry(i,j))) {
                    if (i + diff - j < 0) {
                        Evalue_donut -= subMatrix.getEntry(i,j);
                        Edistvalue_donut -= expected.get(Math.abs(i + diff - j));
                    }
                }
            }
        }
        //Subtract off the cross hairs left side
        for (int i = buffer_width - wsize; i < buffer_width - pwidth; i++) {
            if (!Double.isNaN(subMatrix.getEntry(i,buffer_width))) {
                Evalue_donut -= subMatrix.getEntry(i,buffer_width);
                Edistvalue_donut -= expected.get(Math.abs(i + diff - buffer_width));
            }
            for (int j = -1; j <= 1; j++) {
                Evalue_v += subMatrix.getEntry(i,buffer_width + j);
                Edistvalue_v += expected.get(Math.abs(i + diff - buffer_width - j));
            }
        }
        //Subtract off the cross hairs right side
        for (int i = buffer_width + pwidth + 1; i <= buffer_width + wsize; ++i) {
            if (!Double.isNaN(subMatrix.getEntry(i,buffer_width))) {
                Evalue_donut -= subMatrix.getEntry(i,buffer_width);
                Edistvalue_donut -= expected.get(Math.abs(i + diff - buffer_width));
            }
            for (int j = -1; j <= 1; ++j) {
                Evalue_v += subMatrix.getEntry(i,buffer_width + j);
                Edistvalue_v += expected.get(Math.abs(i + diff - buffer_width - j));
            }
        }
        //Subtract off the cross hairs top side
        for (int j = buffer_width - wsize; j < buffer_width - pwidth; ++j) {
            if (!Double.isNaN(subMatrix.getEntry(buffer_width,j))) {
                Evalue_donut -= subMatrix.getEntry(buffer_width,j);
                Edistvalue_donut -= expected.get(Math.abs(buffer_width + diff - j));
            }
            for (int i = -1; i <= 1; ++i) {
                Evalue_h += subMatrix.getEntry(buffer_width + i,j);
                Edistvalue_h += expected.get(Math.abs(buffer_width + i + diff - j));
            }
        }
        //Subtract off the cross hairs bottom side
        for (int j = buffer_width + pwidth + 1; j <= buffer_width + wsize; ++j) {
            if (!Double.isNaN(subMatrix.getEntry(buffer_width,j))) {
                Evalue_donut -= subMatrix.getEntry(buffer_width,j);
                Edistvalue_donut -= expected.get(Math.abs(buffer_width + diff - j));
            }
            for (int i = -1; i <= 1; ++i) {
                Evalue_h += subMatrix.getEntry(buffer_width + i,j);
                Edistvalue_h += expected.get(Math.abs(buffer_width + i + diff - j));
            }
        }

        e_bl = (float) ((Evalue_bl * expected.get(diagDist)) / Edistvalue_bl) * norm1 * norm2;
        e_donut = (float) ((Evalue_donut * expected.get(diagDist)) / Edistvalue_donut) * norm1 * norm2;
        e_h = (float) ((Evalue_h * expected.get(diagDist)) / Edistvalue_h) * norm1 * norm2;
        e_v = (float) ((Evalue_v * expected.get(diagDist)) / Edistvalue_v) * norm1 * norm2;

        float[] scores = new float[]{e_bl, e_donut, e_h, e_v};
        return scores;
    }

    private float[] calculateBottomLeftExtrusionScore(RealMatrix subMatrix, ListOfDoubleArrays expected, float norm1, float norm2, int diff, ListOfDoubleArrays distScaling, long binX, long binY, int resolution) {
        int wsize = windowCPU;
        int msize = matrixSizeCPU;
        int pwidth = peakWidthCPU;
        int buffer_width = HiCCUPS.regionMargin;

        float leftSum = 0, downSum = 0, centerSum = 0, centerSumDC = 0, leftSumDC = 0, downSumDC = 0;

        int sigma = 2;
        for (int i = -1 * 3 * sigma; i <= 3 * sigma; i++) {
            for (int j = -1 * 3 * sigma; j <= 3 * sigma; j++) {
                centerSum += ((subMatrix.getEntry(buffer_width+i, buffer_width+j)) * gaussFilter(i, j, sigma));
                centerSumDC += ((subMatrix.getEntry(buffer_width+i, buffer_width+j) - computeDistCorrection(Math.abs(i + diff - j), distScaling, binX, binY, resolution)) * gaussFilter(i, j, sigma));
                leftSum += ((subMatrix.getEntry(buffer_width+i, buffer_width-(6*sigma)+j)) * gaussFilter(i,j,sigma));
                leftSumDC += ((subMatrix.getEntry(buffer_width+i, buffer_width-(6*sigma)+j) - computeDistCorrection(Math.abs(i + diff - j - (6*sigma)), distScaling, binX, binY, resolution)) * gaussFilter(i,j,sigma));
                downSum += ((subMatrix.getEntry(buffer_width+(6*sigma)+i, buffer_width+j)) * gaussFilter(i,j,sigma));
                downSumDC += ((subMatrix.getEntry(buffer_width+(6*sigma)+i, buffer_width+j) - computeDistCorrection(Math.abs(i + diff - j - (6*sigma)), distScaling, binX, binY, resolution)) * gaussFilter(i,j,sigma));
            }
        }

        if (centerSumDC < 0) {
            centerSumDC = 0;
        }
        if (leftSumDC < 0) {
            leftSumDC = 0;
        }
        if (downSumDC < 0) {
            downSumDC = 0;
        }

        float[] scores = new float[]{centerSum, leftSum, downSum, centerSumDC, leftSumDC, downSumDC};
        return scores;

    }

    private float computeDistCorrection(int dist, ListOfDoubleArrays distScaling, long binX, long binY, int resolution) {
        float normSum = 0, normAverage = 0;
        int normCounter = 0;
        for (long i = binX; i < binY - (normDist / resolution) + 1; i++) {
            normSum += distScaling.get(i);
            normCounter++;
        }
        normAverage = normSum / normCounter;
        float distCorrection = (float) Math.pow((dist * 1.0 ) / (normDist / resolution), scaling);
        return distCorrection * normAverage;
    }

    private float gaussFilter(int i, int j, int sigma) {
        return (float) (Math.exp(-1 * (Math.pow(i,2)+Math.pow(j,2)) / (2*Math.pow(sigma,2))) / (2 * Math.PI * Math.pow(sigma, 2)));
    }

    private void initializeDistScaling(MatrixZoomData zd, ListOfDoubleArrays distScaling, int dist) {
        try {
            zd.initializeDistScaling(distScaling, dist, norm);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
