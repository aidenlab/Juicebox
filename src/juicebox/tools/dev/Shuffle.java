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

import juicebox.HiCGlobals;
import juicebox.data.*;
import juicebox.tools.clt.CommandLineParserForJuicer;
import juicebox.tools.clt.JuicerCLT;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.tools.utils.common.UNIXTools;
import juicebox.tools.utils.dev.LocalGenomeRegion;
import juicebox.tools.utils.juicer.grind.DistortionFinder;
import juicebox.tools.utils.juicer.grind.GrindUtils;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.Feature2DParser;
import juicebox.track.feature.FeatureFunction;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.RealMatrix;
import org.broad.igv.feature.Chromosome;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class Shuffle extends JuicerCLT {

    private Dataset ds;
    private File outputDirectory;
    private static int expectedCliqueSize = 3, expectedCliqueSizeWithBuffer = 12;
    private int resolution = 25000;
    private int neuralNetSize = 500;
    private int highResolution = 1000;
    private int numPixelOverlapWhileSliding = 5;

    public Shuffle() {
        super("shuffle [-r res1,res2] [-k normalization] [-w clique size] [-m matrixSizesForNeuralNet]" +
                "  <hicFile> <assembly_file> <output_directory>");
        HiCGlobals.useCache = false;
    }

    public static void writeStrictIntsToFile(File path, List<Integer> positions) {
        try (ObjectOutputStream write = new ObjectOutputStream(new FileOutputStream(path))) {
            for (Integer pos : positions) {
                write.writeObject(pos);
            }
        } catch (Exception eo) {
            eo.printStackTrace();
        }
    }

    public static void writeStrictMapToFile(File path, Map<Integer, LocalGenomeRegion> indexToRegion) {
        List<Integer> keys = new ArrayList<>(indexToRegion.keySet());
        Collections.sort(keys);

        try (ObjectOutputStream write = new ObjectOutputStream(new FileOutputStream(path))) {
            for (Integer key : keys) {
                LocalGenomeRegion region = indexToRegion.get(key);

                write.writeObject(region.toString());
            }
        } catch (Exception eo) {
            eo.printStackTrace();
        }
    }

    @Override
    protected void readJuicerArguments(String[] args, CommandLineParserForJuicer juicerParser) {
        if (args.length != 4) {
            // 3 - standard, 5 - when list/control provided
            printUsageAndExit();  // this will exit
        }

        ds = HiCFileTools.extractDatasetForCLT(Arrays.asList(args[1].split("\\+")), true);
        outputDirectory = HiCFileTools.createValidDirectory(args[3]);


        NormalizationType preferredNorm = juicerParser.getNormalizationTypeOption(ds.getNormalizationHandler());
        if (preferredNorm != null)
            norm = preferredNorm;

        List<String> potentialResolution = juicerParser.getMultipleResolutionOptions();
        if (potentialResolution != null) {
            resolution = Integer.parseInt(potentialResolution.get(0));
            highResolution = Integer.parseInt(potentialResolution.get(1));
        }

        int specifiedCliqueSize = juicerParser.getAPAWindowSizeOption();
        if (specifiedCliqueSize > 1) {
            expectedCliqueSize = specifiedCliqueSize;
            expectedCliqueSizeWithBuffer = specifiedCliqueSize * 4;
        }

        updateNumberOfCPUThreads(juicerParser);

        int specifiedMatrixSize = juicerParser.getMatrixSizeOption();
        if (specifiedMatrixSize > 10) {
            neuralNetSize = specifiedMatrixSize;
        }
    }

    public static void getContiguousDataAndSaveToFile(MatrixZoomData zd, int box1XIndex, String chrom1Name,
                                                      int imgHalfSliceWidth, NormalizationType norm,
                                                      String negPath, Writer negDataWriter) {

        int fullWidth = 2 * imgHalfSliceWidth;
        int box1RectUL = box1XIndex;
        int box1Mid = box1XIndex + imgHalfSliceWidth;
        int box1RectLR = box1XIndex + fullWidth;

        try {
            RealMatrix localizedRegionDataBox = HiCFileTools.extractLocalBoundedRegion(zd,
                    box1RectUL, box1RectLR, box1RectUL, box1RectLR, fullWidth, fullWidth, norm, true);
            float[][] compositeMatrix = MatrixTools.convertToFloatMatrix(localizedRegionDataBox.getData());
            MatrixTools.cleanUpNaNs(compositeMatrix);

            String filePrefix = "origDiag_" + chrom1Name + "_" + box1XIndex + "_" + chrom1Name + "_" + box1Mid + "_matrix";
            GrindUtils.saveGrindMatrixDataToFile(filePrefix, negPath, compositeMatrix, negDataWriter, false);

        } catch (Exception e) {
        }
    }

    private void assessOutlierContactAndPrintToFile(boolean isBadUpstream, int cliqueSize, FileWriter positionsBEDPE,
                                                    LocalGenomeRegion selectedRegion, int currX1, int currX2) throws IOException {
        int y1 = selectedRegion.getOutlierContacts(isBadUpstream, cliqueSize);
        if (y1 > 0) {
            int y2 = (y1 + 1) * resolution;
            y1 = y1 * resolution;
            if (y1 > currX1) {
                positionsBEDPE.write("assembly\t" + currX1 + "\t" + currX2 +
                        "\tassembly\t" + y1 + "\t" + y2 + "\t50,255,50\n");
            } else {
                positionsBEDPE.write("assembly\t" + y1 + "\t" + y2 +
                        "\tassembly\t" + currX1 + "\t" + currX2 + "\t50,255,50\n");
            }
        }
    }

    @Override
    public void run() {

        Map<Integer, LocalGenomeRegion> indexToRegion = getMapOfIndicesToRegion();

        for (LocalGenomeRegion region : indexToRegion.values()) {
            region.filterDownValues(expectedCliqueSizeWithBuffer);
        }


        File outputBEDPEFile = new File(outputDirectory, "breakpoints.txt");
        try {
            final FileWriter positionsBED = new FileWriter(new File(outputDirectory, "breakpoints.bed"));
            final FileWriter positionsBEDPE = new FileWriter(outputBEDPEFile);
            positionsBEDPE.write("#chr1\tx1\tx2\tch2\ty1\ty2\tcolor\n");

            ArrayList<Integer> positions = new ArrayList<>(indexToRegion.keySet());
            Collections.sort(positions);
            int maxIndex = positions.get(positions.size() - 1);

            // iterate for every region
            for (int index : positions) {
                // selected region
                LocalGenomeRegion selectedRegion = indexToRegion.get(index);
                boolean thisRegionisNotFineUpstream = false;
                boolean thisRegionisNotFineDownstream = false;

                // expected closest upstream neighbors
                int upmostIndex = Math.max(0, index - expectedCliqueSize);
                for (int k = upmostIndex; k < index; k++) {
                    if (indexToRegion.containsKey(k)) {
                        LocalGenomeRegion upstreamRegion = indexToRegion.get(k);
                        if (selectedRegion.notConnectedWith(k) && upstreamRegion.notConnectedWith(index)) {
                            thisRegionisNotFineUpstream = true;
                        }
                    } else {
                        System.err.println(k + " missing");
                    }
                }

                // expected closest downstream neighbors
                int downmostIndex = Math.min(maxIndex, index + expectedCliqueSize);
                for (int k = index + 1; k < downmostIndex; k++) {
                    if (indexToRegion.containsKey(k)) {
                        LocalGenomeRegion downstreamRegion = indexToRegion.get(k);
                        if (selectedRegion.notConnectedWith(k) && downstreamRegion.notConnectedWith(index)) {
                            thisRegionisNotFineDownstream = true;
                        }
                    } else {
                        System.err.println(k + " missing");
                    }
                }

                int currX1 = index * resolution;
                int currX2 = (index + 1) * resolution;

                if (thisRegionisNotFineDownstream || thisRegionisNotFineUpstream) {
                    positionsBED.write("assembly\t" + currX1 + "\t" + currX2 + "\n");
                }

                if (thisRegionisNotFineUpstream && thisRegionisNotFineDownstream) {
                    positionsBEDPE.write("assembly\t" + currX1 + "\t" + currX2 +
                            "\tassembly\t" + currX1 + "\t" + currX2 + "\t0,0,0\n");
                }

                if (thisRegionisNotFineUpstream) {
                    assessOutlierContactAndPrintToFile(true, expectedCliqueSizeWithBuffer,
                            positionsBEDPE, selectedRegion, currX1, currX2);
                }

                if (thisRegionisNotFineDownstream) {
                    assessOutlierContactAndPrintToFile(false, expectedCliqueSizeWithBuffer,
                            positionsBEDPE, selectedRegion, currX1, currX2);
                }
            }

            positionsBED.close();
            positionsBEDPE.close();

            createInputForNeuralNet(ds, outputBEDPEFile, outputDirectory);
            createDiagonalInputForNeuralNet(ds, outputDirectory);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void createInputForNeuralNet(Dataset ds, File outputBEDPEFile, File outputDirectory) {

        Feature2DList features = Feature2DParser.loadFeatures(outputBEDPEFile.getAbsolutePath(), ds.getChromosomeHandler(), false, null, true);
        System.out.println("Neural Net Size " + neuralNetSize);

        features.processLists(new FeatureFunction() {
            @Override
            public void process(String chr, List<Feature2D> feature2DList) {
                Chromosome chrom = ds.getChromosomeHandler().getChromosomeFromName(feature2DList.get(0).getChr1());

                final MatrixZoomData zd = HiCFileTools.getMatrixZoomData(ds, chrom, chrom, highResolution);
                if (zd == null) return;

                System.out.println("Currently on: " + chr);

                File outputFolderForRes = new File(outputDirectory, "neural_net_run_" + highResolution);
                UNIXTools.makeDir(outputFolderForRes);

                try {
                    String fileNameForWriter = outputDirectory.getAbsolutePath() + "/chr_" + chrom.getName() + "_" + highResolution + ".txt";
                    Writer fileNameWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(fileNameForWriter), StandardCharsets.UTF_8));

                    int[] breakPointsForBoundaries = getBreakPointsBasedOnCPUThreads(feature2DList.size());

                    ExecutorService executor = Executors.newFixedThreadPool(numCPUThreads);

                    for (int k = 0; k < numCPUThreads; k++) {
                        int indxStart = breakPointsForBoundaries[k];
                        int indxEnd = breakPointsForBoundaries[k + 1];

                        Runnable worker = new Runnable() {
                            @Override
                            public void run() {
                                for (int i = indxStart; i < indxEnd; i++) {
                                    Feature2D feature2D = feature2DList.get(i);
                                    processFeature2DAndGetRegionsOfInterestFromIt(feature2D, chrom, zd, fileNameWriter, outputFolderForRes.getAbsolutePath());
                                }
                            }
                        };
                        executor.execute(worker);
                    }

                    executor.shutdown();

                    // Wait until all threads finish
                    while (!executor.isTerminated()) {
                    }

                    fileNameWriter.close();

                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });

    }

    public void processFeature2DAndGetRegionsOfInterestFromIt(Feature2D feature2D, Chromosome chrom,
                                                              MatrixZoomData zd, Writer fileNameWriter, String outputFolderForRedPath) {
        int i0 = Math.max(0, (feature2D.getStart1() - resolution) / highResolution);
        int j0 = Math.max(0, (feature2D.getStart2() - resolution) / highResolution);
        int iMax = Math.min((feature2D.getEnd1() + resolution) / highResolution, chrom.getLength() / highResolution);
        int jMax = Math.min((feature2D.getEnd2() + resolution) / highResolution, chrom.getLength() / highResolution);
        int neuralNetworkSlidingIncrement = neuralNetSize / 2 - numPixelOverlapWhileSliding;

        if (feature2D.getMidPt2() < feature2D.getMidPt1()) {
            //System.err.println("jM < iM; skipping - iM " + feature2D.getMidPt1() + " jM " + feature2D.getMidPt2() + " i0 " + i0 + " j0 " + j0);
            return;
        }
        for (int i = i0; i < iMax; i += neuralNetworkSlidingIncrement) {
            for (int j = j0; j < jMax; j += neuralNetworkSlidingIncrement) {

                if (j < i) {
                    //System.err.println("j < i; skipping - i " + i + " j " + j + " i0 " + i0 + " j0 " + j0);
                    continue;
                }

                int newJ = j;
                boolean isContinuousRegion = false;
                if (j <= i + neuralNetSize / 2) {
                    newJ = i + neuralNetSize / 2;
                    isContinuousRegion = true;
                }

                if (isContinuousRegion) {
                    getContiguousDataAndSaveToFile(zd, i, "assembly",
                            neuralNetSize / 2, norm, outputFolderForRedPath, fileNameWriter);
                } else {
                    DistortionFinder.getTrainingDataAndSaveToFile(zd, zd, zd, i, newJ,
                            "assembly", "assembly", isContinuousRegion, neuralNetSize / 2, norm,
                            false, 0, "", null, outputFolderForRedPath,
                            null, null, null, fileNameWriter, null, null,
                            null, null, null, null, false);
                }
            }
        }
    }

    private void createDiagonalInputForNeuralNet(Dataset ds, File outputDirectory) {

        System.out.println("Neural Net Size " + neuralNetSize);

        Chromosome chrom = ds.getChromosomeHandler().getChromosomeFromName("assembly");

        final MatrixZoomData zd = HiCFileTools.getMatrixZoomData(ds, chrom, chrom, highResolution);

        if (zd == null) return;

        System.out.println("Currently on: " + chrom);

        File outputFolderForRes = new File(outputDirectory, "neural_net_diag_run_" + highResolution);
        UNIXTools.makeDir(outputFolderForRes);

        try {
            String diagNameWriterFile = outputDirectory.getAbsolutePath() + "/diag_chr_" + chrom.getName() + "_" + highResolution + ".txt";
            Writer fileNameWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(diagNameWriterFile), StandardCharsets.UTF_8));

            ExecutorService executor = Executors.newFixedThreadPool(numCPUThreads);

            int iMax = chrom.getLength() / highResolution;

            int[] breakPointsForBoundaries = getBreakPointsBasedOnCPUThreads(iMax);

            for (int k = 0; k < numCPUThreads; k++) {
                int iStart = breakPointsForBoundaries[k];
                int iEnd = breakPointsForBoundaries[k + 1];

                Runnable worker = new Runnable() {
                    @Override
                    public void run() {
                        for (int i = iStart; i < iEnd; i += neuralNetSize / 2) {
                            getContiguousDataAndSaveToFile(zd, i, "assembly",
                                    neuralNetSize / 2, norm, outputFolderForRes.getAbsolutePath(), fileNameWriter);
                        }
                    }
                };
                executor.execute(worker);
            }

            executor.shutdown();

            // Wait until all threads finish
            while (!executor.isTerminated()) {
            }

            fileNameWriter.close();

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public int[] getBreakPointsBasedOnCPUThreads(int maxNumIndices) {
        int[] breakPoints = new int[numCPUThreads + 1];
        for (int k = 1; k < numCPUThreads; k++) {
            breakPoints[k] = (k * maxNumIndices / numCPUThreads) + 1;
        }
        breakPoints[numCPUThreads] = maxNumIndices;
        return breakPoints;
    }

    private Map<Integer, LocalGenomeRegion> getMapOfIndicesToRegion() {
        Map<Integer, LocalGenomeRegion> indexToRegion = new HashMap<>();
        System.out.println("Num of CPU threads: " + numCPUThreads);
        ExecutorService executor = Executors.newFixedThreadPool(numCPUThreads);

        for (Chromosome chr : ds.getChromosomeHandler().getChromosomeArrayWithoutAllByAll()) {
            MatrixZoomData zd = HiCFileTools.getMatrixZoomData(ds, chr, chr, resolution);
            if (zd == null) continue;

            int maxIndex = chr.getLength() / resolution + 1;
            for (int k = 0; k < maxIndex; k++) {
                indexToRegion.put(k, new LocalGenomeRegion(k, 2 * expectedCliqueSizeWithBuffer));
            }

            List<Block> blocks = zd.getNormalizedBlocksOverlapping(0, 0, maxIndex, maxIndex,
                    norm, false, false);
            for (Block b : blocks) {
                if (b != null) {
                    Runnable worker = new Runnable() {
                        @Override
                        public void run() {
                            for (ContactRecord cr : b.getContactRecords()) {
                                final int x = cr.getBinX();
                                final int y = cr.getBinY();
                                final float counts = cr.getCounts();

                                synchronized (indexToRegion) {
                                    if (x != y) {
                                        indexToRegion.get(x).addNeighbor(y, counts);
                                        indexToRegion.get(y).addNeighbor(x, counts);
                                    }
                                }
                            }
                        }
                    };
                    executor.execute(worker);
                } else {
                    System.err.println("Block is null?");
                }
            }
        }

        executor.shutdown();

        // Wait until all threads finish
        while (!executor.isTerminated()) {
        }
        return indexToRegion;
    }
}
