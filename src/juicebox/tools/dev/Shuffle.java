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
import juicebox.tools.utils.dev.LocalGenomeRegion;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationHandler;
import juicebox.windowui.NormalizationType;
import org.broad.igv.feature.Chromosome;

import java.io.File;
import java.io.FileWriter;
import java.util.*;

public class Shuffle extends JuicerCLT {

    private Dataset ds;
    private File outputDirectory;
    private int resolution = 100000;
    public static int expectedCliqueSize = 5;

    public Shuffle() {
        super("shuffle  <hicFile> <assembly_file> <output_directory>");
        HiCGlobals.useCache = false;
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
        }

        int specifiedCliqueSize = juicerParser.getMatrixSizeOption();
        if (specifiedCliqueSize > 1) {
            expectedCliqueSize = specifiedCliqueSize;
        }

        updateNumberOfCPUThreads(juicerParser);
    }

    @Override
    public void run() {

        Map<Integer, LocalGenomeRegion> indexToRegion = new HashMap<>();


        HiCZoom zoom = ds.getZoomForBPResolution(resolution);
        for (Chromosome chr : ds.getChromosomeHandler().getChromosomeArrayWithoutAllByAll()) {
            Matrix matrix = ds.getMatrix(chr, chr);
            if (matrix == null) continue;
            MatrixZoomData zd = matrix.getZoomData(zoom);


            int maxIndex = chr.getLength() / resolution + 1;

            List<Block> blocks = zd.getNormalizedBlocksOverlapping(0, 0, maxIndex, maxIndex, NormalizationHandler.KR, false);
            for (Block b : blocks) {
                for (ContactRecord cr : b.getContactRecords()) {
                    final int x = cr.getBinX();
                    final int y = cr.getBinY();
                    final float counts = cr.getCounts();

                    if (!indexToRegion.containsKey(x)) {
                        indexToRegion.put(x, new LocalGenomeRegion(x));
                    }

                    if (!indexToRegion.containsKey(y)) {
                        indexToRegion.put(y, new LocalGenomeRegion(y));
                    }

                    if (x != y) {
                        indexToRegion.get(x).addNeighbor(y, counts);
                        indexToRegion.get(y).addNeighbor(x, counts);
                    }
                }
            }

            /*
            Iterator<ContactRecord> iter = zd.getNewContactRecordIterator();
            while (iter.hasNext()) {
                ContactRecord cr = iter.next();
                final int x = cr.getBinX();
                final int y = cr.getBinY();
                final float counts = cr.getCounts();

                if(!indexToRegion.containsKey(x)){
                    indexToRegion.put(x, new LocalGenomeRegion(x));
                }

                if(!indexToRegion.containsKey(y)){
                    indexToRegion.put(y, new LocalGenomeRegion(y));
                }

                if(x != y){
                    indexToRegion.get(x).addNeighbor(y, counts);
                    indexToRegion.get(y).addNeighbor(x, counts);
                }
            }
            */
        }

        int actualCliqueSize = 3;
        for (LocalGenomeRegion region : indexToRegion.values()) {
            region.filterDownValues(4 * expectedCliqueSize);
        }

        File outputFile = new File(outputDirectory, "breakpoints.bed");
        File outputBEDPEFile = new File(outputDirectory, "breakpoints.txt");
        try {
            final FileWriter positionsBED = new FileWriter(outputFile);
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
                int upmostIndex = Math.max(0, index - actualCliqueSize);
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
                int downmostIndex = Math.min(maxIndex, index + actualCliqueSize);
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
                } else if (thisRegionisNotFineUpstream) {
                    int y1 = selectedRegion.getOutlierContacts(true, 4 * actualCliqueSize);
                    if (y1 > 0) {
                        int y2 = (y1 + 1) * resolution;
                        y1 = y1 * resolution;

                        positionsBEDPE.write("assembly\t" + currX1 + "\t" + currX2 +
                                "\tassembly\t" + y1 + "\t" + y2 + "\t50,255,50\n");

                    }
                } else if (thisRegionisNotFineDownstream) {
                    int y1 = selectedRegion.getOutlierContacts(false, 4 * actualCliqueSize);
                    if (y1 > 0) {
                        int y2 = (y1 + 1) * resolution;
                        y1 = y1 * resolution;
                        positionsBEDPE.write("assembly\t" + currX1 + "\t" + currX2 +
                                "\tassembly\t" + y1 + "\t" + y2 + "\t0,0,255\n");
                    }
                }
            }

            positionsBED.close();
            positionsBEDPE.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
