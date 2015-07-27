/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2015 Broad Institute, Aiden Lab
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

package juicebox.tools.clt;

import juicebox.HiCGlobals;
import juicebox.data.Dataset;
import juicebox.data.DatasetReaderV2;
import juicebox.data.Matrix;
import juicebox.data.MatrixZoomData;
import juicebox.tools.HiCTools;
import juicebox.tools.utils.common.HiCFileTools;
import juicebox.tools.utils.juicer.apa.APADataStack;
import juicebox.tools.utils.juicer.apa.APAUtils;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.Feature2DParser;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import org.broad.igv.Globals;
import org.broad.igv.feature.Chromosome;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * TODO - once fully debugged, change notation convention from underscore to camelcase (match the rest of the files)
 */
public class APA extends JuiceboxCLT {

    public static final int regionWidth = 6; //size of boxes
    private final boolean saveAllData = true;
    private String[] files;
    //defaults
    private double minPeakDist = 30; // distance between two bins, can be changed in opts
    private double maxPeakDist = Double.POSITIVE_INFINITY;
    private int window = 10;
    private Set<String> givenChromosomes = null;
    //int peakwidth = 2; //for enrichment calculation of crosshair norm
    private int[] resolutions = new int[]{25000, 10000};

    public APA() {
        super("apa [-n minval] [-x maxval] [-w window]  [-r resolution(s)] [-c chromosomes] [-x do not cache memory blocks] <hic file(s)> <PeaksFile> <SaveFolder> [SavePrefix]");
    }

    @Override
    public void readArguments(String[] args, HiCTools.CommandLineParser parser) throws IOException {

        if (!(args.length > 3 && args.length < 6)) {
            throw new IOException("1");
        }

        files = new String[4];
        files[3] = "";
        System.arraycopy(args, 1, files, 0, args.length - 1);


        for (String s : files)
            System.out.println("--- " + s);

        //if (files.length > 4)
        //    restrictionSiteFilename = files[4];
        //[min value, max value, window, resolution]


        minPeakDist = parser.getAPAMinVal();
        if (minPeakDist == 0)
            minPeakDist = 30;

        maxPeakDist = parser.getAPAMaxVal();
        if (maxPeakDist == 0)
            maxPeakDist = Double.POSITIVE_INFINITY;

        window = parser.getAPAWindowSizeOption();
        if (window == 0)
            window = 10;

        Set<String> possibleResolutions = parser.getMultipleResolutionOptions();
        if (possibleResolutions != null) {
            resolutions = new int[possibleResolutions.size()];
            int i = 0;
            for (String res : possibleResolutions) {
                resolutions[i] = Integer.parseInt(res);
                i++;
            }
        }
        givenChromosomes = parser.getChromosomeOption();
    }

    @Override
    public void run() {

        //Calculate parameters that will need later
        int L = 2 * window + 1;

        try {
            String[] hicFiles = files[0].split(",");
            for (String hicFile : hicFiles) {
                for (int resolution : resolutions) {

                    Integer[] gwPeakNumbers = new Integer[3];
                    for (int i = 0; i < gwPeakNumbers.length; i++)
                        gwPeakNumbers[i] = 0;

                    System.out.println("Accessing " + hicFile);
                    DatasetReaderV2 reader = new DatasetReaderV2(hicFile);
                    Dataset ds = reader.read();
                    HiCGlobals.verifySupportedHiCFileVersion(reader.getVersion());

                    // select zoom level closest to the requested one
                    HiCZoom zoom = HiCFileTools.getZoomLevel(ds, resolution);
                    resolution = zoom.getBinSize();

                    List<Chromosome> chromosomes = ds.getChromosomes();
                    if (givenChromosomes != null)
                        chromosomes = new ArrayList<Chromosome>(HiCFileTools.stringToChromosomes(givenChromosomes,
                                chromosomes));

                    Feature2DList loopList = Feature2DParser.parseLoopFile(files[1], chromosomes,
                            true, minPeakDist, maxPeakDist, resolution, false);

                    for (Chromosome chr : chromosomes) {
                        APADataStack apaDataStack = new APADataStack(L, files[2] , (hicFile+"_"+resolution).replace("/","_"));

                        if (chr.getName().equals(Globals.CHR_ALL)) continue;

                        Matrix matrix = ds.getMatrix(chr, chr);
                        if (matrix == null) continue;

                        MatrixZoomData zd = matrix.getZoomData(zoom);

                        System.out.println("CHR "+chr.getName() +" "+chr.getIndex());
                        List<Feature2D> loops = loopList.get(chr.getIndex(), chr.getIndex());
                        if(loops == null || loops.size() == 0) {
                            System.out.println("CHR " + chr.getName() + " " + chr.getIndex() + " - no loops found or other error");
                            continue;
                        }

                        Integer[] peakNumbers = loopList.getFilterMetrics(chr);

                        if (loops.size() != peakNumbers[0])
                            System.out.println("Error reading stat numbers fro " + chr);

                        for (int i = 0; i < peakNumbers.length; i++) {
                            //System.out.println(chr + " " + i + " " + peakNumbers[i] + " " + gwPeakNumbers[i]);
                            gwPeakNumbers[i] += peakNumbers[i];
                        }

                        //System.out.println("Loop");
                        for (Feature2D loop : loops) {
                            //System.out.println(loop.getMidPt1()/resolution +"\t"+loop.getMidPt2()/resolution);
                            apaDataStack.addData(APAUtils.extractLocalizedData(zd, loop, L, resolution, window,
                                    NormalizationType.NONE));
                        }

                        apaDataStack.updateGenomeWideData();
                        if (saveAllData)
                            apaDataStack.exportDataSet(chr.getName(), peakNumbers);

                    }
                    APADataStack.exportGenomeWideData(gwPeakNumbers);
                    APADataStack.clearAllData();
                }
            }
        } catch (IOException e) {
            System.out.println("Unable to extract apa data");
            e.printStackTrace();
            System.exit(-3);
        }


        System.exit(7);
    }
}