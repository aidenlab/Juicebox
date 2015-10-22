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

package juicebox.tools.clt.juicer;

import jargs.gnu.CmdLineParser;
import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.data.Dataset;
import juicebox.data.HiCFileTools;
import juicebox.data.Matrix;
import juicebox.data.MatrixZoomData;
import juicebox.tools.clt.CommandLineParserForJuicer;
import juicebox.tools.clt.JuicerCLT;
import juicebox.tools.utils.juicer.apa.APADataStack;
import juicebox.tools.utils.juicer.apa.APAUtils;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.Feature2DParser;
import juicebox.track.feature.FeatureFilter;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import org.broad.igv.Globals;
import org.broad.igv.feature.Chromosome;

import java.io.IOException;
import java.util.*;

/**
 * Aggregate Peak Analysis developed by mhuntley
 * (AKA PSEA - peak set enrichment analysis)
 *
 * @author mshamim
 */
public class APA extends JuicerCLT {

    public static final int regionWidth = 6; //size of boxes
    private final boolean saveAllData = true;
    private String[] files;

    //defaults
    private double minPeakDist = 30; // distance between two bins, can be changed in opts
    private double maxPeakDist = Double.POSITIVE_INFINITY;
    private int window = 10;
    private Set<String> givenChromosomes = null;
    private int[] resolutions = new int[]{25000, 10000};
    private NormalizationType norm;

    /**
     * Usage for APA
     */
    public APA() {
        super("apa [-n minval] [-x maxval] [-w window]  [-r resolution(s)] [-c chromosomes] <NONE/VC/VC_SQRT/KR> <hic file(s)> <PeaksFile> <SaveFolder> [SavePrefix]");
        HiCGlobals.useCache = false; // TODO fix memory leak of contact records in cache (dataset?)
    }

    @Override
    public void readArguments(String[] args, CmdLineParser parser) {

        CommandLineParserForJuicer juicerParser = (CommandLineParserForJuicer) parser;

        if (!(args.length > 3 && args.length < 6)) {
            printUsage();
        }

        norm = retrieveNormalization(args[1]);

        files = new String[4];
        files[3] = "";
        System.arraycopy(args, 2, files, 0, args.length - 2);

        for (String s : files)
            System.out.println("--- " + s);

        //if (files.length > 4)
        //    restrictionSiteFilename = files[4];
        //[min value, max value, window, resolution]
        minPeakDist = juicerParser.getAPAMinVal();
        if (minPeakDist == 0)
            minPeakDist = 30;

        maxPeakDist = juicerParser.getAPAMaxVal();
        if (maxPeakDist == 0)
            maxPeakDist = Double.POSITIVE_INFINITY;

        window = juicerParser.getAPAWindowSizeOption();
        if (window == 0)
            window = 10;

        List<String> possibleResolutions = juicerParser.getMultipleResolutionOptions();
        if (possibleResolutions != null) {
            resolutions = new int[possibleResolutions.size()];
            int i = 0;
            for (String res : possibleResolutions) {
                resolutions[i] = Integer.parseInt(res);
                i++;
            }
        }

        List<String> possibleChromosomes = juicerParser.getChromosomeOption();
        if (possibleResolutions != null && possibleChromosomes.size() > 0) {
            givenChromosomes = new HashSet<String>(possibleChromosomes);
        }
    }

    @Override
    public void run() {

        //Calculate parameters that will need later
        int L = 2 * window + 1;
        List<String> summedHiCFiles = Arrays.asList(files[0].split("\\+"));

        Integer[] gwPeakNumbers = new Integer[3];
        for (int i = 0; i < gwPeakNumbers.length; i++)
            gwPeakNumbers[i] = 0;

        Dataset ds = HiCFileTools.extractDatasetForCLT(summedHiCFiles, true);
        for (final int resolution : HiCFileTools.filterResolutions(ds, resolutions)) {

            System.out.println("res " + resolution);
            HiCZoom zoom = new HiCZoom(HiC.Unit.BP, resolution);

            List<Chromosome> chromosomes = ds.getChromosomes();
            if (givenChromosomes != null)
                chromosomes = new ArrayList<Chromosome>(HiCFileTools.stringToChromosomes(givenChromosomes,
                        chromosomes));

            // Metrics resulting from apa filtering
            final Map<String, Integer[]> filterMetrics = new HashMap<String, Integer[]>();

            Feature2DList loopList = Feature2DParser.loadFeatures(files[1], chromosomes, false,
                    new FeatureFilter() {
                        // Remove duplicates and filters by size
                        // also save internal metrics for these measures
                        @Override
                        public List<Feature2D> filter(String chr, List<Feature2D> features) {

                            List<Feature2D> uniqueFeatures = new ArrayList<Feature2D>(new HashSet<Feature2D>(features));
                            List<Feature2D> filteredUniqueFeatures = APAUtils.filterFeaturesBySize(uniqueFeatures,
                                    minPeakDist, maxPeakDist, resolution);

                            filterMetrics.put(chr,
                                    new Integer[]{filteredUniqueFeatures.size(), uniqueFeatures.size(), features.size()});

                            return filteredUniqueFeatures;
                        }
                    });
            if (loopList.getNumTotalFeatures() > 0) {

                for (Chromosome chr : chromosomes) {
                    APADataStack apaDataStack = new APADataStack(L, files[2], (files[0] + "_" + resolution).replace("/", "_"));

                    if (chr.getName().equals(Globals.CHR_ALL)) continue;

                    Matrix matrix = ds.getMatrix(chr, chr);
                    if (matrix == null) continue;

                    MatrixZoomData zd = matrix.getZoomData(zoom);

                    System.out.println("CHR " + chr.getName() + " " + chr.getIndex());
                    List<Feature2D> loops = loopList.get(chr.getIndex(), chr.getIndex());
                    if (loops == null || loops.size() == 0) {
                        System.out.println("CHR " + chr.getName() + " - no loops, check loop filtering constraints");
                        continue;
                    }

                    Integer[] peakNumbers = filterMetrics.get(Feature2DList.getKey(chr, chr));


                    if (loops.size() != peakNumbers[0])
                        System.out.println("Error reading stat numbers fro " + chr);

                    for (int i = 0; i < peakNumbers.length; i++) {
                        //System.out.println(chr + " " + i + " " + peakNumbers[i] + " " + gwPeakNumbers[i]);
                        gwPeakNumbers[i] += peakNumbers[i];
                    }

                    //System.out.println("Loop");
                    for (Feature2D loop : loops) {
                        //System.out.println(loop.getMidPt1()/resolution +"\t"+loop.getMidPt2()/resolution);
                        try {
                            apaDataStack.addData(APAUtils.extractLocalizedData(zd, loop, L, resolution, window,
                                    norm));
                        } catch (IOException e) {
                            System.err.println("Unable to find data for loop: " + loop);
                        }
                    }

                    apaDataStack.updateGenomeWideData();
                    if (saveAllData)
                        apaDataStack.exportDataSet(chr.getName(), peakNumbers);

                }
                APADataStack.exportGenomeWideData(gwPeakNumbers);
                APADataStack.clearAllData();
            } else {
                System.err.println("Looplist is empty or incorrect path provided.");
                System.exit(-8);
            }
        }
        System.exit(0);
    }
}