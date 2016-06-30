/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2016 Broad Institute, Aiden Lab
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

import java.io.File;
import java.io.IOException;
import java.util.*;

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

    private final boolean saveAllData = true;
    private String hicFilePaths, loopListPath;
    private File outputDirectory;

    //defaults
    // TODO right now these units are based on n*res/sqrt(2)
    // TODO the sqrt(2) scaling should be removed (i.e. handle scaling internally)
    private double minPeakDist = 30; // distance between two bins, can be changed in opts
    private double maxPeakDist = Double.POSITIVE_INFINITY;
    private int window = 10;
    private int[] resolutions = new int[]{10000, 5000};
    private int[] regionWidths = new int[]{6, 3};

    /**
     * Usage for APA
     */
    public APA() {
        super("apa [-n minval] [-x maxval] [-w window] [-r resolution(s)] [-c chromosomes]" +
                " [-k NONE/VC/VC_SQRT/KR] [-q corner_width] <hicFile(s)> <PeaksFile> <SaveFolder>");
        HiCGlobals.useCache = false;
    }

    public static String getBasicUsage() {
        return "apa <hicFile(s)> <PeaksFile> <SaveFolder>";
    }

    @Override
    protected void readJuicerArguments(String[] args, CommandLineParserForJuicer juicerParser) {
        if (args.length != 4) {
            printUsageAndExit();
        }

        hicFilePaths = args[1];
        loopListPath = args[2];
        outputDirectory = HiCFileTools.createValidDirectory(args[3]);

        NormalizationType preferredNorm = juicerParser.getNormalizationTypeOption();
        if (preferredNorm != null)
            norm = preferredNorm;

        double potentialMinPeakDist = juicerParser.getAPAMinVal();
        if (potentialMinPeakDist >= 0)
            minPeakDist = potentialMinPeakDist;

        double potentialMaxPeakDist = juicerParser.getAPAMaxVal();
        if (potentialMaxPeakDist > 0)
            maxPeakDist = potentialMaxPeakDist;

        int potentialWindow = juicerParser.getAPAWindowSizeOption();
        if (potentialWindow > 0)
            window = potentialWindow;

        List<String> possibleRegionWidths = juicerParser.getAPACornerRegionDimensionOptions();
        if (possibleRegionWidths != null) {
            List<Integer> widths = new ArrayList<Integer>();
            for (String res : possibleRegionWidths) {
                widths.add(Integer.parseInt(res));
            }
            regionWidths = Ints.toArray(widths);
        }

        List<String> possibleResolutions = juicerParser.getMultipleResolutionOptions();
        if (possibleResolutions != null) {
            List<Integer> intResolutions = new ArrayList<Integer>();
            for (String res : possibleResolutions) {
                intResolutions.add(Integer.parseInt(res));
            }
            resolutions = Ints.toArray(intResolutions);
        }
    }

    @Override
    public void run() {

        //Calculate parameters that will need later
        int L = 2 * window + 1;
        List<String> summedHiCFiles = Arrays.asList(hicFilePaths.split("\\+"));

        Integer[] gwPeakNumbers = new Integer[3];
        for (int i = 0; i < gwPeakNumbers.length; i++)
            gwPeakNumbers[i] = 0;

        Dataset ds = HiCFileTools.extractDatasetForCLT(summedHiCFiles, true);
        for (final int resolution : HiCFileTools.filterResolutions(ds.getBpZooms(), resolutions)) {

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

            System.out.println("Processing APA for resolution " + resolution);
            HiCZoom zoom = new HiCZoom(HiC.Unit.BP, resolution);

            List<Chromosome> chromosomes = ds.getChromosomes();
            if (givenChromosomes != null)
                chromosomes = new ArrayList<Chromosome>(HiCFileTools.stringToChromosomes(givenChromosomes,
                        chromosomes));

            // Metrics resulting from apa filtering
            final Map<String, Integer[]> filterMetrics = new HashMap<String, Integer[]>();

            Feature2DList loopList = Feature2DParser.loadFeatures(loopListPath, chromosomes, false,
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
                    }, false);

            if (loopList.getNumTotalFeatures() > 0) {

                double maxProgressStatus = chromosomes.size();
                int currentProgressStatus = 0;

                for (Chromosome chr : chromosomes) {
                    APADataStack apaDataStack = new APADataStack(L, outputDirectory, "" + resolution);

                    if (chr.getName().equals(Globals.CHR_ALL)) continue;

                    Matrix matrix = ds.getMatrix(chr, chr);
                    if (matrix == null) continue;

                    MatrixZoomData zd = matrix.getZoomData(zoom);

                    if (HiCGlobals.printVerboseComments) {
                        System.out.println("CHR " + chr.getName() + " " + chr.getIndex());
                    }

                    List<Feature2D> loops = loopList.get(chr.getIndex(), chr.getIndex());
                    if (loops == null || loops.size() == 0) {
                        if (HiCGlobals.printVerboseComments) {
                            System.out.println("CHR " + chr.getName() + " - no loops, check loop filtering constraints");
                        }
                        continue;
                    }

                    Integer[] peakNumbers = filterMetrics.get(Feature2DList.getKey(chr, chr));


                    if (loops.size() != peakNumbers[0])
                        System.err.println("Error reading statistics from " + chr);

                    for (int i = 0; i < peakNumbers.length; i++) {
                        gwPeakNumbers[i] += peakNumbers[i];
                    }

                    for (Feature2D loop : loops) {
                        try {
                            apaDataStack.addData(APAUtils.extractLocalizedData(zd, loop, L, resolution, window, norm));
                        } catch (IOException e) {
                            System.err.println("Unable to find data for loop: " + loop);
                        }
                    }

                    apaDataStack.updateGenomeWideData();
                    if (saveAllData) {
                        apaDataStack.exportDataSet(chr.getName(), peakNumbers, currentRegionWidth);
                    }

                    System.out.print(((int) Math.floor((100.0 * ++currentProgressStatus) / maxProgressStatus)) + "% ");
                }
                System.out.println("Exporting APA results...");
                APADataStack.exportGenomeWideData(gwPeakNumbers, currentRegionWidth);
                APADataStack.clearAllData();
            } else {
                System.err.println("Loop list is empty or incorrect path provided.");
                System.exit(3);
            }
        }
        System.out.println("APA complete");
    }
}