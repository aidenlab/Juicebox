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

import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.data.Dataset;
import juicebox.data.HiCFileTools;
import juicebox.data.Matrix;
import juicebox.data.MatrixZoomData;
import juicebox.tools.clt.CommandLineParserForJuicer;
import juicebox.tools.clt.JuicerCLT;
import juicebox.tools.utils.juicer.afa.AFAUtils;
import juicebox.tools.utils.juicer.afa.LocationType;
import juicebox.tools.utils.juicer.apa.APADataStack;
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
 * Aggregate Feature Analysis (adapted from APA)
 *
 * @author mshamim
 */
public class AFA extends JuicerCLT {

    private static final int regionWidth = 6; //size of boxes
    private final boolean saveAllData = true;
    private String[] files;

    //defaults
    private int window = 30;
    private int[] resolutions = new int[]{25000, 10000};
    private NormalizationType norm;
    private LocationType relativeLocation = LocationType.TL;
    private Set<String> attributes;
    private boolean thresholdPlots = true;

    /**
     * Usage for AFA
     */
    public AFA() {
        super("afa [-w window]  [-r resolution(s)] [-c chromosomes] [-a attribute(s)] [-l TopLeft/BottomRight/Center] <NONE/VC/VC_SQRT/KR> <hic file(s)> <FeatureFile> <SaveFolder> [SavePrefix]");
        HiCGlobals.useCache = false;
    }

    @Override
    protected void readJuicerArguments(String[] args, CommandLineParserForJuicer juicerParser) {
        if (!(args.length > 4 && args.length < 7)) {
            printUsage();
        }

        try {
            norm = NormalizationType.valueOf(args[1]);
        } catch (IllegalArgumentException error) {
            System.err.println("Normalization must be one of \"NONE\", \"VC\", \"VC_SQRT\", \"KR\", \"GW_KR\", \"GW_VC\", \"INTER_KR\", or \"INTER_VC\".");
            System.err.println("Value given: " + args[1]);
            System.exit(-1);
        }


        files = new String[4];
        files[3] = "";
        System.arraycopy(args, 2, files, 0, args.length - 2);


        String location = juicerParser.getRelativeLocationOption();
        if (location != null && location.length() > 0) {
            relativeLocation = LocationType.enumValueFromString(location);
            if (relativeLocation == null) {
                System.err.println("RelativeLocation " + location + " is an invalid type or has a syntax error");
                printUsage();
            }
        }

        attributes = new HashSet<String>(juicerParser.getAttributeOption());
        if (attributes == null || attributes.size() == 0) {
            attributes = new HashSet<String>();
            attributes.add("");
        }

        for (String s : files)
            System.out.println("--- " + s);

        window = juicerParser.getAPAWindowSizeOption();
        if (window == 0)
            window = 30;

        List<String> possibleResolutions = juicerParser.getMultipleResolutionOptions();
        if (possibleResolutions != null) {
            resolutions = new int[possibleResolutions.size()];
            int i = 0;
            for (String res : possibleResolutions) {
                resolutions[i] = Integer.parseInt(res);
                i++;
            }
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
        for (int resolution : HiCFileTools.filterResolutions(ds, resolutions)) {

            System.out.println("res " + resolution);
            HiCZoom zoom = new HiCZoom(HiC.Unit.BP, resolution);

            List<Chromosome> chromosomes = ds.getChromosomes();
            if (givenChromosomes != null)
                chromosomes = new ArrayList<Chromosome>(HiCFileTools.stringToChromosomes(givenChromosomes,
                        chromosomes));

            for (final String attribute : attributes) {


                // Metrics resulting from apa filtering
                final Map<String, Integer[]> filterMetrics = new HashMap<String, Integer[]>();

                Feature2DList featureList = Feature2DParser.loadFeatures(files[1], chromosomes, true,
                        new FeatureFilter() {
                            // Remove duplicates and filters by size
                            // also save internal metrics for these measures
                            @Override
                            public List<Feature2D> filter(String chr, List<Feature2D> features) {

                                List<Feature2D> uniqueFeatures = new ArrayList<Feature2D>(new HashSet<Feature2D>(features));

                                List<Feature2D> filteredUniqueFeatures;
                                if (attribute.length() > 0) {
                                    filteredUniqueFeatures = AFAUtils.filterFeaturesByAttribute(uniqueFeatures, attribute);
                                } else {
                                    System.out.println("No filtering by attribute");
                                    filteredUniqueFeatures = uniqueFeatures;
                                }

                                filterMetrics.put(chr,
                                        new Integer[]{filteredUniqueFeatures.size(), uniqueFeatures.size(), features.size()});

                                return filteredUniqueFeatures;
                            }
                        }, false);

                for (Chromosome chr : chromosomes) {
                    APADataStack apaDataStack = new APADataStack(L, files[2], (files[0] + "_" + resolution + "_" + attribute).replace("/", "_"));

                    if (chr.getName().equals(Globals.CHR_ALL)) continue;

                    Matrix matrix = ds.getMatrix(chr, chr);
                    if (matrix == null) continue;

                    MatrixZoomData zd = matrix.getZoomData(zoom);

                    System.out.println("CHR " + chr.getName() + " " + chr.getIndex());
                    List<Feature2D> features = featureList.get(chr.getIndex(), chr.getIndex());
                    if (features == null || features.size() == 0) {
                        System.out.println("CHR " + chr.getName() + " - no loops, check loop filtering constraints");
                        continue;
                    }

                    Integer[] peakNumbers = filterMetrics.get(Feature2DList.getKey(chr, chr));


                    if (features.size() != peakNumbers[0])
                        System.out.println("Error reading stat numbers fro " + chr);

                    for (int i = 0; i < peakNumbers.length; i++) {
                        //System.out.println(chr + " " + i + " " + peakNumbers[i] + " " + gwPeakNumbers[i]);
                        gwPeakNumbers[i] += peakNumbers[i];
                    }

                    //System.out.println("Loop");
                    for (Feature2D feature : features) {
                        //System.out.println(loop.getMidPt1()/resolution +"\t"+loop.getMidPt2()/resolution);
                        try {
                            apaDataStack.addData(AFAUtils.extractLocalizedData(zd, feature, L, resolution, window,
                                    norm, relativeLocation));
                        } catch (IOException e) {
                            System.err.println("Data not available for feature: " + feature);
                        }
                    }

                    if (thresholdPlots)
                        apaDataStack.thresholdPlots(5000);
                    apaDataStack.updateGenomeWideData();
                    if (saveAllData)
                        apaDataStack.exportDataSet(chr.getName(), peakNumbers, regionWidth);

                }
                // TODO update. actually, AFA should extend APA or vice versa; or be eliminated completely
                APADataStack.exportGenomeWideData(gwPeakNumbers, regionWidth);
                APADataStack.clearAllData();

            }
        }
        System.out.println("AFA complete");
    }
}