/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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

import com.google.common.primitives.Ints;
import juicebox.HiCGlobals;
import juicebox.data.HiCFileTools;
import juicebox.tools.clt.CommandLineParserForJuicer;
import juicebox.tools.clt.JuicerCLT;
import juicebox.tools.clt.juicer.APA;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.io.File;


/**
 * Aggregated Peak Analysis vs Distance
 * Created by muhammadsaadshamim on 1/19/16.
 * Developed by Fanny Huang
 * Implemented by Nathan Musial
 * <p/>
 * Except for superloops, we don't observe long-range loops. Why not?
 * <p/>
 * The first possibility is that long-range loops do not form, either because:
 * <p/>
 * a) there is some mechanism that creates a hard cap on the length of loops,
 * such as the processivity of the excom, or
 * <p/>
 * b) given a convergent pair A/B separated by >2Mb,
 * there are too many competing ctcf motifs in between.
 * <p/>
 * Alternatively, loops do form between pairs of convergent CTCF sites that are far apart,
 * but those loops are too weak for us to see in our maps.
 * <p/>
 * A simple way to probe this is to do APA. Bin pairs of convergent loop anchors by 1d distance,
 * and then do APA on the pairs in each bin. You should get a strong apa score at 300kb.
 * what about 3mb? 30mb?
 */
public class APAvsDistance extends JuicerCLT  {

    private String hicFilePaths;
    private String PeaksFile;
    private String SaveFolderPath;
    private File   SaveFolder;

    //TODO add new flags for (exponent, numBins)
    //todo appears to have bugs / see commented code for processing flags
    //todo adjust binning algorithm so that there is enough features in each bucket for apa to run
    private int[] resolutions = new int[]{25000};
    private int numBuckets=8;
    private double exponent=2;
    private double minPeakDist=0;
    private double maxPeakDist=30;

    public APAvsDistance() {
        super("APAvsDistance [-n minval] [-x maxval] [-w window] [-r resolution(s)] [-c chromosomes]" +
                " [-k NONE/VC/VC_SQRT/KR] [-q corner_width] [-e include_inter_chr] [-u save_all_data]" +
                " <hicFile(s)> <PeaksFile> <SaveFolder>");
        HiCGlobals.useCache = false;
    }

    public static String getBasicUsage() {
        return "apa_vs_distance <hicFile(s)> <PeaksFile> <SaveFolder>";
    }

    public void initializeDirectly(String inputHiCFileName, String inputPeaksFile, String outputDirectoryPath,
                                   int numBuckets, double exponent, double minPeakDist, double maxPeakDist) {
        this.hicFilePaths=inputHiCFileName;
        this.PeaksFile=inputPeaksFile;
        this.SaveFolderPath=outputDirectoryPath;
        this.numBuckets=numBuckets;
        this.exponent=exponent;
        this.minPeakDist=minPeakDist;
        this.maxPeakDist=maxPeakDist;
    }

    @Override
    protected void readJuicerArguments(String[] args, CommandLineParserForJuicer juicerParser) {

        if (args.length != 4) {
            printUsageAndExit();
        }

        hicFilePaths = args[1];
        PeaksFile = args[2];
        SaveFolderPath=args[3];
        SaveFolder = HiCFileTools.createValidDirectory(args[3]);

        /*
        NormalizationType preferredNorm = juicerParser.getNormalizationTypeOption();
        if (preferredNorm != null)
            norm = preferredNorm;
            */

        double potentialMinPeakDist = juicerParser.getAPAMinVal();
        if (potentialMinPeakDist > -1)
            minPeakDist = potentialMinPeakDist;

        double potentialMaxPeakDist = juicerParser.getAPAMaxVal();
        if (potentialMaxPeakDist > -1)
            maxPeakDist = potentialMaxPeakDist;

        /*
        int potentialWindow = juicerParser.getAPAWindowSizeOption();
        if (potentialWindow > 0)
            window = potentialWindow;
        includeInterChr = juicerParser.getIncludeInterChromosomal();
        saveAllData = juicerParser.getAPASaveAllData();
        List<String> possibleRegionWidths = juicerParser.getAPACornerRegionDimensionOptions();
        if (possibleRegionWidths != null) {
            List<Integer> widths = new ArrayList<>();
            for (String res : possibleRegionWidths) {
                widths.add(Integer.parseInt(res));
            }
            regionWidths = Ints.toArray(widths);
        }

        */
        List<String> possibleResolutions = juicerParser.getMultipleResolutionOptions();
        if (possibleResolutions != null) {
            List<Integer> intResolutions = new ArrayList<>();
            for (String res : possibleResolutions) {
                intResolutions.add(Integer.parseInt(res));
            }
            resolutions = Ints.toArray(intResolutions);
        }
    }

    private static void printResults(String[] windows, double[] results, String SaveFolderPath) {

        File outFolder = new File(SaveFolderPath + "/results.txt");
        try {
            PrintWriter pw = new PrintWriter(outFolder);
            pw.println("PeaktoPeak Distance\tAPA Score");
            for (int i = 0; i < results.length; i++) {
                pw.println(windows[i] + "\t" + results[i]);
            }
            pw.close();
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    private static void plotChart(String SaveFolderPath, XYSeries results) {
        File file = new File(SaveFolderPath + "/results.png");

        final XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(results);

        JFreeChart Chart = ChartFactory.createXYLineChart(
                "APA vs Distance",
                "Distance Bucket",
                "APA Score",
                dataset,
                PlotOrientation.VERTICAL,
                true, true, false);

       /*
       LogarithmicAxis logAxis= new LogarithmicAxis("Distance (log)");
       XYPlot plot= Chart.getXYPlot();
       plot.setDomainAxis(logAxis);
       XYLineAndShapeRenderer renderer = (XYLineAndShapeRenderer)plot.getRenderer();
       renderer.setSeriesShapesVisible(0, true);
       ChartFrame frame = new ChartFrame("My Chart", Chart);
       frame.pack();
       frame.setVisible(true);
       */

        int width = 640;   /* Width of the image */
        int height = 480;  /* Height of the image */

        try {
            ChartUtilities.saveChartAsPNG(file, Chart, width, height);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    @Override
    public void run()  {

        for (int resolution : resolutions) {
            SaveFolder = new File(SaveFolderPath + resolution);
            numBuckets = 8;
            exponent = 2;
            double[] results = new double[numBuckets];
            String[] windows = new String[numBuckets];
            XYSeries XYresults = new XYSeries("APA Result: " + resolution);
            for (int i = 0; i < numBuckets; i++) {
                APA apa = new APA();
                apa.initializeDirectly(hicFilePaths, PeaksFile, SaveFolderPath + File.separator + (int) minPeakDist + "-" + (int) maxPeakDist, new int[]{resolution}, minPeakDist, maxPeakDist);
                windows[i] = minPeakDist + "-" + maxPeakDist;
                System.out.println("Bucket:" + (i + 1) + " Window: " + windows[i]);

                //results[i]=i; //for testing binning algorithm
                results[i] = apa.runWithReturn().getPeak2LL(); // calls APA returns results of apa and gets LL score
                System.out.println(results[i]);

                XYresults.add(Math.log(maxPeakDist), results[i]);
                minPeakDist = maxPeakDist;
                maxPeakDist *= exponent;
            }
            plotChart(SaveFolderPath + resolution, XYresults);
            printResults(windows, results, SaveFolderPath + resolution);
        }
    }
}