/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2017 Broad Institute, Aiden Lab
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
import jcuda.utils.Print;
import juicebox.HiCGlobals;
import juicebox.data.ChromosomeHandler;
import juicebox.data.HiCFileTools;
import juicebox.data.anchor.MotifAnchor;
import juicebox.data.anchor.MotifAnchorParser;
import juicebox.data.feature.Feature;
import juicebox.data.feature.GenomeWideList;
import juicebox.tools.clt.CommandLineParser;
import juicebox.tools.clt.CommandLineParserForJuicer;
import juicebox.tools.clt.JuicerCLT;
import juicebox.tools.clt.juicer.APA;
import juicebox.tools.clt.juicer.MotifFinder;
import juicebox.tools.utils.juicer.apa.APARegionStatistics;
import juicebox.tools.utils.juicer.apa.APAUtils;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.Feature2DParser;
import juicebox.track.feature.FeatureFilter;
import juicebox.windowui.NormalizationType;

import org.jfree.chart.*;
import org.jfree.chart.axis.LogarithmicAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLine3DRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;




import java.io.*;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;


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

    //Defaults todo adjust binning algorithm so that there is enough features in each bucket for apa to run
    private int[]resolutions;
    private int numBuckets=8;
    private double exponent=2;
    private double minPeakDist=0;
    private double maxPeakDist=30;




    public APAvsDistance(){ //TODO add new flags for (exponent, numBins)
        super("APAvsDistance [-n minval] [-x maxval] [-w window] [-r resolution(s)] [-c chromosomes]" +
                " [-k NONE/VC/VC_SQRT/KR] [-q corner_width] [-e include_inter_chr] [-u save_all_data]" +
                " <hicFile(s)> <PeaksFile> <SaveFolder>");
        HiCGlobals.useCache = false;


    }
    public static String getBasicUsage() {
        return "APAvsDistance <hicFile(s)> <PeaksFile> <SaveFolder>";
    } //todo change to match apa vs distance


    public void initializeDirectly(String inputHiCFileName, String inputPeaksFile, String outputDirectoryPath, int numBuckets, double exponent,double
                                   minPeakDist, double maxPeakDist){

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
        if (potentialMinPeakDist >= 0)
            minPeakDist = potentialMinPeakDist;

        double potentialMaxPeakDist = juicerParser.getAPAMaxVal();
        if (potentialMaxPeakDist > 0)
            maxPeakDist = potentialMaxPeakDist;

        /*
        int potentialWindow = juicerParser.getAPAWindowSizeOption();
        if (potentialWindow > 0)
            window = potentialWindow;
            */
/*
        includeInterChr = juicerParser.getIncludeInterChromosomal();

        saveAllData = juicerParser.getAPASaveAllData();

        */

/*
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



   @Override
    public void run()  {


       hicFilePaths="/Users/nathanielmusial/CS_Projects/SMART_Projects/Testing_Files/HiC/gm12878_intra_nofrag_30.hic";//.Hic
       PeaksFile="/Users/nathanielmusial/CS_Projects/SMART_Projects/Testing_Files/Other/GM12878_loop_list.txt";//.txt
       SaveFolderPath="/Users/nathanielmusial/CS_Projects/SMART_Projects/Output";
       SaveFolder= new File(SaveFolderPath);
       numBuckets=8;

       exponent=2;
      // minPeakDist=30;
      // maxPeakDist=40;
       //  minPeakDist=0;
       //maxPeakDist=30;
       double[] results= new double[numBuckets];
       String[] windows= new String[numBuckets];
       XYSeries XYresults=new XYSeries("APA Result");
       File outputDirectory;
       APA apa;



       //APA apa1= new APA();


      for(int i=0;i<numBuckets;i++)
      {
          apa=new APA();
         // apa1.hicFilePaths = HiCFiles;
         // apa1.loopListPath = PeaksFile;
         // outputDirectory = new File(SaveFolderPath+"/"+(int)minPeakDist+"-"+(int)maxPeakDist);//cut off decimals
         // outputDirectory.mkdir();
         // apa1.outputDirectory =outPutDirectory;

         // apa1.resolutions = new int[]{25000};
           resolutions = new int[]{25000};


          //apa1.minPeakDist=minPeakDist;
         // apa1.maxPeakDist=maxPeakDist;

          apa.initializeDirectly(hicFilePaths,PeaksFile,SaveFolderPath+"/"+(int)minPeakDist+"-"+(int)maxPeakDist,resolutions,minPeakDist,maxPeakDist);
          windows[i]=minPeakDist+"-"+maxPeakDist;



          //apa1.maxPeakDistance/minPeakDistance
          System.out.println("Bucket:"+(i+1)+" Window: "+minPeakDist+"-"+maxPeakDist);



           results[i]=apa.runWithReturn().getPeak2LL();
          //APARegionStatistics
       //   System.out.println(results[i]);
        //if specifes too many bins
         //results[i]=i;
          XYresults.add(Math.log(maxPeakDist),results[i]);
          minPeakDist=maxPeakDist;
          maxPeakDist*=exponent;
          //if(i==numBuckets-2)
            //  maxPeakDist=Double.POSITIVE_INFINITY;

          //apa=null;

      }
       plotChart(SaveFolderPath,XYresults);
      printResults(windows,results,SaveFolderPath);





   }

   private static void printResults(String[] windows,double[] results, String SaveFolderPath){

      File outFolder= new File(SaveFolderPath+"/results.txt");


       try {
           PrintWriter pw= new PrintWriter(outFolder);
            pw.println("PeaktoPeak Distance\tAPA Score");
           for( int i =0; i<results.length;i++){
               pw.println(windows[i]+"\t"+results[i]);

           }
           pw.close();

       }
       catch (IOException ex) {
           ex.printStackTrace();
       }



    }


   private static void plotChart(String SaveFolderPath,XYSeries results){
       File file= new File(SaveFolderPath+"/results.png");


       final XYSeriesCollection dataset = new XYSeriesCollection( );
       dataset.addSeries( results );


       JFreeChart Chart = ChartFactory.createXYLineChart(
               "APA vs Distance",
               "Distance Bucket",
               "APA Score",
               dataset,
               PlotOrientation.VERTICAL,
               true, true, false);

       // LogarithmicAxis logAxis= new LogarithmicAxis("Distance (log)");
        //XYPlot plot= Chart.getXYPlot();
       // plot.setDomainAxis(logAxis);
        //Chart.


       //XYLineAndShapeRenderer renderer = (XYLineAndShapeRenderer)plot.getRenderer();
       //renderer.setSeriesShapesVisible(0, true);

       //ChartFrame frame = new ChartFrame("My Chart", Chart);
       //frame.pack();
       //frame.setVisible(true);

       int width = 640;   /* Width of the image */
       int height = 480;  /* Height of the image */
       //File XYChart = new File( "XYLineChart.jpeg" );

       try {
           ChartUtilities.saveChartAsPNG( file, Chart, width, height);
          // ChartUtilities.

       }
        catch (IOException ex) {
           ex.printStackTrace();
       }
    }




    public static void bin(String loopListPath, ChromosomeHandler handler, String outputDirectory, final int initialCutoff, int exponent, int resolution){

        int minPeakDist=0;
        int maxPeakDist=initialCutoff;
        String outputPath;
        
        for (int i=1;i<10;i++)
        {
            outputPath=outputDirectory+"/bin_"+i+"_"+minPeakDist+"-"+maxPeakDist;
            bin(outputPath,loopListPath,handler,minPeakDist,maxPeakDist,resolution);
            minPeakDist=maxPeakDist;
            maxPeakDist+=maxPeakDist*exponent;
        }
    }


    private static void bin(String outputPath, String loopListPath, ChromosomeHandler handler, final double minPeakDist, final double maxPeakDist, final int resolution) {
        Feature2DList loopList = Feature2DParser.loadFeatures(loopListPath, handler, false,
                new FeatureFilter() {
                    // Remove duplicates and filters by size
                    // also save internal metrics for these measures
                    @Override
                    public List<Feature2D> filter(String chr, List<Feature2D> features) {

                        List<Feature2D> uniqueFeatures = new ArrayList<>(new HashSet<>(features));
                        return APAUtils.filterFeaturesBySize(uniqueFeatures,
                                minPeakDist, maxPeakDist, resolution);
                        /*
                        List<Feature2D> filteredUniqueFeatures = APAUtils.filterFeaturesBySize(uniqueFeatures,
                                minPeakDist, maxPeakDist, resolution);


                            filterMetrics.put(chr,
                                    new Integer[]{filteredUniqueFeatures.size(), uniqueFeatures.size(), features.size()});
                            */

                       // return filteredUniqueFeatures;
                    }
                }, false);
        File outputFile = new File(outputPath);
        loopList.exportFeatureList(outputFile, false, Feature2DList.ListFormat.NA);
    }


}



        // preservative intersection of these protein list with motif list


        // extract positive anchors and negative anchors

        // create loops from all possible valid intersections

        // Feature2DList

        // bin loops by distance between loci


        // calculate APA score for each bin_list



        // plot APA score vs binned distance

        /*
         * Detailed psuedo code
         * Main
         Read In Loop File
            Bin by distance
                Not sure if want to write files or create a linked list of (loop lists) and pass into
            Get apa score
            Plot apa graph
            Get apa chart

         Linked List<loop> readInLoopFile (loop file)
            Open loopfile
            While next line
                 Read line
                 Split by tabs
                 Create Linked List<loop>
                 List.add (line[0:6])
                 Close file



         Void Bin by distance ( loop list, int start dist, int exponent, wdir? ,output path ) : binned loop files
            int bucket=1
             create bucket 1
             int offset= initialCutoff
             int cutoff = minDistance+offset
             if distance < cutoff
                add to bucket
             Else
                 close bucket
                 cutoff = cutoff+expOffset
                 offset= offset*exponent
             Run apa score
                insert logiv

             Get apa score (apa tool path,
                insert logic
             Plot apa graph
                insert logic
             Get apa graph
                insert logic








public static void main()
{
        String outputDir;
        int initialCutoff;
        int exponent;
        String loopFilePath;
        int exponent;

        Map<Integer, String> map = sortByValues(hmap); //test

        Feature2DList looplist = Feature2DParser.loadFeatures(loopFilePath, commonChromosomesHandler, true, null, false);
        looplist=sortByDiffDistance(looplist);
        bin(looplist, initialCutoff, exponent, outputDir);

        /*
        Then need to calucualte apa score for each binned loop file
        create graph with info from apa and plot apa score vs distance
        obtain apa heatmap from same folder that recived score from

        */


/*

  private static HashMap sortByValues(HashMap map) {
       List list = new LinkedList(map.entrySet());
       // Defined Custom Comparator here
       Collections.sort(list, new Comparator() {
            public int compare(Object o1, Object o2) {
               return ((Comparable) ((Map.Entry) (o1)).getValue())
                  .compareTo(((Map.Entry) (o2)).getValue());
            }
       });

       // Here I am copying the sorted list in HashMap
       // using LinkedHashMap to preserve the insertion order
       HashMap sortedHashMap = new LinkedHashMap();
       for (Iterator it = list.iterator(); it.hasNext();) {
              Map.Entry entry = (Map.Entry) it.next();
              sortedHashMap.put(entry.getKey(), entry.getValue());
       }
       return sortedHashMap;
  }



    public static Feature2DList sortByDiffDistance(Feature2DList){
        // may be a way to extend
    }

    public static int getDiffDistance(2DFeature feature)
    { return feature.getStart2-feature.getStart1;}

    public void Feature2DList bin( Feature2DList loopList, int cutoff, int exponent, String outputDir) { //accepts a sorted loops list by Diffdistance
        int counter=1;
        int offset=cutoff;
        Feature2DList currentBin = new Feature2DList();//want a blank feature 2d list
        Iterator it = looplist.entrySet().iterator(); //interate though Feature2DList
        while (it.hasNext()) {

            Map.Entry pair = (Map.Entry)it.next();
            2DFeature feature = pair.getValue());

            if(getDiffDistance(feature) < cutoff)
            {
                currentBin.add(feature)
                it.remove(); // avoids a ConcurrentModificationException
            }
            else
            {
                counter++;
                currentBin.tofile(outputDir, etc...);  //write currentBin to file with path name bin number and cutoff size
                currentBin.close();
                currentBin = new Feature2DList;
                cutoff=cutoff+offset;
                offset=offset*exponent;
            }
        }
    }



         */











/*
        GenomeWideList<MotifAnchor> motifs = MotifAnchorParser.loadMotifsFromGenomeID("hg19", null);
        ChromosomeHandler handler = HiCFileTools.loadChromosomes("hg19");

        // read in all smc3, rad21, ctcf tracks and intersect them
        List<String> bedFiles = new ArrayList<>();

        File folder = new File("/users/name" + "directoryPath");
        File[] listOfFiles = folder.listFiles();

        for (File file : listOfFiles != null ? listOfFiles : new File[0]) {
            if (file.isFile()) {
                String path = file.getAbsolutePath();
                if (path.endsWith(".bed")) {
                    bedFiles.add(path);
                }
            }
        }

        GenomeWideList<MotifAnchor> proteins = MotifFinder.getIntersectionOfBEDFiles(handler, bedFiles);
         */