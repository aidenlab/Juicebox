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

import juicebox.data.ChromosomeHandler;
import juicebox.data.HiCFileTools;
import juicebox.data.anchor.MotifAnchor;
import juicebox.data.anchor.MotifAnchorParser;
import juicebox.data.feature.Feature;
import juicebox.data.feature.GenomeWideList;
import juicebox.tools.clt.JuicerCLT;
import juicebox.tools.clt.juicer.MotifFinder;
import juicebox.tools.utils.juicer.apa.APAUtils;
import juicebox.track.feature.*;
import juicebox.windowui.NormalizationType;


import java.io.*;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;


/**
 * Created by muhammadsaadshamim on 1/19/16.
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
public class APAvsDistance extends JuicerCLT implements  {

    public APAvsDistance(){
        super("apa [-n minval] [-x maxval] [-w window] [-r resolution(s)] [-c chromosomes]" +
                " [-k NONE/VC/VC_SQRT/KR] [-q corner_width] [-e include_inter_chr] [-u save_all_data]" +
                " <hicFile(s)> <PeaksFile> <SaveFolder>");
        run();
    }

   @Override
    public void run() {
        String HiCFiles="/Users/nathanielmusial/CS_Projects/SMART_Projects/Testing_Files";
        String PeaksFile="/Users/nathanielmusial/CS_Projects/SMART_Projects/Testing_Files";
        String SaveFolder="/Users/nathanielmusial/CS_Projects/SMART_Projects/Output";
        int resolution=5000;
        String chromosomes="Chr19";
        NormalizationType preferredNorm=NormalizationType.KR;// Knight-Ruiz

        List<ChromosomeHandler> ChromList=new List<ChromosomeHandler>;
        ChromosomeHandler handler=new ChromosomeHandler(ChromList);
        int initialCutoff=5000;
        int exponent=2;
        //int resolution;



        bin(PeaksFile,handler,SaveFolder,initialCutoff,exponent,resolution);



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