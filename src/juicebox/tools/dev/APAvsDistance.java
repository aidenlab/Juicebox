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
import juicebox.data.feature.GenomeWideList;
import juicebox.tools.clt.juicer.MotifFinder;

import java.io.File;
import java.util.ArrayList;
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
class APAvsDistance {

    public static void main() {



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

         Loop object
         String Chr 1, Int X1, Int X2, String Chr 2, Int Y1, Int Y2, String Color, Int distance
         calcDis
         Return y1-x1
         Getters and setters
         constructor(chr1, x1 , x2, chr2, y1 ,y2 , color)
         this=*;
         Distance = calculate distance







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

    }
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






    }




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