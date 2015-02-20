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

import juicebox.data.*;
import juicebox.tools.HiCTools;
import juicebox.tools.utils.APAUtils;
import juicebox.track.Feature2D;
import juicebox.windowui.HiCZoom;
import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.broad.igv.Globals;
import org.broad.igv.feature.Chromosome;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 *
 * TODO - once fully debugged, change notation convention from underscore to camelcase (match the rest of the files)
 */
public class APA extends JuiceboxCLT {

    private String[] files;

    private String restrictionSiteFilename = "/aidenlab/restriction_sites/hg19_HindIII.txt";

    public APA(){
        super("apa [-n minval] [-x maxval] [-w window]  [-r resolution] <CountsFolder> <PeaksFile> <SaveFolder> [SavePrefix] [RestrictionSiteFile]");
    }

    //defaults
    private double min_peak_dist = 30; // distance between two bins, can be changed in opts
    private double max_peak_dist= Double.POSITIVE_INFINITY;
    private int window = 10;
    int width=6; //size of boxes
    int peakwidth = 2; //for enrichment calculation of crosshair norm
    private int resolution = 10000;
    private final boolean saveAllData = true;

    private final String workingdirectory = System.getProperty("user.dir");

    @Override
    public void readArguments(String[] args, HiCTools.CommandLineParser parser) throws IOException {

        if (!(args.length > 3 && args.length < 7)) {
            throw new IOException("1");
        }


        files = new String[args.length-1];
        System.arraycopy(args, 1, files, 0, files.length);

        for(String s : files)
            System.out.println("--- "+s);

        if (files.length > 4)
            restrictionSiteFilename = files[4];

        Number[] optionalAPAFlags = parser.getAPAOptions();

        if(optionalAPAFlags[0] != null)
            min_peak_dist = optionalAPAFlags[0].doubleValue();

        if(optionalAPAFlags[1] != null)
            max_peak_dist = optionalAPAFlags[1].doubleValue();

        if(optionalAPAFlags[2] != null)
            window = optionalAPAFlags[2].intValue();

        if(optionalAPAFlags[3] != null)
            resolution = optionalAPAFlags[3].intValue();

    }

    @Override
    public void run(){

        //Calculate parameters that will need later
        int L = 2*window+1;
        //int midpoint = window*(2*window + 1) + window; //midpoint of flattened matrix
        //int[] shift = APAUtils.range(-window,window+1); //window on which to do psea
        //int mdpt = shift.length/2;

        //define gw data structures
        //int gw_npeaks = 0;
        int gw_npeaks_used = 0;
        //int gw_npeaks_used_nonunique = 0;
        Array2DRowRealMatrix gw_psea = APAUtils.cleanArray2DMatrix(L, L);
        Array2DRowRealMatrix gw_normed_psea = APAUtils.cleanArray2DMatrix(L, L);
        Array2DRowRealMatrix gw_center_normed_psea = APAUtils.cleanArray2DMatrix(L, L);
        Array2DRowRealMatrix gw_rank_psea = APAUtils.cleanArray2DMatrix(L, L);
        //Array2DRowRealMatrix gw_coverage = APAUtils.cleanArray2DMatrix(L, L);
        List<Double> gw_enhancement = new ArrayList<Double>();

        try {
            System.out.println("Accessing "+files[0]);
            DatasetReaderV2 reader = new DatasetReaderV2(files[0]);
            Dataset ds = reader.read();

            if (reader.getVersion() < 5) {
                throw new RuntimeException("This file is version " + reader.getVersion() +
                        ". Only versions 5 and greater are supported at this time.");
            }

            List<Chromosome> chromosomes = ds.getChromosomes();
            Map<Chromosome,ArrayList<Feature2D>> chrToLoops =
                    APAUtils.loadLoopList(files[1], new ArrayList<Chromosome>(chromosomes), min_peak_dist, max_peak_dist);

            Set<Chromosome> commonChromosomes = getSetIntersection(chrToLoops.keySet(),
                    new HashSet<Chromosome>(chromosomes));

            // Loop through chromosomes
            for (Chromosome chr : commonChromosomes) {



                Array2DRowRealMatrix psea = APAUtils.cleanArray2DMatrix(L, L);
                Array2DRowRealMatrix normed_psea = APAUtils.cleanArray2DMatrix(L, L);
                Array2DRowRealMatrix center_normed_psea = APAUtils.cleanArray2DMatrix(L, L);
                Array2DRowRealMatrix rank_psea = APAUtils.cleanArray2DMatrix(L, L);
                //Array2DRowRealMatrix coverage = APAUtils.cleanArray2DMatrix(L, L);
                List<Double> enhancement = new ArrayList<Double>();

                if (chr.getName().equals(Globals.CHR_ALL)) continue;

                // get all loops
                if(chrToLoops.containsKey(chr)) {
                    ArrayList<Feature2D> loops = chrToLoops.get(chr);
                    int npeaks_used_nonunique = loops.size();
                    //gw_npeaks_used_nonunique += npeaks_used_nonunique;

                    // remove repeats
                    Set<Feature2D> uniqueLoops = new HashSet<Feature2D>(loops);
                    int npeaks_used = loops.size();
                    gw_npeaks_used += npeaks_used;

                    Matrix matrix = ds.getMatrix(chr, chr);
                    if (matrix == null) continue;

                    // select zoom level closest to the requested one
                    List<HiCZoom> resolutions = ds.getBpZooms();
                    HiCZoom zoom = resolutions.get(0);
                    int currentDistance = Math.abs(zoom.getBinSize() - resolution);
                    // Loop through resolutions
                    for (HiCZoom subZoom : resolutions) {
                        int newDistance = Math.abs(subZoom.getBinSize() - resolution);
                        if (newDistance < currentDistance) {
                            currentDistance = newDistance;
                            zoom = subZoom;
                        }
                    }
                    resolution = zoom.getBinSize();
                    System.out.println("Adjusting resolution to " + resolution);
                    MatrixZoomData zd = matrix.getZoomData(zoom);

                    for (Feature2D loop : uniqueLoops) {
                        Array2DRowRealMatrix newData = APAUtils.extractLocalizedData(zd, loop, L, resolution, window);
                        psea.add(newData);
                        normed_psea.add(APAUtils.standardNormalization(newData));
                        center_normed_psea.add(APAUtils.centerNormalization(newData));
                        rank_psea.add(APAUtils.rankPercentile(newData));
                        enhancement.add(APAUtils.peakEnhancement(newData));
                    }


                    gw_psea.add(psea);
                    gw_normed_psea.add(normed_psea);
                    gw_center_normed_psea.add(center_normed_psea);
                    gw_rank_psea.add(rank_psea);
                    gw_enhancement.addAll(enhancement);

                    double npeaks_used_inv = 1. / npeaks_used;
                    normed_psea.scalarMultiply(npeaks_used_inv);
                    center_normed_psea.scalarMultiply(npeaks_used_inv);
                    rank_psea.scalarMultiply(npeaks_used_inv);

                    if (saveAllData)
                        saveDataSet(window, psea, normed_psea, rank_psea, enhancement, chr.getName());
                }
            }
        }
        catch (IOException e){
            System.out.println("Unable to extract APA data");
            e.printStackTrace();
            System.exit(-3);
        }



        double gw_npeaks_used_inv = 1./gw_npeaks_used;
        gw_normed_psea.scalarMultiply(gw_npeaks_used_inv);
        gw_center_normed_psea.scalarMultiply(gw_npeaks_used_inv);
        gw_rank_psea.scalarMultiply(gw_npeaks_used_inv);

        saveDataSet(window, gw_psea, gw_normed_psea, gw_rank_psea, gw_enhancement, "gw");

        System.exit(7);
    }



    //http://stackoverflow.com/questions/7574311/efficiently-compute-intersection-of-two-sets-in-java
    private static Set<Chromosome> getSetIntersection (Set<Chromosome> set1, Set<Chromosome> set2) {
        boolean set1IsLarger = set1.size() > set2.size();
        Set<Chromosome> cloneSet = new HashSet<Chromosome>(set1IsLarger ? set2 : set1);
        cloneSet.retainAll(set1IsLarger ? set1 : set2);
        return cloneSet;
    }


    private void saveDataSet(int mdpt, Array2DRowRealMatrix psea, Array2DRowRealMatrix normed_psea,
                             Array2DRowRealMatrix rank_psea, List<Double> enhancement, String prefix){
        System.out.println(files[2]);
        File dataDirHome = new File(files[2]);
        if(!dataDirHome.exists()){
            boolean result = dataDirHome.mkdir();
            if(!result){
                System.out.println("Error creating directory (data not saved): " + dataDirHome);
                return;
            }
        }

        String dataFolder = dataDirHome.getAbsolutePath() + "/" + prefix + "_" + new SimpleDateFormat("yyyy.MM.dd.HH.mm").format(new Date());
        File dataDir = new File(dataFolder);
        boolean result = dataDir.mkdir();
        if(!result){
            System.out.println("Error creating directory (data not saved): " + dataDir);
            return;
        }

        System.out.println("Saving "+prefix+" data to " + dataFolder);

        //save gw data
        APAUtils.saveMatrixText(dataFolder + "/"+prefix+"_psea.txt", psea);
        APAUtils.saveMatrixText(dataFolder + "/"+prefix+"_normed_psea.txt", normed_psea);
        APAUtils.saveMatrixText(dataFolder + "/"+prefix+"_rank_psea.txt", rank_psea);
        APAUtils.saveListText(dataFolder + "/" + prefix + "_enhancement.txt", enhancement);
        APAUtils.saveMeasures(dataFolder + "/"+prefix+"_measures.txt", psea);
    }
}