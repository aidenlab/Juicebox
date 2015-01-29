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

import juicebox.HiC;
import juicebox.data.*;
import juicebox.tools.HiCTools;
import juicebox.track.Feature2D;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.broad.igv.Globals;
import org.broad.igv.feature.Chromosome;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Map;

public class APA extends JuiceboxCLT {

    private Array2DRowRealMatrix xMatrix;

    private String[] files;
    private double[] bounds;

    private String restrictionSiteFilename = "/aidenlab/restriction_sites/hg19_HindIII.txt";

    //defaults
    double min_peak_dist = 30; // distance between two bins, can be changed in opts
    double max_peak_dist= Double.POSITIVE_INFINITY;
    int window = 10;
    int width=6; //size of boxes
    int peakwidth = 2; //for enrichment calculation of crosshair norm
    int resolution = 10000;
    boolean save_all = false;

    private String workingdirectory = System.getProperty("user.dir");
    private String dataFolder = workingdirectory +"/Data";

    @Override
    public void readArguments(String[] args, HiCTools.CommandLineParser parser) throws IOException {
        //setUsage("juicebox apa <minval maxval window  resolution> input_file.hic PeaksFile/PeaksFolder SaveFolder SavePrefix");

        if (!(args.length == 7 || args.length == 8)) {
            throw new IOException("1");
        }
        files = new String[args.length - 4];

        System.arraycopy(args, 4, files, 0, files.length);

        if (files.length > 4)
            restrictionSiteFilename = files[4];

        bounds = new double[2];

        try {
            bounds[0] = Double.valueOf(args[0]);
            bounds[1] = Double.valueOf(args[1]);
            window = Integer.valueOf(args[2]);
            resolution = Integer.valueOf(args[3]);
        } catch (NumberFormatException error) {
            throw new IOException("2");
        }

    }

    @Override
    public void run(){

        //Calculate parameters that will need later
        int L = 2*window+1;
        int midpoint = window*(2*window + 1) + window; //midpoint of flattened matrix
        int[] shift = APAUtils.range(-window,window+1); //window on which to do psea
        int mdpt = shift.length/2;

        //define gw data structures
        int GW_npeaks = 0;
        int GW_npeaks_used = 0;
        int GW_npeaks_used_nonunique = 0;
        Array2DRowRealMatrix GW_psea = cleanArray2DMatrix(L,L);
        Array2DRowRealMatrix GW_normed_psea = cleanArray2DMatrix(L,L);
        Array2DRowRealMatrix GW_center_normed_psea = cleanArray2DMatrix(L,L);
        Array2DRowRealMatrix GW_rank_psea = cleanArray2DMatrix(L,L);
        Array2DRowRealMatrix GW_coverage = cleanArray2DMatrix(L,L);
        double[] GW_enhancement = new double [0];

        try {
            DatasetReaderV2 reader = new DatasetReaderV2(files[0]);
            Dataset ds = reader.read();

            if (reader.getVersion() < 5) {
                throw new RuntimeException("This file is version " + reader.getVersion() +
                        ". Only versions 5 and greater are supported at this time.");
            }

            List<Chromosome> chromosomes = ds.getChromosomes();
            Map<Chromosome,ArrayList<Feature2D>> chrToLoops =
                    APAUtils.loadLoopList(files[1], new ArrayList<Chromosome>(chromosomes));

            // Loop through chromosomes
            for (Chromosome chr : chromosomes) {

                Array2DRowRealMatrix psea = cleanArray2DMatrix(L,L);
                Array2DRowRealMatrix normed_psea = cleanArray2DMatrix(L,L);
                Array2DRowRealMatrix center_normed_psea = cleanArray2DMatrix(L,L);
                Array2DRowRealMatrix rank_psea = cleanArray2DMatrix(L,L);
                Array2DRowRealMatrix coverage = cleanArray2DMatrix(L,L);
                double[] enhancement = new double [0];
                int npeaks_used = 0, npeaks_used_nonunique;


                if (chr.getName().equals(Globals.CHR_ALL)) continue;

                ArrayList<Feature2D> loops = chrToLoops.get(chr);
                Matrix matrix = ds.getMatrix(chr, chr);
                if (matrix == null) continue;

                List<HiCZoom> resolutions = ds.getBpZooms();
                HiCZoom zoom = resolutions.get(0);
                int currentDistance = Math.abs(zoom.getBinSize() - resolution);
                // Loop through resolutions
                for (HiCZoom subZoom : resolutions) {
                    int newDistance = Math.abs(subZoom.getBinSize() - resolution);
                    if (newDistance < currentDistance) {
                        currentDistance = newDistance;
                        zoom=subZoom;
                    }
                }
                MatrixZoomData zd = matrix.getZoomData(zoom);

                // TODO filter loops
                // TODO loop num statistics

                for (Feature2D loop : loops){

                    int loopX = loop.getStart1(), loopY = loop.getStart2();

                    int binXStart = loopX - resolution*(window+1);
                    int binXEnd = loopX + resolution*(window+1);
                    int binYStart = loopY - resolution*(window+1);
                    int binYEnd = loopY + resolution*(window+1);

                    List<Block> blocks = zd.getNormalizedBlocksOverlapping(binXStart, binYStart, binXEnd, binYEnd,
                            NormalizationType.NONE);
                    for (Block b : blocks) {
                        for (ContactRecord rec : b.getContactRecords()) {
                            //, rec.getBinY(), rec.getCounts()

                            int relativeX = window + (rec.getBinX() - loopX)/resolution;
                            int relativeY = window + (rec.getBinY() - loopY)/resolution;

                            if(relativeX >= 0 && relativeX < L){
                                if(relativeY >= 0 && relativeY < L){
                                    psea.addToEntry(relativeX, relativeY, rec.getCounts());
                                }
                            }

                        }
                    }
                }


                GW_psea.add(psea);
                GW_normed_psea.add(normed_psea);
                GW_center_normed_psea.add(center_normed_psea);
                GW_rank_psea.add(rank_psea);
                // TODO GW_enhancement;

                double scalar = 1./npeaks_used;
                normed_psea.scalarMultiply(scalar);
                center_normed_psea.scalarMultiply(scalar);
                rank_psea.scalarMultiply(scalar);

                saveDataSet(window, psea, normed_psea, rank_psea, enhancement, ""+chr.getName());

            }
        }
        catch (IOException e){
            System.out.println("Unable to extract APA data");
            e.printStackTrace();
            System.exit(-3);
        }



        double GW_npeaks_used_inv = 1./GW_npeaks_used;
        GW_normed_psea.scalarMultiply(GW_npeaks_used_inv);
        GW_center_normed_psea.scalarMultiply(GW_npeaks_used_inv);
        GW_rank_psea.scalarMultiply(GW_npeaks_used_inv);


        saveDataSet(window, GW_psea, GW_normed_psea, GW_rank_psea, GW_enhancement, "GW");

        System.err.println("This method is not currently implemented.");
        System.exit(7);
    }


    private Array2DRowRealMatrix cleanArray2DMatrix(int rows, int cols){
        Array2DRowRealMatrix matrix = new Array2DRowRealMatrix(rows,cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                matrix.setEntry(r,c,0);
        return matrix;
    }


    private void saveDataSet(int mdpt, Array2DRowRealMatrix psea, Array2DRowRealMatrix normed_psea,
                             Array2DRowRealMatrix rank_psea, double[] enhancement, String prefix){

        dataFolder += "."+prefix + new SimpleDateFormat("yyyy.MM.dd.HH.mm").format(new Date());
        File dataDir = new File(dataFolder);
        dataDir.mkdir();
        System.out.println("Saving "+prefix+" data to " + dataFolder);

        //save GW data
        APAUtils.saveMatrixText(dataFolder + "/"+prefix+"_psea.txt", psea);
        APAUtils.saveMatrixText(dataFolder + "/"+prefix+"_normed_psea.txt", normed_psea);
        APAUtils.saveMatrixText(dataFolder + "/"+prefix+"_rank_psea.txt", rank_psea);
        APAUtils.saveArrayText(dataFolder + "/" + prefix + "_enhancement.txt", enhancement);
        APAUtils.saveMeasures(psea, mdpt, dataFolder + "/"+prefix+"_measures.txt");
    }
}