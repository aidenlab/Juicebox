/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2021 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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
 *  FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.tools.clt;


import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.data.*;
import juicebox.data.basics.Chromosome;
import juicebox.matrix.BasicMatrix;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.tools.utils.juicer.hiccups.HiCCUPSConfiguration;
import juicebox.tools.utils.juicer.hiccups.HiCCUPSUtils;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.Feature2DParser;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationHandler;
import juicebox.windowui.NormalizationType;

import java.io.File;
import java.util.*;

/**
 * Created by muhammadsaadshamim on 7/22/15.
 */
class UnitTests {

    public static void pearsonsAndEigenvector() {

        List<String> files = new ArrayList<>();
        files.add("/Users/mshamim/Desktop/hicfiles/gm12878_rh14_30.hic");
        Dataset ds = HiCFileTools.extractDatasetForCLT(files, false);
        Chromosome chrom = ds.getChromosomeHandler().getChromosomeFromName("10");
        Matrix matrix = ds.getMatrix(chrom, chrom);
        HiCGlobals.MAX_PEARSON_ZOOM = 50000;
        HiCZoom zoom = new HiCZoom(HiC.Unit.BP, 50000);
        MatrixZoomData zd = matrix.getZoomData(zoom);
        ExpectedValueFunction df = ds.getExpectedValues(zoom, NormalizationHandler.KR);
        HiCGlobals.guiIsCurrentlyActive = true;
        long time0 = System.nanoTime();
        BasicMatrix bm1 = zd.getPearsons(df);
        long time1 = System.nanoTime();
        System.out.println("Pearsons Time: " + (time1 - time0) * 1e-9);

        long Etime0 = System.nanoTime();
        double[] eig = zd.computeEigenvector(df, 0);
        long Etime1 = System.nanoTime();
        System.out.println("Eig Time: " + (Etime1 - Etime0) * 1e-9);


        MatrixTools.saveMatrixTextNumpy("/Users/mshamim/Desktop/research/pearson/c20.npy", toDenseMatrix(bm1));

        MatrixTools.saveMatrixTextNumpy("/Users/mshamim/Desktop/research/pearson/E20.npy", eig);
    }

    private static float[][] toDenseMatrix(BasicMatrix bm1) {
        float[][] vals = new float[bm1.getRowDimension()][bm1.getColumnDimension()];
        for (int i = 0; i < vals.length; i++) {
            for (int j = 0; j < vals[i].length; j++) {
                vals[i][j] = bm1.getEntry(i, j);
            }
        }
        return vals;
    }

    private static void testingMergerOfHiCCUPSPostprocessing() {
        HiCGlobals.printVerboseComments = true;

        // example with hard-coded links
        String folder = "/Users/muhammadsaadshamim/Desktop/T0_48/0/exp1/";
        String baseLink = folder + "postprocessed_pixels_";
        String link1 = baseLink + "5000";
        String link2 = baseLink + "10000";
        String link3 = baseLink + "25000";
        String outputPath = folder + "new_merged_loops";

        Map<Integer, Feature2DList> map = new HashMap<>();
        map.put(5000, Feature2DParser.loadFeatures(link1, "hg19", true, null, false));
        map.put(10000, Feature2DParser.loadFeatures(link2, "hg19", true, null, false));
        map.put(25000, Feature2DParser.loadFeatures(link3, "hg19", true, null, false));

        Feature2DList newMerger = HiCCUPSUtils.mergeAllResolutions(map);
        newMerger.exportFeatureList(new File(outputPath), false, Feature2DList.ListFormat.FINAL);

        folder = "/Users/muhammad/Desktop/local_hiccups_gm12878/results3/";
        baseLink = folder + "enriched_pixels_";
        link1 = baseLink + "5000.bedpe";
        link2 = baseLink + "10000.bedpe";
        link3 = baseLink + "25000.bedpe";

        map = new HashMap<>();
        map.put(5000, Feature2DParser.loadFeatures(link1, "hg19", true, null, false));
        map.put(10000, Feature2DParser.loadFeatures(link2, "hg19", true, null, false));
        map.put(25000, Feature2DParser.loadFeatures(link3, "hg19", true, null, false));

        Dataset ds1 = HiCFileTools.extractDatasetForCLT(Arrays.asList("/Users/muhammad/Desktop/local_hic_files/gm12878_intra_nofrag_30.hic"), true);

        File outputDirectory = new File("/Users/muhammad/Desktop/local_hiccups_gm12878/results5");
        File outputMergedGivenFile = new File(outputDirectory, HiCCUPSUtils.getMergedRequestedLoopsFileName());

        HiCCUPSUtils.postProcess(map, ds1, ds1.getChromosomeHandler(),
                HiCCUPSConfiguration.getDefaultSetOfConfigsForUsers(),
                NormalizationHandler.KR, outputDirectory,
                false, outputMergedGivenFile, 1);
    }

    public static void testingHiCCUPSPostprocessing() {
        // example with hard-coded links
        String folder = "/Users/muhammadsaadshamim/Desktop/test_adam/";
        File outputDirectory = new File(folder);
        Dataset ds = HiCFileTools.extractDatasetForCLT(Collections.singletonList(folder + "inter_30.hic"), true);
        File outputMergedFile = new File(outputDirectory, "merged_loops");
        ChromosomeHandler chromosomeHandler = ds.getChromosomeHandler();
        NormalizationType norm = NormalizationHandler.KR;

        List<HiCCUPSConfiguration> filteredConfigurations = new ArrayList<>();
        filteredConfigurations.add(new HiCCUPSConfiguration(10000, 10, 2, 5, 20000));
        filteredConfigurations.add(new HiCCUPSConfiguration(5000, 10, 4, 7, 20000));

        String baseLink = folder + "enriched_pixels_";
        String link1 = baseLink + "5000";
        String link2 = baseLink + "10000";

        Map<Integer, Feature2DList> loopLists = new HashMap<>();
        loopLists.put(5000, Feature2DParser.loadFeatures(link1, chromosomeHandler, true, null, false));
        loopLists.put(10000, Feature2DParser.loadFeatures(link2, chromosomeHandler, true, null, false));

        HiCCUPSUtils.postProcess(loopLists, ds, chromosomeHandler,
                filteredConfigurations, norm, outputDirectory, false, outputMergedFile, 1);
    }

    /*
    public static void testCustomFastScaling() {
        ArrayList<String> files = new ArrayList<>();
        files.add("/Users/muhammad/Desktop/local_hic_files/gm12878_intra_nofrag_30.hic");
        Dataset ds = HiCFileTools.extractDatasetForCLT(files, false);
        Chromosome chr1 = ds.getChromosomeHandler().getAutosomalChromosomesArray()[0];
        MatrixZoomData zd = HiCFileTools.getMatrixZoomData(ds, chr1, chr1, 50000);

        int numEntries = (int)(chr1.getLength() / 50000) + 1; // assume for this test
        double[] targetVectorInitial = new double[numEntries];
        Arrays.fill(targetVectorInitial, 1);

        HiCGlobals.printVerboseComments = true;

        BigContactRecordList listOfLists = new ArrayList<>();
        listOfLists.addAll(zd.getContactRecordList());
        double[] result = ZeroScale.scale(listOfLists, targetVectorInitial, zd.getKey());

        System.out.println(Arrays.toString(result));
    }

    public void runUnitTests() {

        // TODO tests for all matrix/array tools
        // TODO tests for all CLTs
        // TODO tests for other standalone parts of code

        /*
        MatrixTools.print(MatrixTools.reshapeFlatMatrix(new float[]{1, 2, 3, 1, 2, 3}, 3, 2));



        RealMatrix rm = MatrixTools.cleanArray2DMatrix(2000);
        for(int i = 0; i < 20; i++){
            for(int j = 0; j < 20; j++){
                rm.setEntry(i,j,1);
            }
        }

        for(int i = 99; i < 120; i++){
            for(int j = 99; j < 120; j++){
                rm.setEntry(i,j,1);
            }
        }

        for(int i = 299; i < 400; i++){
            for(int j = 299; j < 400; j++){
                rm.setEntry(i,j,20000);
            }
        }

        for(int i = 349; i < 600; i++){
            for(int j = 349; j < 600; j++){
                rm.setEntry(i,j,5000);
            }
        }

        Random generator = new Random();
        for(int i = 1399; i < 1600; i++){
            for(int j = 1399; j < 1600; j++){
                rm.setEntry(i,j,5000*generator.nextDouble());
            }
        }

        BlockResults b = new BlockResults(rm, 1000, .4, new ArrowheadScoreList(), new ArrowheadScoreList());
        b.offsetResultsIndex(1);
        for(HighScore h : b.getResults())
            System.out.println(h);



            // super("arrowhead <input_HiC_file> <output_file> <resolution>");
        // http://adam.bcma.bcm.edu/hiseq/
*/

         /*
        String[] l1 = {"hiccups",
                "-r", "50000",
                "-c", "1",
                "-m", "100",
                "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
                "/Users/muhammadsaadshamim/Desktop/j3/out1_100",
                "/Users/muhammadsaadshamim/Desktop/j3/out2_100"};

        long time = System.currentTimeMillis();
        HiCTools.main(l1);
        time = (System.currentTimeMillis() - time) / 1000;
        long mins = time/60;
        long secs = time%60;
        System.out.println("Total time " + mins + " min "+ secs + " sec");

        /*


        String[] l1 = {"dump","observed", HiCFileTools.KR, "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
                "1", "1", HiCFileTools.BP, "5000", "/Users/muhammadsaadshamim/Desktop/perseus/chr1.bin"};

        String[] l4 = {"apa",
                "-r","50000",
                "-c","17,18",
                //"http://adam.bcma.bcm.edu/miseq/HIC1357.hic,http://adam.bcma.bcm.edu/miseq/HIC1357_30.hic,http://adam.bcma.bcm.edu/miseq/HIC1358.hic,http://adam.bcma.bcm.edu/miseq/HIC1358_30.hic,http://adam.bcma.bcm.edu/miseq/HIC1359.hic,http://adam.bcma.bcm.edu/miseq/HIC1359_30.hic,http://adam.bcma.bcm.edu/miseq/HIC1360.hic,http://adam.bcma.bcm.edu/miseq/HIC1360_30.hic,http://adam.bcma.bcm.edu/miseq/HIC1361.hic,http://adam.bcma.bcm.edu/miseq/HIC1361_30.hic,http://adam.bcma.bcm.edu/miseq/HIC1362.hic,http://adam.bcma.bcm.edu/miseq/HIC1362_30.hic,http://adam.bcma.bcm.edu/miseq/HIC1363.hic,http://adam.bcma.bcm.edu/miseq/HIC1363_30.hic",
                "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
                "/Users/muhammadsaadshamim/Desktop/Elena_APA/all_loops.txt",
                "/Users/muhammadsaadshamim/Desktop/Elena_APA/newt"};

        long time = System.currentTimeMillis();
        HiCTools.main(l3);
        time = System.currentTimeMillis() - time;
        System.out.println("Total time (ms): "+time);
        */



        /*
         * Example: this dumps data of each chromosome
         * for 5 single cell Hi-C experiments
         * at 5, 10, and 25 kb resolutions
         *
         * https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic
         */

        /*


        String[] l1 = {"apa","-r","5000",
                "/Users/muhammadsaadshamim/Desktop/Leviathan/nagano/cell-1/inter.hic",
                "/Users/muhammadsaadshamim/Desktop/Leviathan/nagano/mouse_list.txt",
                "/Users/muhammadsaadshamim/Desktop/apaTest1"};

        RealMatrix rm = new Array2DRowRealMatrix(new double[][]
            {   {0.0605,    0.6280,    0.1672,    0.3395,    0.2691},
                {0.3993,    0.2920,    0.1062,    0.9516,    0.4228},
                {0.5269,    0.4317,    0.3724,    0.9203,    0.5479},
                {0.4168,    0.0155,    0.1981,    0.0527,    0.9427},
                {0.6569,    0.9841,    0.4897,    0.7379,    0.4177}});

        rm = new Array2DRowRealMatrix(new double[][]
                {       {1,0,0,0,0,0,0,0},
                        {0,1,0,0,0,0,0,0},
                        {0,2,0,0,0,0,0,0},
                        {0,0,1,0,0,0,0,0},
                        {0,0,0,1,0,0,1,0},
                        {0,0,0,0,1,0,0,0},
                        {3,0,0,1,0,0,0,0},
                        {1,1,1,0,0,0,0,0}});



        String[] chrs = {"1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","X"};
        String[] kbs = {"5","10","25"};

        for(String kb : kbs) {
            for (int i = 1; i < 6; i++) {
                for (String chr : chrs) {
                    String[] line = {"dump",
                            "observed",
                            "NONE",
                            "/Users/muhammadsaadshamim/Desktop/nagano/cell-" + i + "/inter.hic",
                            "chr" + chr,
                            "chr" + chr,
                            HiCFileTools.BP,
                            kb+"000",
                            "/Users/muhammadsaadshamim/Desktop/nagano/apa_"+kb+"kb_" + i + "/counts/counts_" + chr + ".txt"};
                    HiCTools.main(line);
                }
            }
        }
        */

        /*
        int[] is = {5};
        for(int i : is) {
            for (String chr : chrs) {
                String[] line = {"dump", "observed", "NONE",
                        "/Users/muhammadsaadshamim/Desktop/nagano/cell-" + i + "/inter.hic",
                        "chr" + chr, "chr" + chr, HiCFileTools.BP, "5000",
                        "/Users/muhammadsaadshamim/Desktop/nagano/apa_5kb_" + i + "/counts/counts_" + chr + ".txt"};
                HiCTools.main(line);
            }
        }
        */


        /*
         * For verifying file identity using python:
         * {
         * import filecmp
         * print filecmp.cmp('output1.hic', 'output2.hic') # byte by byte comparison of output files
         * }
         */


        /*
        String[] l2 = {"addNorm",
                "/Users/muhammadsaadshamim/Desktop/testing/mouse.hic",
                "100000000"};
        String[] l3 = {"binToPairs",
                "/Users/muhammadsaadshamim/Desktop/testing/mouse.hic",
                "/Users/muhammadsaadshamim/Desktop/testing/mousesc3.hic"};
        String[] l4 = {"calcKR",
                "/Users/muhammadsaadshamim/Desktop/testing/mouse.hic"};
        String[] l5 = {"dump",
                "observed",
                "NONE",
                "/Users/muhammadsaadshamim/Desktop/testing/mouse.hic",
                "chr2",
                "chr2",
                HiCFileTools.BP,
                "1000000",
                "/Users/muhammadsaadshamim/Desktop/testing/mousesc.txt"};
        String[] l6 = {"pairsToBin",
                "/Users/muhammadsaadshamim/Desktop/testing/mouse.hic",
                "/Users/muhammadsaadshamim/Desktop/testing/mousesc2.hic",
                "mm10"};

        String[] l7 = { "pre",
                        "/Users/muhammadsaadshamim/Desktop/HIC156_smaller.txt",
                        "/Users/muhammadsaadshamim/Desktop/HIC156_smaller",
                        "hg19"
        };

        */


        /**
         *
         * testing dump
         *
         *  HiCGlobals.printVerboseComments = true;
         String[] ajkhsd = {"dump", "observed", HiCFileTools.KR, "/Users/muhammadsaadshamim/Desktop/LocalFiles/rice_mbr19_30.hic",
         "19","19", HiCFileTools.BP, "5000", "/Users/muhammadsaadshamim/Desktop/test_dump/dump_5k_19_full_kr"};

         //HiCTools.main(ajkhsd);

         ajkhsd = new String[]{"dump", "observed", HiCFileTools.KR, "/Users/muhammadsaadshamim/Desktop/LocalFiles/rice_mbr19_30.hic",
         "19","19", HiCFileTools.BP, "5000", "/Users/muhammadsaadshamim/Desktop/test_dump/dump_5k_19_full_kr"};

         HiCTools.main(ajkhsd);

         ajkhsd = new String[]{"dump", "observed", HiCFileTools.KR, "/Users/muhammadsaadshamim/Desktop/LocalFiles/rice_mbr19_30.hic",
         "19:0:59128983","19:10000000:20000000", HiCFileTools.BP, "5000", "/Users/muhammadsaadshamim/Desktop/test_dump/dump_5k_19_sub1_kr"};

         HiCTools.main(ajkhsd);

         ajkhsd = new String[]{"dump", "observed", HiCFileTools.KR, "/Users/muhammadsaadshamim/Desktop/LocalFiles/rice_mbr19_30.hic",
         "19","19:10000000:20000000", HiCFileTools.BP, "5000", "/Users/muhammadsaadshamim/Desktop/test_dump/dump_5k_19_sub1_kr_v2"};

         HiCTools.main(ajkhsd);

         ajkhsd = new String[]{"dump", "observed", HiCFileTools.KR, "/Users/muhammadsaadshamim/Desktop/LocalFiles/rice_mbr19_30.hic",
         "19:10000000:20000000","19:0:59128983", HiCFileTools.BP, "5000", "/Users/muhammadsaadshamim/Desktop/test_dump/dump_5k_19_sub2_kr"};

         HiCTools.main(ajkhsd);

         ajkhsd = new String[]{"dump", "observed", HiCFileTools.KR, "/Users/muhammadsaadshamim/Desktop/LocalFiles/rice_mbr19_30.hic",
         "19:10000000:20000000","19", HiCFileTools.BP, "5000", "/Users/muhammadsaadshamim/Desktop/test_dump/dump_5k_19_sub2_kr_v2"};

         HiCTools.main(ajkhsd);

         ajkhsd = new String[]{"dump", "observed", HiCFileTools.KR, "/Users/muhammadsaadshamim/Desktop/LocalFiles/rice_mbr19_30.hic",
         "19:10000000:20000000","19:10000000:20000000", HiCFileTools.BP, "5000", "/Users/muhammadsaadshamim/Desktop/test_dump/dump_5k_19_sub3_kr"};

         HiCTools.main(ajkhsd);
         *
     
         }
         */
}
