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

import jargs.gnu.CmdLineParser;
import juicebox.tools.HiCTools;
import juicebox.tools.utils.Common.MatrixTools;
import juicebox.tools.utils.Juicer.APA.APAPlotter;
import juicebox.tools.utils.Juicer.Arrowhead.ConnectedComponents.BinaryConnectedComponents;
import juicebox.tools.utils.Juicer.Arrowhead.DynamicProgrammingUtils;
import juicebox.tools.utils.Juicer.HiCCUPS.GPUTesting;
import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.RealMatrix;

import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.List;

/**
 * Created for testing multiple CLTs at once
 */
public class AggregateProcessing {




    public static void main(String[] argv) throws IOException, CmdLineParser.UnknownOptionException, CmdLineParser.IllegalOptionValueException {


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

            System.out.println(BinaryConnectedComponents.detection(rm.getData(), 0));


        //HiCTools.main(l4);
        //HiCTools.main(l5);
        //HiCTools.main(l6);
        //HiCTools.main(l3);
        //GPUTesting.test();





    }
}

        /*
         * Example: this dumps data of each chromosome
         * for 5 single cell Hi-C experiments
         * at 5, 10, and 25 kb resolutions
         */
        /*


        String[] l1 = {"apa","-r","5000",
                "/Users/muhammadsaadshamim/Desktop/Leviathan/nagano/cell-1/inter.hic",
                "/Users/muhammadsaadshamim/Desktop/Leviathan/nagano/mouse_list.txt",
                "/Users/muhammadsaadshamim/Desktop/apaTest1"};

        String[] l2 = {"hiccups",
                "/Users/muhammadsaadshamim/Desktop/Leviathan/nagano/cell-1/inter.hic",
                "/Users/muhammadsaadshamim/Desktop/156_fdr",
                "/Users/muhammadsaadshamim/Desktop/156_peaks"};

        String[] l3 = {"hiccups",
                "-r","50000",
                "-c","21",
                "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
                "/Users/muhammadsaadshamim/Desktop/GM12878/21_fdr",
                "/Users/muhammadsaadshamim/Desktop/GM12878/21_peaks"};


        //dump <observed/oe/norm/expected> <NONE/VC/VC_SQRT/KR> <hicFile(s)> <chr1> <chr2> <BP/FRAG> <binsize> [outfile]
        String[] l4 = {"dump","observed", "KR", "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
                "21", "21", "BP", "5000", "/Users/muhammadsaadshamim/Desktop/GM12878/21py/chr21.bin"};

        String[] l5 = {"dump","expected", "KR", "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
                "21", "21", "BP", "5000", "/Users/muhammadsaadshamim/Desktop/GM12878/21py/chr21exp.bin"};

        String[] l6 = {"dump","norm", "KR", "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
                "21", "21", "BP", "5000", "/Users/muhammadsaadshamim/Desktop/GM12878/21py/chr21norm.bin"};


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
                            "BP",
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
                        "chr" + chr, "chr" + chr, "BP", "5000",
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
        String[] l1 = {"addGWNorm",
                "/Users/muhammadsaadshamim/Desktop/testing/mouse.hic",
                "100000000"};
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
                "BP",
                "1000000",
                "/Users/muhammadsaadshamim/Desktop/testing/mousesc.txt"};
        String[] l6 = {"pairsToBin",
                "/Users/muhammadsaadshamim/Desktop/testing/mouse.hic",
                "/Users/muhammadsaadshamim/Desktop/testing/mousesc2.hic",
                "mm10"};

        String[][] cmds = {l1, l2, l3, l4, l5, l6};

        //HiCTools.main(l1);
        //HiCTools.main(l6);

        String[] l7 = { "pre",
                        "/Users/muhammadsaadshamim/Desktop/HIC156_smaller.txt",
                        "/Users/muhammadsaadshamim/Desktop/HIC156_smaller",
                        "hg19"
        };
        //HiCTools.main(l7);

        String[] l8 = {"dump","observed","NONE","/Users/muhammadsaadshamim/Desktop/temp_Juice/Juicebox/testing/HIC156_smaller_2.hic","1","1","BP","10000","/Users/muhammadsaadshamim/Desktop/temp_Juice/Juicebox/testing/temp6"};
        HiCTools.main(l8);
        */