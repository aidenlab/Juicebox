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
import juicebox.tools.utils.APAPlotter;

import java.awt.*;
import java.io.File;
import java.io.IOException;

/**
 * Created for testing multiple CLTs at once
 */
public class AggregateProcessing {




    public static void main(String[] argv) throws IOException, CmdLineParser.UnknownOptionException, CmdLineParser.IllegalOptionValueException {

        double[][] dataAPA =
                        {{0, 0, 0, 0, 1, 1, 0, 1, 2, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 5},
                        {1, 0, 0, 0, 1, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2},
                        {0, 0, 1, 0, 0, 1, 0, 1, 0, 3, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 5},
                        {1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1},
                        {0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 5},
                        {0, 0, 0, 0, 2, 0, 0, 0, 1, 1, 0, 2, 1, 3, 0, 0, 0, 2, 1, 0, 1, 2},
                        {0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1},
                        {0, 2, 0, 0, 1, 0, 1, 2, 1, 0, 1, 0, 1, 2, 1, 2, 1, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 1, 0, 1, 1, 1, 3, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 5},
                        {1, 0, 0, 0, 20, 1, 0, 2, 0, 0, 4, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 5},
                        {2, 2, 0, 1, 0, 1, 2, 0, 2, 1, 4, 1, 2, 0, 1, 1, 1, 1, 1, 0, 0, 1},
                        {0, 1, 0, 0, 0, 2, 3, 0, 0, 3, 0, 0, 3, 2, 0, 0, 0, 1, 0, 1, 2, 2},
                        {1, 2, 0, 0, 1, 1, 1, 4, 1, 1, 0, 0, 0, 3, 0, 2, 0, 0, 1, 0, 2, 1},
                        {0, 1, 0, 0, 0, 0, 0, 2, 0, 3, 2, 0, 0, 0, 0, 06, 2, 0, 0, 0, 0, 1},
                        {1, 2, 0, 0, 1, 0, 2, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 2, 0, 1, 1},
                        {1, 1, 1, 1, 0, 1, 1, 1, 15, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1},
                        {0, 1, 0, 1, 0, 2, 1, 1, 2, 73, 1, 1, 0, 0, 0, 1, 0, 17, 0, 0, 0, 1},
                        {0, 0, 0, 0, 1, 2, 2, 1, 0, 1, 1, 1, 0, 0, 0, 5, 2, 1, 0, 0, 0, 1},
                        {0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                        {1, 1, 1, 0, 2, 2, 1, 0, 1, 1, 3, 0, 1, 0, 0, 0, 0, 1, 0, 0, 2, 1},

                                {1, 0, 0, 0, 1, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2},
                                {0, 0, 1, 0, 0, 1, 0, 1, 0, 3, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 5},
                                {1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1},
                                {0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 5},
                                {0, 0, 0, 0, 2, 0, 0, 0, 1, 1, 0, 2, 1, 3, 0, 0, 0, 2, 1, 0, 1, 2},
                                {0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1},
                                {0, 2, 0, 0, 1, 0, 1, 2, 1, 0, 1, 0, 1, 2, 1, 2, 1, 0, 0, 0, 0, 0},
                                {0, 0, 0, 0, 1, 0, 1, 1, 1, 3, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 5},
                                {1, 0, 0, 0, 20, 1, 0, 2, 0, 0, 4, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 5},
                                {2, 2, 0, 1, 0, 1, 2, 0, 2, 1, 4, 1, 2, 0, 1, 1, 1, 1, 1, 0, 0, 1},
                                {0, 1, 0, 0, 0, 2, 3, 0, 0, 3, 0, 0, 3, 2, 0, 0, 0, 1, 0, 1, 2, 2},
                                {1, 2, 0, 0, 1, 1, 1, 4, 1, 1, 0, 0, 0, 3, 0, 2, 0, 0, 1, 0, 2, 1},
                                {0, 1, 0, 0, 0, 0, 0, 2, 0, 3, 2, 0, 0, 0, 0, 06, 2, 0, 0, 0, 0, 1},
                                {1, 2, 0, 0, 1, 0, 2, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 2, 0, 1, 1},
                                {1, 1, 1, 1, 0, 1, 1, 1, 15, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1},
                                {0, 1, 0, 1, 0, 2, 1, 1, 2, 73, 1, 1, 0, 0, 0, 1, 0, 17, 0, 0, 0, 1},
                                {0, 0, 0, 0, 1, 2, 2, 1, 0, 1, 1, 1, 0, 0, 0, 5, 2, 1, 0, 0, 0, 1},
                                {0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                                {1, 1, 1, 0, 2, 2, 1, 0, 1, 1, 3, 0, 1, 0, 0, 0, 0, 1, 0, 0, 2, 1},
                                {1, 0, 0, 0, 1, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2},
                                {0, 0, 1, 0, 0, 1, 0, 1, 0, 3, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 5},
                                {1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1},
                                {0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 5},
                                {0, 0, 0, 0, 2, 0, 0, 0, 1, 1, 0, 2, 1, 3, 0, 0, 0, 2, 1, 0, 1, 2},
                                {0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1},
                                {0, 2, 0, 0, 1, 0, 1, 2, 1, 0, 1, 0, 1, 2, 1, 2, 1, 0, 0, 0, 0, 0},
                                {0, 0, 0, 0, 1, 0, 1, 1, 1, 3, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 5},
                                {1, 0, 0, 0, 20, 1, 0, 2, 0, 0, 4, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 5},
                                {2, 2, 0, 1, 0, 1, 2, 0, 2, 1, 4, 1, 2, 0, 1, 1, 1, 1, 1, 0, 0, 1},
                                {0, 1, 0, 0, 0, 2, 3, 0, 0, 3, 0, 0, 3, 2, 0, 0, 0, 1, 0, 1, 2, 2},
                                {1, 2, 0, 0, 1, 1, 1, 4, 1, 1, 0, 0, 0, 3, 0, 2, 0, 0, 1, 0, 2, 1},
                                {0, 1, 0, 0, 0, 0, 0, 2, 0, 3, 2, 0, 0, 0, 0, 06, 2, 0, 0, 0, 0, 1},
                                {1, 2, 0, 0, 1, 0, 2, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 2, 0, 1, 1},
                                {1, 1, 1, 1, 0, 1, 1, 1, 15, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1},
                                {0, 1, 0, 1, 0, 2, 1, 1, 2, 73, 1, 1, 0, 0, 0, 1, 0, 17, 0, 0, 0, 1},
                                {0, 0, 0, 0, 1, 2, 2, 1, 0, 1, 1, 1, 0, 0, 0, 5, 2, 1, 0, 0, 0, 1},
                                {0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                                {1, 1, 1, 0, 2, 2, 1, 0, 1, 1, 3, 0, 1, 0, 0, 0, 0, 1, 0, 0, 2, 1},
                                {1, 0, 0, 0, 1, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2},
                                {0, 0, 1, 0, 0, 1, 0, 1, 0, 3, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 5},
                                {1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1},
                                {0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 5},
                                {0, 0, 0, 0, 2, 0, 0, 0, 1, 1, 0, 2, 1, 3, 0, 0, 0, 2, 1, 0, 1, 2},
                                {0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1},
                                {0, 2, 0, 0, 1, 0, 1, 2, 1, 0, 1, 0, 1, 2, 1, 2, 1, 0, 0, 0, 0, 0},
                                {0, 0, 0, 0, 1, 0, 1, 1, 1, 3, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 5},
                                {1, 0, 0, 0, 20, 1, 0, 2, 0, 0, 4, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 5},
                                {2, 2, 0, 1, 0, 1, 2, 0, 2, 1, 4, 1, 2, 0, 1, 1, 1, 1, 1, 0, 0, 1},
                                {0, 1, 0, 0, 0, 2, 3, 0, 0, 3, 0, 0, 3, 2, 0, 0, 0, 1, 0, 1, 2, 2},
                                {1, 2, 0, 0, 1, 1, 1, 4, 1, 1, 0, 0, 0, 3, 0, 2, 0, 0, 1, 0, 2, 1},
                                {0, 1, 0, 0, 0, 0, 0, 2, 0, 3, 2, 0, 0, 0, 0, 06, 2, 0, 0, 0, 0, 1},
                                {1, 2, 0, 0, 1, 0, 2, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 2, 0, 1, 1},
                                {1, 1, 1, 1, 0, 1, 1, 1, 15, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1},
                                {0, 1, 0, 1, 0, 2, 1, 1, 2, 73, 1, 1, 0, 0, 0, 1, 0, 17, 0, 0, 0, 1},
                                {0, 0, 0, 0, 1, 2, 2, 1, 0, 1, 1, 1, 0, 0, 0, 5, 2, 1, 0, 0, 0, 1},
                                {0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                                {1, 1, 1, 0, 2, 2, 1, 0, 1, 1, 3, 0, 1, 0, 0, 0, 0, 1, 0, 0, 2, 1},
                                {1, 0, 0, 0, 1, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2},
                                {0, 0, 1, 0, 0, 1, 0, 1, 0, 3, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 5},
                                {1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1},
                                {0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 5},
                                {0, 0, 0, 0, 2, 0, 0, 0, 1, 1, 0, 2, 1, 3, 0, 0, 0, 2, 1, 0, 1, 2},
                                {0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1},
                                {0, 2, 0, 0, 1, 0, 1, 2, 1, 0, 1, 0, 1, 2, 1, 2, 1, 0, 0, 0, 0, 0},
                                {0, 0, 0, 0, 1, 0, 1, 1, 1, 3, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 5},
                                {1, 0, 0, 0, 20, 1, 0, 2, 0, 0, 4, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 5},
                                {2, 2, 0, 1, 0, 1, 2, 0, 2, 1, 4, 1, 2, 0, 1, 1, 1, 1, 1, 0, 0, 1},
                                {0, 1, 0, 0, 0, 2, 3, 0, 0, 3, 0, 0, 3, 2, 0, 0, 0, 1, 0, 1, 2, 2},
                                {1, 2, 0, 0, 1, 1, 1, 4, 1, 1, 0, 0, 0, 3, 0, 2, 0, 0, 1, 0, 2, 1},
                                {0, 1, 0, 0, 0, 0, 0, 2, 0, 3, 2, 0, 0, 0, 0, 06, 2, 0, 0, 0, 0, 1},
                                {1, 2, 0, 0, 1, 0, 2, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 2, 0, 1, 1},
                                {1, 1, 1, 1, 0, 1, 1, 1, 15, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1},
                                {0, 1, 0, 1, 0, 2, 1, 1, 2, 73, 1, 1, 0, 0, 0, 1, 0, 17, 0, 0, 0, 1},
                                {0, 0, 0, 0, 1, 2, 2, 1, 0, 1, 1, 1, 0, 0, 0, 5, 2, 1, 0, 0, 0, 1},
                                {0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                                {1, 1, 1, 0, 2, 2, 1, 0, 1, 1, 3, 0, 1, 0, 0, 0, 0, 1, 0, 0, 2, 1},
                        {1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 3, 1, 1, 1, 1, 3, 1, 1, 0, 1, 1}};
        int[] axesRange = {-dataAPA.length / 2, 1, -dataAPA.length / 2, 1};
        double[] regionValues = {0,0,0,0};
        Dimension regionDimensions = new Dimension(5,7);
        String title = "N=2330(2330)/3331, P2LL: 5.538";
        File ofile = new File("/Users/muhammadsaadshamim/Desktop/im0.png");

        APAPlotter.plot(dataAPA,  axesRange, regionValues, regionDimensions, title, ofile, "png");
    }
}

        /*
         * Example: this dumps data of each chromosome
         * for 5 single cell Hi-C experiments
         * at 5, 10, and 25 kb resolutions
         */
        /*
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

        // ? HiCTools.main(l1);
        //HiCTools.main(l2);  // worked
        //HiCTools.main(l3);
        // ? HiCTools.main(l4);
        //HiCTools.main(l5);
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