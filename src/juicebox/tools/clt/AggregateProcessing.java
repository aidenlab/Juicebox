/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2019 Broad Institute, Aiden Lab
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

import juicebox.tools.HiCTools;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Created for testing multiple CLTs at once
 * Basically scratch space
 */
class AggregateProcessing {


    public static void main(String[] argv) throws Exception {

        String[] stripestestrun = new String[]{
                "grind", "-c", "6", "-r", "25000",
                "/Volumes/AidenLabWD7/Backup/AidenLab/LocalFiles/rh6wmo0b6a3l7d1cqsnzwfqxsnq6ie_dgwt.hic",
                "/Users/muhammad/Desktop/locationofzoghbideephicmapsformachinelearninganaly/Stripes_DGWT_IB_3-4-6.bedpe",
                "16,128,1000000",
                "/Users/muhammad/Desktop/stripes3"
        };

        HiCTools.main(stripestestrun);


//        String[] ll51231123 = new String[]{"grind",
//                "/Users/muhammad/Desktop/bin/local_hic_files/imr90_intra_nofrag_30.hic",
//                "https://hicfiles.s3.amazonaws.com/hiseq/imr90/in-situ/combined_peaks_with_motifs.txt",
//                "40,40,20000",
//                "/Users/muhammad/Desktop/grind/exploration/"
//        };
//
//        String[] testrun = {"afa", "-u", "-r", "25000", "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic", "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined_peaks_with_motifs.txt",
//        "C:/Users/Dat/Desktop/Juicebox/aparesulttest"};
//
//        HiCTools.main(testrun);
//
//        //UnitTests.testCustomFastScaling();

    }

    private static void writeMergedNoDupsFromTimeSeq(String seqPath, String newPath) {
        List<Integer[]> listPositions = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(seqPath))) {
            for (String line; (line = br.readLine()) != null; ) {
                String[] parts = line.split(",");
                listPositions.add(new Integer[]{Integer.parseInt(parts[0]), Integer.parseInt(parts[1])});
            }
        } catch (Exception ignored) {
            ignored.printStackTrace();
        }


        try {
            PrintWriter p0 = new PrintWriter(new FileWriter(newPath));
            for (int i = 0; i < listPositions.size(); i++) {
                Integer[] pos_xy_1 = listPositions.get(i);
                for (int j = i; j < listPositions.size(); j++) {
                    Integer[] pos_xy_2 = listPositions.get(j);
                    double value = 1. / Math.max(1, Math.sqrt((pos_xy_1[0] - pos_xy_2[0]) ^ 2 + (pos_xy_1[1] - pos_xy_2[1]) ^ 2));
                    float conv_val = (float) value;
                    if (!Float.isNaN(conv_val) && conv_val > 0) {
                        p0.println("0 art " + i + " 0 16 art " + j + " 1 " + conv_val);
                    }
                }
            }
            p0.close();
        } catch (IOException ignored) {
            ignored.printStackTrace();
        }
    }
}