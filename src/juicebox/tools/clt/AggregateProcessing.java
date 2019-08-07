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

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Created for testing multiple CLTs at once
 * Basically scratch space
 */
class AggregateProcessing {


    public static void main(String[] argv) throws Exception {



        ChromosomeCalculation.sum(1000000, 10,
                "/Volumes/AidenLabWD7/Backup/AidenLab/LocalFiles/k562/combined_30.hic",
                "/Users/muhammad/Desktop/Sandra_v2/k562_chromcalc_1mb_r10");

        ChromosomeCalculation.sum(1000000, 10,
                "https://hicfiles.s3.amazonaws.com/hiseq/hap1/in-situ/combined.hic",
                "/Users/muhammad/Desktop/Sandra_v2/hap1_chromcalc_1mb_r10");


        ChromosomeCalculation.sum(1000000, 10,
                "/Volumes/AidenLabWD7/Backup/AidenLab/LocalFiles/ATDC5_Differentiated_Megamap_MAPQ30.hic",
                "/Users/muhammad/Desktop/Sandra_v2/atdc5_chromcalc_1mb_r10");

                ChromosomeCalculation.sum(1000000, 20,
                "/Volumes/AidenLabWD7/Backup/AidenLab/LocalFiles/ATDC5_Differentiated_Megamap_MAPQ30.hic",
                "/Users/muhammad/Desktop/Sandra_v2/atdc5_chromcalc_1mb_r20");
        */

        ChromosomeCalculation.sum(100000, 40,
                "/Volumes/AidenLabWD7/Backup/AidenLab/LocalFiles/ATDC5_Differentiated_Megamap_MAPQ30.hic",
                "/Users/muhammad/Desktop/Sandra_v2/atdc5_chromcalc_500kb_r40");


        String[] stripestestrun = new String[]{
                "grind", "-c", "4", "-r", "25000",
                "/Users/audreylu/Downloads/rh6wmo0b6a3l7d1cqsnzwfqxsnq6ie_dgwt.hic",
                "/Users/audreylu/Downloads/Stripes_DGWT_IB_3-4-6.bedpe",
                "30,300,1000000",
                "/Users/audreylu/Downloads/stripes_data"
        };

        HiCTools.main(stripestestrun);


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