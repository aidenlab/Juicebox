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

/**
 * Created for testing multiple CLTs at once
 * Basically scratch space
 */
class AggregateProcessing {


    public static void main(String[] argv) throws Exception {


/*

        String[] strings = new String[]{"compare", "-m", "30000", "0", "hg19",
                "/Users/mshamim/Desktop/trident_degron/hct116/hct116_wt_no9_merged_loops.bedpe",
                "/Users/mshamim/Desktop/trident_degron/hct116/hct116_no9_DT_Loops_Merged.bedpe",
                "/Users/mshamim/Desktop/trident_degron/hct116/hiccups_vs_all_delta"
        };
        HiCTools.main(strings);

        /*

        String[] strings = new String[]{"network",
                "/Users/mshamim/Desktop/hicfiles/gm12878_rh14_30.hic",
                "/Users/mshamim/Desktop/in-situ-trident/predictions/gm12878_hiccups_no9_merged_loops.bedpe",
                "/Users/mshamim/Desktop/in-situ-trident/network/hiccups_"
        };
        HiCTools.main(strings);

        strings = new String[]{"network",
                "/Users/mshamim/Desktop/hicfiles/gm12878_rh14_30.hic",
                "/Users/mshamim/Desktop/in-situ-trident/predictions/GM12878_DT_Loops_Merged.bedpe",
                "/Users/mshamim/Desktop/in-situ-trident/network/delta_"
        };
        HiCTools.main(strings);

        /*
        System.out.println("REN");
        String[] strings = new String[]{"compare", "2", "hg19",
                "/Users/mshamim/Desktop/in-situ-trident/motifs/GM12878_DT_Loops_Merged_with_ren_motifs.bedpe",
                "/Users/mshamim/Desktop/in-situ-trident/motifs/GM12878_DT_Loops_Merged_with_ren_motifs.bedpe"
        };
        HiCTools.main(strings);

        strings = new String[]{"compare", "2", "hg19",
                "/Users/mshamim/Desktop/in-situ-trident/motifs/gm12878_hiccups_no9_merged_loops_with_ren_motifs.bedpe",
                "/Users/mshamim/Desktop/in-situ-trident/motifs/gm12878_hiccups_no9_merged_loops_with_ren_motifs.bedpe"
        };
        HiCTools.main(strings);

        System.out.println("ALL");
        strings = new String[]{"compare", "2", "hg19",
                "/Users/mshamim/Desktop/in-situ-trident/motifs/GM12878_DT_Loops_Merged_with_all_motifs.bedpe",
                "/Users/mshamim/Desktop/in-situ-trident/motifs/GM12878_DT_Loops_Merged_with_all_motifs.bedpe"
        };
        HiCTools.main(strings);

        strings = new String[]{"compare", "2", "hg19",
                "/Users/mshamim/Desktop/in-situ-trident/motifs/gm12878_hiccups_no9_merged_loops_with_all_motifs.bedpe",
                "/Users/mshamim/Desktop/in-situ-trident/motifs/gm12878_hiccups_no9_merged_loops_with_all_motifs.bedpe"
        };
        HiCTools.main(strings);




        /*
        String[] bedpes = new String[]{
                "GM12878_DT_Loops_Merged_lt_85.bedpe",
                "GM12878_DT_Loops_Merged_gt_85.bedpe",
                "GM12878_DT_Loops_Merged_btwn_85_90.bedpe",
                "GM12878_DT_Loops_Merged_gt_90.bedpe",
                "GM12878_DT_Loops_Merged_lt_90.bedpe",
                "gm12878_hiccups_no9_merged_loops.bedpe",
                "GM12878_DT_Loops_Merged.bedpe",
                "comparisons/hiccups_vs_all_delta_AAA.bedpe",
                "comparisons/hiccups_vs_all_delta_BBB.bedpe"
        };

        String[] outnames = new String[]{
                "apa_lt_85","apa_gt_85","apa_85_90",
                "apa_gt_90","apa_lt_90","apa_hiccups","apa_all_delta",
                "apa_specific_to_hiccups", "apa_specific_to_delta"
        };

        for(int k = 7; k < bedpes.length; k++) {
            String[] strings = new String[]{"apa",
                    "-r", "5000", "-k", "KR", "--threads", "6", //"-c", "1", //"--verbose",
                    "/Users/mshamim/Desktop/hicfiles/gm12878_rh14_30.hic",
                    "/Users/mshamim/Desktop/in-situ-trident/predictions/"+bedpes[k],
                    "/Users/mshamim/Desktop/in-situ-trident/predictions/"+outnames[k]};
            HiCTools.main(strings);
        }

        String[] strings = new String[]{"compare", "-m", "25000", "0", "hg19",
                "/Users/mshamim/Desktop/in-situ-trident/predictions/"+bedpes[5],
                "/Users/mshamim/Desktop/in-situ-trident/predictions/"+bedpes[6],
                "/Users/mshamim/Desktop/in-situ-trident/predictions/hiccups_vs_all_delta"
        };
        //HiCTools.main(strings);

        strings = new String[]{"compare", "-m", "25000", "0", "hg19",
                "/Users/mshamim/Desktop/in-situ-trident/predictions/"+bedpes[5],
                "/Users/mshamim/Desktop/in-situ-trident/predictions/"+bedpes[1],
                "/Users/mshamim/Desktop/in-situ-trident/predictions/hiccups_vs_delta_gt85"
        };
        //HiCTools.main(strings);

        strings = new String[]{"compare", "-m", "25000", "0", "hg19",
                "/Users/mshamim/Desktop/in-situ-trident/predictions/"+bedpes[5],
                "/Users/mshamim/Desktop/in-situ-trident/predictions/"+bedpes[3],
                "/Users/mshamim/Desktop/in-situ-trident/predictions/hiccups_vs_delta_gt90"
        };
        //HiCTools.main(strings);

        strings = new String[]{"finetune", "-k", "KR",
                "/Users/mshamim/Desktop/hicfiles/gm12878_rh14_30.hic",
                "/Users/mshamim/Desktop/in-situ-trident/predictions/"+bedpes[6],
                "/Users/mshamim/Desktop/in-situ-trident/finetune"
        };
        //HiCTools.main(strings);

        /*
        strings = new String[]{"grind",
                "-k", "KR", "-r", "25000",// "5000,10000,25000",
                "--stride", "1500", "-c", "4,5",
                "--dense-labels", "--distort",
                "/Users/muhammad/Desktop/local_hic_files/HIC053_30.hic",
                "null", "2000,12,100",
                "/Users/muhammad/Desktop/deeplearning/testing/distortion_bank_4_5_debug_version"};

        for (int k = 1; k < 2; k++) {
            UNIXTools.makeDir("/Users/muhammad/Desktop/test_pre/multi_test_finalscale" + k);
            strings = new String[]{"pre", "--threads", "" + k, "--mndindex",
                    "/Users/muhammad/Desktop/test_pre/indices.txt", "--skip-intra-frag", //"-n",
                    //"/Users/muhammad/JuiceboxAgain/data/test.txt.gz",
                    "/Users/muhammad/Desktop/test_pre/test.txt",
                    "/Users/muhammad/Desktop/test_pre/multi_test_finalscale" + k + "/test" + k + ".hic",
                    "hg19"};

            HiCTools.main(strings);
            System.gc();
        }

        // load the model

        /*
        String simpleMlp = "/Users/muhammad/Desktop/deeplearning/models/Clean64DistortionDiffHalfLocalizerV0BinCross.h5";
        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(simpleMlp);


        // make a random sample
        int inputs = 10;
        INDArray features = Nd4j.zeros(inputs);
        for (int i=0; i<inputs; i++) {
            features.putScalar(new int[]{i}, Math.random() < 0.5 ? 0 : 1);
        }
// get the prediction
        //double prediction = model.output(features).getDouble(0);

         */



    }
}
