/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2016 Broad Institute, Aiden Lab
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
import juicebox.HiCGlobals;
import juicebox.tools.utils.juicer.hiccups.HiCCUPSUtils;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.Feature2DParser;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Created for testing multiple CLTs at once
 * Basically scratch space
 */
class AggregateProcessing {


    public static void main(String[] argv) throws IOException, CmdLineParser.UnknownOptionException, CmdLineParser.IllegalOptionValueException {

        /*
        /Users/muhammadsaadshamim/Desktop/test_motifs/original
muhammads-mbp:original muhammadsaadshamim$ ls
geo.txt		java_motifs.txt
         */

        HiCGlobals.printVerboseComments = true;

        Map<Integer, Feature2DList> map = new HashMap<Integer, Feature2DList>();
        map.put(5000, Feature2DParser.loadFeatures("/Users/muhammadsaadshamim/Desktop/T0_48/48/exp1/postprocessed_pixels_5000"
                , "hg19", true, null, false));
        map.put(10000, Feature2DParser.loadFeatures("/Users/muhammadsaadshamim/Desktop/T0_48/48/exp1/postprocessed_pixels_10000"
                , "hg19", true, null, false));
        map.put(25000, Feature2DParser.loadFeatures("/Users/muhammadsaadshamim/Desktop/T0_48/48/exp1/postprocessed_pixels_25000"
                , "hg19", true, null, false));

        Feature2DList newMerger = HiCCUPSUtils.mergeAllResolutions(map);

        newMerger.exportFeatureList("/Users/muhammadsaadshamim/Desktop/T0_48/48/exp1/new_merged_loops", false, Feature2DList.ListFormat.FINAL);


        /*
        GenomeWideList<MotifAnchor> l1 = MotifAnchorParser.loadMotifsFromGenomeID("hg19", null);

        GenomeWideList<MotifAnchor> l2 = MotifAnchorParser.loadMotifsFromLocalFile(
                "/Users/muhammadsaadshamim/Desktop/GenomeWideMotifs/motif_list_hg19_M1/M1_fimo_full_out_1M_t1e3/hg19.motifs.txt", "hg19", null);

        GenomeWideList<MotifAnchor> l3 = MotifAnchorParser.loadMotifsFromURL(
                "http://hicfiles.s3.amazonaws.com/internal/motifs/hg19.motifs.txt", "hg19", null);

        System.out.println(l1.size());
        System.out.println(l2.size());
        System.out.println(l3.size());
        */


/*
        String[] ajkhsd = {"loop_domains",
                "hg19",
                "/Users/muhammadsaadshamim/Desktop/test/test_ld/loops.txt",
                "/Users/muhammadsaadshamim/Desktop/test/test_ld/domains.txt",
                "/Users/muhammadsaadshamim/Desktop/test/test_ld/ld.txt"};

        //HiCGlobals.printVerboseComments = true;
        //HiCTools.main(ajkhsd);


                /*
        String[] ll51231123 = {"motifs",
                "hg19",
                "/Users/muhammadsaadshamim/Desktop/test/test_motifs/gm12878",
                "/Users/muhammadsaadshamim/Desktop/test/test_motifs/loops_clean.txt",
                "/Users/muhammadsaadshamim/Dropbox (Lab at Large)/GenomeWideMotifs/motif_list/REN_fimo_full_out_1M/fimo.txt"};

        //HiCGlobals.printVerboseComments = true;
        HiCTools.main(ll51231123);

        String[] ll512123431123 = new String[]{"compare",
                "0", "-m", "25000",
                "hg19",
                "/Users/muhammadsaadshamim/Desktop/test/test_aws_hiccups/geo.txt",
                "/Users/muhammadsaadshamim/Desktop/test/test_aws_hiccups/aws_loops_30.txt"};

        ll512123431123 = new String[]{"compare",
                "-m", "5000", "1",
                "hg19",
                "/Users/muhammadsaadshamim/Desktop/test/test_motifs/original/new_suhas_list.txt",
                "/Users/muhammadsaadshamim/Desktop/test/test_motifs/loops_clean_with_motifs.txt"};

        HiCTools.main(ll512123431123);

        ll512123431123 = new String[]{"compare",
                "-m", "5000", "2",
                "hg19",
                "/Users/muhammadsaadshamim/Desktop/test/test_motifs/original/new_suhas_list.txt",
                "/Users/muhammadsaadshamim/Desktop/test/test_motifs/loops_clean_with_motifs.txt"};

        HiCTools.main(ll512123431123);

        ll512123431123 = new String[]{"compare",
                "-m", "5000", "2",
                "hg19",
                "/Users/muhammadsaadshamim/Desktop/test/test_motifs/loops_clean_with_motifs.txt",
                "/Users/muhammadsaadshamim/Desktop/test/test_motifs/original/new_suhas_list.txt"};

        HiCTools.main(ll512123431123);
        */
    }
}