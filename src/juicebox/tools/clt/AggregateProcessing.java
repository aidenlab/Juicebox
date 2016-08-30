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
import juicebox.tools.HiCTools;

import java.io.IOException;

/**
 * Created for testing multiple CLTs at once
 * Basically scratch space
 */
class AggregateProcessing {


    public static void main(String[] argv) throws IOException, CmdLineParser.UnknownOptionException, CmdLineParser.IllegalOptionValueException {

        // genes <genomeID> <bed_file> <looplist> [output]
        String[] ajkhsd = {"dump", "observed", "VC",
                "https://hicfiles.s3.amazonaws.com/hiseq/ch12-lx-b-lymphoblasts/in-situ/combined.hic",
                "1:0:5000", "1:0:10000", "BP", "5000", "/Users/muhammadsaadshamim/Desktop/test/ignore/chr1_VCnorm_5kb.txt"
        };

        ajkhsd = new String[]{"genes", "hg19", "/Users/muhammadsaadshamim/Desktop/intersected_cbx28_y1.bed",
                "https://hicfiles.s3.amazonaws.com/hiseq/k562/in-situ/combined_peaks_with_motifs.txt",
                "/Users/muhammadsaadshamim/Desktop/k562_cbx28_yy1_genes"};

        ajkhsd = new String[]{"hiccupsdiff", "/Users/muhammadsaadshamim/Desktop/LocalFiles/k562/k562_combined_30.hic",
                "/Users/muhammadsaadshamim/Desktop/LocalFiles/k562/k562_combined_30.hic",
                "https://hicfiles.s3.amazonaws.com/hiseq/k562/in-situ/combined_peaks_with_motifs.txt",
                "https://hicfiles.s3.amazonaws.com/hiseq/k562/in-situ/combined_peaks_with_motifs.txt",
                "/Users/muhammadsaadshamim/Desktop/LocalFiles/general/rice_mbr19_30hic_fakediff"};
/*
        ajkhsd = new String[]{"dump", "pearson", "KR",
                "https://hicfiles.s3.amazonaws.com/hiseq/k562/in-situ/combined.hic",
                "12", "12", "BP", "500000", "/Users/muhammadsaadshamim/Desktop/test/ignore/k562_chr12_pearson"
        };

        ajkhsd = new String[]{"pre", "/Users/muhammadsaadshamim/Documents/Github/Hydra/JuiceboxDev/data/data2"
                , "/Users/muhammadsaadshamim/Documents/Github/Hydra/JuiceboxDev/data/data2.hic", "hg19"
        };
*/
        HiCGlobals.printVerboseComments = true;
        HiCTools.main(ajkhsd);
    }
}