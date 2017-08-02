/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2017 Broad Institute, Aiden Lab
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

import java.io.IOException;

/**
 * Created for testing multiple CLTs at once
 * Basically scratch space
 */
class AggregateProcessing {


    public static void main(String[] argv) throws IOException, CmdLineParser.UnknownOptionException, CmdLineParser.IllegalOptionValueException {

        String hicFilePaths="/Users/nathanielmusial/CS_Projects/SMART_Projects/Testing_Files/HiC/gm12878_intra_nofrag_30.hic";//.Hic
        String PeaksFile="/Users/nathanielmusial/CS_Projects/SMART_Projects/Testing_Files/Other/GM12878_loop_list.txt";//.txt
        String SaveFolderPath="/Users/nathanielmusial/CS_Projects/SMART_Projects/Output";

        /*
        APAvsDistance test= new APAvsDistance();
        test.run();

        */

        String[] ll51231123 = {"apa_vs_distance", hicFilePaths,PeaksFile,SaveFolderPath};

        HiCTools.main(ll51231123);


        /*
        String[] ll51231123 = {"motifs",
                "hg19",
                "/Users/muhammadsaadshamim/Desktop/test_motifs/gm12878_2",
                "/Users/muhammadsaadshamim/Desktop/test_motifs/loops_clean.txt"};


        HiCTools.main(ll51231123);
*/

    }
}