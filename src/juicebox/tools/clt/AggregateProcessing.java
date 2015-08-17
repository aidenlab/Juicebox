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

import java.io.IOException;

/**
 * Created for testing multiple CLTs at once
 * Basically scratch space
 */
class AggregateProcessing {


    public static void main(String[] argv) throws IOException, CmdLineParser.UnknownOptionException, CmdLineParser.IllegalOptionValueException {
        String[] l4 = {"arrowhead", "-c", "22", "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
                "/Users/muhammadsaadshamim/Desktop/j3/out8/", "25000"};

        String[] l5 = {"dump", "observed", "KR", "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
                "22", "22", "BP", "25000", "/Users/muhammadsaadshamim/Desktop/BioScripts/22_blocks"};

        String[] l6 = {"apa", "-r", "5000", "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
                "/Users/muhammadsaadshamim/Downloads/APA_Ivan/gmloops.txt.gz",
                "/Users/muhammadsaadshamim/Downloads/APA_Ivan/temp3"};

        String[] l7 = {"hiccups", "-c", "22", "-m", "90",
                "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
                "/Users/muhammadsaadshamim/Desktop/j3/out8/loops"};

        HiCTools.main(l7);
    }
}