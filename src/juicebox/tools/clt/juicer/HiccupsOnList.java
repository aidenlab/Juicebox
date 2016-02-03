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

package juicebox.tools.clt.juicer;

import juicebox.tools.clt.CommandLineParserForJuicer;
import juicebox.tools.clt.JuicerCLT;

/**
 * Created by Marie on 10/23/15.
 */
public class HiccupsOnList extends JuicerCLT {

    public HiccupsOnList() {
        super("enrichments [-m matrixSize] [-c chromosome(s)] [-r resolution(s)] [-f fdr] [-p peak width] [-i window] " +
                "[-t thresholds] [-d centroid distances] <hicFile(s)> <finalLoopsList>\n" +
                "\nhiccups [-m matrixSize] [-c chromosome(s)] [-r resolution(s)] [-f fdr] [-p peak width] [-i window] " +
                "<hicFile(s)> <fdrThresholds> <enrichedPixelsList>\n");
    }

///data/eleanor/suhas/peakcalling/scripts/peakcallingGPU18_short3.py    input_file_name    output_file_name1    output_file_name2    fdr    p    w
///data/eleanor/suhas/peakcalling/scripts/peakcallingGPU_listgivD.py    input_file_name    output_file_name1    output_file_name2    fdr     input_list    p    w

    @Override
    protected void readJuicerArguments(String[] args, CommandLineParserForJuicer juicerParser) {

    }

    @Override
    public void run() {

    }
}
