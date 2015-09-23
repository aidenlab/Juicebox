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

package juicebox.tools.clt.old;

import jargs.gnu.CmdLineParser;
import juicebox.tools.clt.JuiceboxCLT;
import juicebox.tools.utils.original.BigWigUtils;


public class BigWig extends JuiceboxCLT {

    private int version = -1;
    private int start = -1, end = -1, windowSize = -1;
    private String chr, path;

    public BigWig() {
        super("bigWig <bigWig path or URL> <window size in bp> [chr] [start base] [end base]");
    }

    @Override
    public void readArguments(String[] args, CmdLineParser parser) {
        //setUsage("juicebox bigWig <bigWig path or URL> <window size in bp> [chr] [start base] [end base]");
        if (!(args.length == 3 || args.length == 4 || args.length == 6)) {
            printUsage();
        }
        path = args[1];
        windowSize = Integer.parseInt(args[2]);

        if (args.length == 3) {
            version = 0;
        } else {
            chr = args[3];
            if (args.length == 4) {
                version = 1;
            } else {
                start = Integer.parseInt(args[4]) - 1;  // Convert to "zero" based coords
                end = Integer.parseInt(args[5]);
                version = 2;
            }
        }
    }

    @Override
    public void run() {
        try {
            if (version == 0)
                BigWigUtils.computeBins(path, windowSize);
            else if (version == 1)
                BigWigUtils.computeBins(path, chr, 0, Integer.MAX_VALUE, windowSize);
            else if (version == 2)
                BigWigUtils.computeBins(path, chr, start, end, windowSize);
            else {
                System.err.println("Invalid Option Setup");
                printUsage();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}