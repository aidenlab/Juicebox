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

package juicebox.tools.utils;

import jargs.gnu.CmdLineParser;
import juicebox.HiCGlobals;
import juicebox.tools.clt.CLTFactory;
import juicebox.tools.clt.CommandLineParser;
import juicebox.tools.clt.CommandLineParserForJuicer;
import juicebox.tools.clt.old.Dump;
import org.broad.igv.Globals;

import java.io.IOException;

/**
 * Created by nchernia on 8/4/16.
 */

public class Benchmark {

    public static void main(String[] argv) throws IOException,  CmdLineParser.UnknownOptionException, CmdLineParser.IllegalOptionValueException {

        Globals.setHeadless(true);

        if (argv.length == 0) {
            CLTFactory.generalUsage();
            System.exit(0);
        }
        String cmdName = argv[0].toLowerCase();

        CmdLineParser parser = new CommandLineParser();
        if (CommandLineParserForJuicer.isJuicerCommand(cmdName)) {
            parser = new CommandLineParserForJuicer();
            HiCGlobals.useCache = false;
        }

        parser.parse(argv);
        String[] args = parser.getRemainingArgs();

        Dump dump = new Dump();
        dump.readArguments(args, parser);

        // Choose chromosome and resolution to query based on initial arguments
        String chr1 = dump.getChr1();
        String chr2 = dump.getChr2();
        int binSize = dump.getBinSize();

        // Query 100 times at 256x256
        int QUERY_SIZE=256;
        long sum=0;
        for (int i=0; i<100; i++) {
            int start = 1000000 + (1000*i);
            int end = binSize*QUERY_SIZE + start;

            dump.setQuery(chr1+":"+start+":"+end, chr2+":"+start+":"+end);
            long currentTime = System.currentTimeMillis();
            dump.run();
            long totalTime = System.currentTimeMillis() - currentTime;
            sum+=totalTime;
        }
        System.err.println("Average time to query " + QUERY_SIZE + "x" + QUERY_SIZE +": " + sum/100 + " milliseconds");

        QUERY_SIZE=2048;
        sum=0;
        for (int i=0; i<100; i++) {
            int start = 1000000 + (1000*i);
            int end = binSize*QUERY_SIZE + start;

            dump.setQuery(chr1+":"+start+":"+end, chr2+":"+start+":"+end);
            long currentTime = System.currentTimeMillis();
            dump.run();
            long totalTime = System.currentTimeMillis() - currentTime;
            sum+=totalTime;
        }
        System.err.println("Average time to query " + QUERY_SIZE + "x" + QUERY_SIZE +": " + sum/100 + " milliseconds");

    }
}
