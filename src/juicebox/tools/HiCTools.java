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

package juicebox.tools;

import jargs.gnu.CmdLineParser;
import juicebox.HiCGlobals;
import juicebox.tools.clt.CLTFactory;
import juicebox.tools.clt.CommandLineParser;
import juicebox.tools.clt.CommandLineParserForJuicer;
import juicebox.tools.clt.JuiceboxCLT;
import org.broad.igv.Globals;

import java.io.IOException;


/**
 * Command line tool handling through factory model
 *
 * @author Muhammad Shamim
 * @date 1/20/2015
 */
class HiCTools {

    public static void main(String[] argv) throws IOException, CmdLineParser.UnknownOptionException, CmdLineParser.IllegalOptionValueException {
        Globals.setHeadless(true);

        if (argv.length == 0) {
            CLTFactory.generalUsage();
            System.exit(42);
        }
        String cmdName = argv[0].toLowerCase();

        CmdLineParser parser = new CommandLineParser();
        if (CommandLineParserForJuicer.isJuicerCommand(cmdName)) {
            parser = new CommandLineParserForJuicer();
            HiCGlobals.useCache = false; //TODO until memory leak cleared
        }

        parser.parse(argv);
        String[] args = parser.getRemainingArgs();

        JuiceboxCLT instanceOfCLT;
        String cmd = "";
        if (args.length == 0) {
            instanceOfCLT = null;
        } else {
            cmd = args[0];
            instanceOfCLT = CLTFactory.getCLTCommand(cmd);
        }
        if (instanceOfCLT != null) {
            if (args.length == 1) {
                instanceOfCLT.printUsageAndExit();
            }
            instanceOfCLT.readArguments(args, parser);
            instanceOfCLT.run();
        } else {
            throw new RuntimeException("Unknown command: " + cmd);
        }
    }
}
