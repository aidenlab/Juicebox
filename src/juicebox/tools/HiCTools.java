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

package juicebox.tools;

import jargs.gnu.CmdLineParser;
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
public class HiCTools {

    public static void main(String[] argv) throws IOException, CmdLineParser.UnknownOptionException, CmdLineParser.IllegalOptionValueException {
        Globals.setHeadless(true);

        String cmdName = argv[0].toLowerCase();
        CmdLineParser parser = new CommandLineParser();
        if (isJuicerCommand(cmdName)) {
            parser = new CommandLineParserForJuicer();
        }

        parser.parse(argv);
        String[] args = parser.getRemainingArgs();

        try {
            String cmd = args[0].toLowerCase();

            JuiceboxCLT instanceOfCLT = CLTFactory.getCLTCommand(cmd);

            try {
                instanceOfCLT.readArguments(args, parser);
            } catch (IOException e) {
                instanceOfCLT.printUsage(); // error reading arguments, print specific usage help
                System.exit(Integer.parseInt(e.getMessage()));
            }

            try {
                instanceOfCLT.run();
            } catch (Exception e) {
                System.out.println(e.getMessage());
                e.printStackTrace();
                System.exit(-7);
                // error running the code, these shouldn't occur i.e. error checking
                // should be added within each CLT for better error tracing
            }
        } catch (Exception e) {
            if (isJuicerCommand(cmdName)) {
                CLTFactory.juicerUsage();
            } else {
                CLTFactory.generalUsage();
            }
            System.exit(2);
        }
    }

    private static boolean isJuicerCommand(String cmd) {
        return cmd.equals("hiccups") || cmd.equals("apa") || cmd.equals("arrowhead") || cmd.equals("motif_finder");
    }
}
