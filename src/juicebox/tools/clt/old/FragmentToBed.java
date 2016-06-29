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

package juicebox.tools.clt.old;

import com.google.common.base.CharMatcher;
import com.google.common.base.Splitter;
import jargs.gnu.CmdLineParser;
import juicebox.HiCGlobals;
import juicebox.tools.clt.JuiceboxCLT;

import java.io.*;
import java.util.List;
import java.util.regex.Pattern;


public class FragmentToBed extends JuiceboxCLT {

    private static final Splitter MY_SPLITTER = Splitter.on(CharMatcher.BREAKING_WHITESPACE).trimResults().omitEmptyStrings();
    private String filename;

    public FragmentToBed() {
        super("fragmentToBed <fragmentFile>");
    }

    /**
     * Convert a fragment site file to a "bed" file
     *
     * @param filename fragment site file
     * @throws java.io.IOException
     */
    private static void fragToBed(String filename) throws IOException {
        BufferedReader reader = null;
        PrintWriter writer = null;
        try {
            File inputFile = new File(filename);
            reader = new BufferedReader(new FileReader(inputFile), HiCGlobals.bufferSize);

            writer = new PrintWriter(new BufferedWriter(new FileWriter(filename + ".bed")));

            Pattern pattern = Pattern.compile("\\s");
            String nextLine;
            while ((nextLine = reader.readLine()) != null) {
                //String[] tokens = pattern.split(nextLine);
                List<String> tokens = MY_SPLITTER.splitToList(nextLine);

                String chr = tokens.get(0);
                int fragNumber = 0;
                int beg = Integer.parseInt(tokens.get(1)) - 1;  // 1 vs 0 based coords
                for (int i = 2; i < tokens.size(); i++) {
                    int end = Integer.parseInt(tokens.get(i)) - 1;
                    writer.println(chr + "\t" + beg + "\t" + end + "\t" + fragNumber);
                    beg = end;
                    fragNumber++;
                }

            }
        } finally {
            if (reader != null) reader.close();
        }

    }

    @Override
    public void readArguments(String[] args, CmdLineParser parser) {
        //setUsage("juicebox fragmentToBed <fragmentFile>");
        if (args.length != 2) {
            printUsageAndExit();
        }
        filename = args[1];
    }

    @Override
    public void run() {
        try {
            fragToBed(filename);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
