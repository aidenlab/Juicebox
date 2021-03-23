/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2021 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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
 *  FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.tools.clt.old;

import juicebox.data.ContactRecord;
import juicebox.data.iterator.IteratorContainer;
import juicebox.data.iterator.ListIteratorContainer;
import juicebox.tools.clt.CommandLineParser;
import juicebox.tools.clt.JuiceboxCLT;
import juicebox.tools.utils.norm.NormalizationCalculations;
import juicebox.windowui.NormalizationHandler;
import org.broad.igv.Globals;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


public class CalcKR extends JuiceboxCLT {

    private String infile = null;

    public CalcKR() {
        super("calcKR <input_?_file>");
    }

    @Override
    public void readArguments(String[] args, CommandLineParser parser) {
        //setUsage("juicebox calcKR <infile>");
        if (!(args.length == 2)) {
            printUsageAndExit();
        }
        infile = args[1];
    }

    public static void calcKR(String path) throws IOException {

        BufferedReader reader = org.broad.igv.util.ParsingUtils.openBufferedReader(path);

        String nextLine;
        int lineCount = 0;
        int maxBin = 0;
        List<ContactRecord> readList = new ArrayList<>();
        while ((nextLine = reader.readLine()) != null) {
            lineCount++;
            String[] tokens = Globals.singleTabMultiSpacePattern.split(nextLine);
            int nTokens = tokens.length;
            if (nTokens != 3) {
                System.err.println("Number of columns incorrect at line" + lineCount + ": " + nextLine);
                System.exit(62);
            }
            int binX = Integer.parseInt(tokens[0]);
            int binY = Integer.parseInt(tokens[1]);
            int count = Integer.parseInt(tokens[2]);
            ContactRecord record = new ContactRecord(binX, binY, count);
            readList.add(record);
            if (binX > maxBin) maxBin = binX;
            if (binY > maxBin) maxBin = binY;
        }
        IteratorContainer ic = new ListIteratorContainer(readList, maxBin + 1);

        NormalizationCalculations nc = new NormalizationCalculations(ic);
        for (float[] array : nc.getNorm(NormalizationHandler.KR).getValues()) {
            for (double d : array) {
                System.out.println(d);
            }
        }

    }

    @Override
    public void run() {
        try {
            calcKR(infile);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
