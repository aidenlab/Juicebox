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

package juicebox.tools.clt.juicer;

import jargs.gnu.CmdLineParser;
import juicebox.HiCGlobals;
import juicebox.tools.clt.CommandLineParserForJuicer;
import juicebox.tools.clt.JuicerCLT;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.Feature2DParser;
import juicebox.track.feature.Feature2DTools;

/**
 * Created by muhammadsaadshamim on 9/14/15.
 */
public class CompareLists extends JuicerCLT {

    private int threshold = 0, compareTypeID = 0;
    private String genomeID, inputFileA, inputFileB, outputPath = "comparison_list";

    public CompareLists() {
        super("compare [-m threshold] <compareType> <genomeID> <list1> <list2> [output_path]");
        HiCGlobals.useCache = false;
    }


    @Override
    public void readArguments(String[] args, CmdLineParser parser) {
        CommandLineParserForJuicer juicerParser = (CommandLineParserForJuicer) parser;
        //setUsage("juicebox arrowhead hicFile resolution");
        if (args.length != 5 && args.length != 6) {
            printUsage();
        }

        compareTypeID = Integer.parseInt(args[1]);
        genomeID = args[2];
        inputFileA = args[3];
        inputFileB = args[4];
        if (args.length == 6) {
            outputPath = args[5];
        }

        int specifiedMatrixSize = juicerParser.getMatrixSizeOption();
        if (specifiedMatrixSize > 0) {
            threshold = specifiedMatrixSize;
        }
    }


    @Override
    public void run() {

        if (compareTypeID == 0) {
            Feature2DList listA = Feature2DParser.loadFeatures(inputFileA, genomeID, true, null, false);
            Feature2DList listB = Feature2DParser.loadFeatures(inputFileB, genomeID, true, null, false);
            Feature2D.tolerance = this.threshold;
            Feature2DList listResults = Feature2DTools.compareLists(listA, listB);
            listResults.exportFeatureList(outputPath, false);
        } else if (compareTypeID == 1) {
            Feature2DList listA = Feature2DParser.loadFeatures(inputFileA, genomeID, true, null, false);
            Feature2DList listB = Feature2DParser.loadFeatures(inputFileB, genomeID, true, null, false);

        }
    }


}
