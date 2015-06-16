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

package juicebox.track.Feature;

import juicebox.tools.utils.Common.HiCFileTools;
import org.broad.igv.Globals;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.ui.color.ColorUtilities;
import org.broad.igv.util.ParsingUtils;

import java.awt.*;
import java.io.BufferedReader;
import java.io.IOException;
import java.util.*;
import java.util.List;

/**
 * Created by muhammadsaadshamim on 6/1/15.
 */
public class Feature2DParser {


    public static Feature2DList parseLoopFile(String path, List<Chromosome> chromosomes,
                                              boolean generateAPAFiltering,
                                              double minPeakDist, double maxPeakDist, int resolution,
                                              boolean loadAttributes) {

        Feature2DList newList = new Feature2DList();
        int attCol = 7;

        try {
            BufferedReader br = ParsingUtils.openBufferedReader(path);
            String nextLine;

            // header
            nextLine = br.readLine();
            String[] headers = Globals.tabPattern.split(nextLine);

            int errorCount = 0;
            int lineNum = 1;
            while ((nextLine = br.readLine()) != null) {
                lineNum++;
                String[] tokens = Globals.tabPattern.split(nextLine);
                if (tokens.length > headers.length) {
                    throw new IOException("Improperly formatted file");
                }
                if (tokens.length < 6) {
                    continue;
                }

                String chr1Name, chr2Name;
                int start1, end1, start2, end2;
                try {
                    chr1Name = tokens[0];
                    start1 = Integer.parseInt(tokens[1]);
                    end1 = Integer.parseInt(tokens[2]);

                    chr2Name = tokens[3];
                    start2 = Integer.parseInt(tokens[4]);
                    end2 = Integer.parseInt(tokens[5]);
                } catch (Exception e) {
                    throw new IOException("Line " + lineNum + " improperly formatted in <br>" +
                            path + "<br>Line format should start with:  CHR1  X1  X2  CHR2  Y1  Y2");
                }


                Color c = tokens.length > 6 ? ColorUtilities.stringToColor(tokens[6].trim()) : Color.black;

                Map<String, String> attrs = new LinkedHashMap<String, String>();
                if (loadAttributes){
                    for (int i = attCol; i < tokens.length; i++) {
                        attrs.put(headers[i], tokens[i]);
                    }
                }

                Chromosome chr1 = HiCFileTools.getChromosomeNamed(chr1Name, chromosomes);
                Chromosome chr2 = HiCFileTools.getChromosomeNamed(chr2Name, chromosomes);
                if (chr1 == null || chr2 == null) {
                    if (errorCount < 100) {
                        System.out.println("Skipping line: " + nextLine);
                    } else if (errorCount == 100) {
                        System.out.println("Maximum error count exceeded.  Further errors will not be logged");
                    }

                    errorCount++;
                    continue;
                }

                int featureNameSepindex = path.lastIndexOf("_");
                String featureName = path.substring(featureNameSepindex + 1);

                if (featureName.equals("blocks.txt")) {
                    featureName = Feature2D.domain;
                } else if (featureName.equals("peaks.txt")) {
                    featureName = Feature2D.peak;
                } else {
                    featureName = Feature2D.generic;
                }
                // Convention is chr1 is lowest "index". Swap if necessary
                Feature2D feature = chr1.getIndex() <= chr2.getIndex() ?
                        new Feature2D(featureName, chr1Name, start1, end1, chr2Name, start2, end2, c, attrs) :
                        new Feature2D(featureName, chr2Name, start2, end2, chr1Name, start1, end1, c, attrs);

                newList.add(chr1.getIndex(), chr2.getIndex(), feature);

            }

            br.close();
        }
        catch (IOException ec){
            ec.printStackTrace();
        }

        if(generateAPAFiltering)
            newList.apaFiltering(minPeakDist, maxPeakDist, resolution);

        return newList;
    }
}
