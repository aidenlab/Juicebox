/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2019 Broad Institute, Aiden Lab
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

package juicebox.tools.dev;

import juicebox.data.ChromosomeHandler;
import juicebox.data.HiCFileTools;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.Feature2DParser;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

class RandomUtils {

    /**
     * Usage
     * <p>
     * String[] inputFiles = new String[]{"/Users/muhammad/Desktop/loops/gm12878.bedpe",
     * "/Users/muhammad/Desktop/loops/hela.bedpe",
     * "/Users/muhammad/Desktop/loops/imr90.bedpe",
     * "/Users/muhammad/Desktop/loops/k562.bedpe",
     * "/Users/muhammad/Desktop/loops/nhek.bedpe",
     * "/Users/muhammad/Desktop/loops/hap1.bedpe",
     * "/Users/muhammad/Desktop/loops/hmec.bedpe",
     * "/Users/muhammad/Desktop/loops/huvec.bedpe"};
     * <p>
     * <p>
     * RandomUtils.mergeLoopLists("hg19", inputFiles, new File("/Users/muhammad/Desktop/loops/merged.bedpe"));
     *
     * @param genomeID
     * @param fileNames
     * @param outputFile
     */
    public static void mergeLoopLists(String genomeID, String[] fileNames, File outputFile) {

        ChromosomeHandler handler = HiCFileTools.loadChromosomes(genomeID);

        List<Feature2DList> lists = new ArrayList<>();
        for (String fileName : fileNames) {
            lists.add(Feature2DParser.loadFeatures(fileName, handler, false, null, false));
        }

        Feature2D.tolerance = 10000;

        Feature2DList exactMatches = Feature2DList.getIntersection(lists.get(0), lists.get(0));

        for (Feature2DList list : lists) {
            exactMatches = Feature2DList.getIntersection(exactMatches, list);
            int numExactMatches = exactMatches.getNumTotalFeatures();
            System.out.println("Number of exact matches: " + numExactMatches);
        }

        exactMatches.exportFeatureList(outputFile, true, Feature2DList.ListFormat.FINAL);
    }
}
