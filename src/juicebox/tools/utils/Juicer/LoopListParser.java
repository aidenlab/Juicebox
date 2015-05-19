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

package juicebox.tools.utils.Juicer;

import juicebox.tools.utils.Common.HiCFileTools;
import juicebox.track.Feature2D;
import juicebox.track.FeatureCoordinate;
import org.broad.igv.Globals;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.util.ParsingUtils;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.*;


/**
 * Created by muhammadsaadshamim on 5/4/15.
 */
public class LoopListParser {

    /**
     * Reads loops from text file
     *
     *
     * @param chromosomes
     * @param path
     * @param minPeakDist
     * @param maxPeakDist
     * @return
     * @throws IOException
     */
    public static LoopContainer parseList(String path,
                                          List<Chromosome> chromosomes,
                                          double minPeakDist,
                                          double maxPeakDist,
                                          int resolution) throws IOException {
        BufferedReader br = null;

        Map<String, ArrayList<Feature2D>> chrToLoopsMap = new HashMap<String, ArrayList<Feature2D>>();

        try {
            br = ParsingUtils.openBufferedReader(path);
            String nextLine;

            nextLine = br.readLine(); // header
            String[] headers = Globals.tabPattern.split(nextLine);

            // TODO int errorCount = 0;
            int lineNum = 1;
            while ((nextLine = br.readLine()) != null) {
                lineNum++;
                String[] tokens = Globals.tabPattern.split(nextLine);
                if (tokens.length > headers.length) {
                    throw new IOException("Improperly formatted loop file");
                }
                if (tokens.length < 6) {
                    continue;
                }

                String chr1Name, chr2Name;
                int start1, end1, start2, end2;
                try {
                    chr1Name = tokens[0].toLowerCase();
                    start1 = Integer.parseInt(tokens[1]);
                    end1 = Integer.parseInt(tokens[2]);

                    chr2Name = tokens[3].toLowerCase();
                    start2 = Integer.parseInt(tokens[4]);
                    end2 = Integer.parseInt(tokens[5]);

                } catch (Exception e) {
                    throw new IOException("Line " + lineNum + " improperly formatted in <br>" +
                            path + "<br>Line format should start with:  CHR1  X1  X2  CHR2  Y1  Y2");
                }

                if (chr1Name.equals(chr2Name)) {
                    FeatureCoordinate coord1 = new FeatureCoordinate(chr1Name, start1, end1);
                    FeatureCoordinate coord2 = new FeatureCoordinate(chr2Name, start2, end2);
                    Feature2D feature;

                    // arbitrary order is not important; but this allows for easy removal of duplicates
                    if (coord1.compareTo(coord2) < 0) {
                        feature = new Feature2D(Feature2D.peak, coord1, coord2, null, null);
                    } else {
                        feature = new Feature2D(Feature2D.peak, coord2, coord1, null, null);
                    }

                    if (chrToLoopsMap.containsKey(chr1Name)) {
                        chrToLoopsMap.get(chr1Name).add(feature);
                    } else {
                        ArrayList<Feature2D> newList = new ArrayList<Feature2D>();
                        newList.add(feature);
                        chrToLoopsMap.put(chr1Name, newList);
                    }
                }
            }
        } finally {
            if (br != null) br.close();
        }

        /** now that data has been extracted, it will be filtered down */
        Map<Chromosome, Set<Feature2D>> filteredChrToLoopsMap = new HashMap<Chromosome, Set<Feature2D>>();
        Map<Chromosome, Integer[]> chrToTotalsMap = new HashMap<Chromosome, Integer[]>();

        Set<String> keys = chrToLoopsMap.keySet();


        // filter down loops by uniqueness, then size, and save the totals at each stage
        for (Chromosome chrKey : chromosomes) {
            for (String stringKey : keys) {
                if (HiCFileTools.equivalentChromosome(stringKey, chrKey)) {
                    ArrayList<Feature2D> loops = chrToLoopsMap.get(stringKey);
                    ArrayList<Feature2D> uniqueLoops = filterLoopsByUniqueness(loops);
                    ArrayList<Feature2D> filteredUniqueLoops = filterLoopsBySize(uniqueLoops,
                            minPeakDist, maxPeakDist, resolution);

                    filteredChrToLoopsMap.put(chrKey, new HashSet<Feature2D>(filteredUniqueLoops));
                    chrToTotalsMap.put(chrKey, new Integer[]{filteredUniqueLoops.size(), uniqueLoops.size(), loops.size()});

                    keys.remove(stringKey);
                    break;
                }
            }
        }

        return new LoopContainer(filteredChrToLoopsMap, chrToTotalsMap) ;
    }

    /**
     * remove duplicates by using a hashset intermediate
     * @param loops
     * @return
     */
    private static ArrayList<Feature2D> filterLoopsByUniqueness(List<Feature2D> loops) {
        return new ArrayList<Feature2D>(new HashSet<Feature2D>(loops));
    }

    /**
     * Size filtering of loops
     *
     * @param loops
     * @param minPeakDist
     * @param maxPeakDist
     * @return
     */
    private static ArrayList<Feature2D> filterLoopsBySize(List<Feature2D> loops,
                                                          double minPeakDist,
                                                          double maxPeakDist,
                                                          int resolution) {
        ArrayList<Feature2D> sizeFilteredLoops = new ArrayList<Feature2D>();

        for (Feature2D loop : loops) {
            int xMidPt = loop.getMidPt1();
            int yMidPt = loop.getMidPt2();

            int dist = Math.abs(xMidPt - yMidPt);
            if (dist >= minPeakDist*resolution)
                if (dist <= maxPeakDist*resolution)
                    sizeFilteredLoops.add(loop);
        }

        return new ArrayList<Feature2D>(sizeFilteredLoops);
    }


}
