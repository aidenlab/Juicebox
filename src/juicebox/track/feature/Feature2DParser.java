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

package juicebox.track.feature;

import juicebox.HiCGlobals;
import juicebox.MainWindow;
import juicebox.data.HiCFileTools;
import juicebox.tools.utils.juicer.arrowhead.ArrowheadScoreList;
import juicebox.tools.utils.juicer.arrowhead.HighScore;
import org.broad.igv.Globals;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.ui.color.ColorUtilities;
import org.broad.igv.util.ParsingUtils;

import javax.swing.*;
import java.awt.*;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by muhammadsaadshamim on 6/1/15.
 */
public class Feature2DParser {

    public static Feature2DList loadFeatures(String path, String genomeID, boolean loadAttributes,
                                             FeatureFilter featureFilter, boolean useFeature2DWithMotif) {
        return loadFeatures(path, HiCFileTools.loadChromosomes(genomeID), loadAttributes, featureFilter, useFeature2DWithMotif);
    }

    public static Feature2DList loadFeatures(String path, List<Chromosome> chromosomes, boolean loadAttributes,
                                             FeatureFilter featureFilter, boolean useFeature2DWithMotif) {
        Feature2DList newList;
        if (path.endsWith(".px")) {
            newList = parseHiCCUPSLoopFile(path, chromosomes, loadAttributes, featureFilter);
        } else if (path.endsWith(".px2")) {
            newList = parseDomainFile(path, chromosomes, loadAttributes, featureFilter);
        } else {
            newList = parseLoopFile(path, chromosomes, loadAttributes, featureFilter, useFeature2DWithMotif);
        }
        newList.removeDuplicates();
        return newList;
    }


    private static Feature2DList parseLoopFile(String path, List<Chromosome> chromosomes, boolean loadAttributes,
                                               FeatureFilter featureFilter, boolean useFeature2DWithMotif) {

        Feature2DList newList = new Feature2DList();
        int attCol = 7;
        try {

            BufferedReader br = ParsingUtils.openBufferedReader(path);
            String nextLine;

            // header
            nextLine = br.readLine();
            if (nextLine == null || nextLine.length() < 1) {
                System.err.println("Empty list provided");
                System.exit(23);
            }

            String[] headers = getHeaders(nextLine);

            int errorCount = 0;
            int lineNum = 1;
            while ((nextLine = br.readLine()) != null) {
                lineNum++;
                String[] tokens = Globals.tabPattern.split(nextLine);
                if (tokens.length > headers.length) { //TODO why greater, use "!=" ? (also below)
                    String text = "Improperly formatted file: \nLine " + lineNum + " has " + tokens.length + " entries" +
                            " while header has " + headers.length;
                    System.err.println(text);
                    if (HiCGlobals.guiIsCurrentlyActive)
                        JOptionPane.showMessageDialog(MainWindow.getInstance(), text, "Error", JOptionPane.ERROR_MESSAGE);
                    throw new IOException(text);
                }
                if (tokens.length < attCol - 1) { // attcol-1 because color is 7th column
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
                    String text = "Line " + lineNum + " improperly formatted in <br>" +
                            path + "<br>Line format should start with:  CHR1  X1  X2  CHR2  Y1  Y2";
                    System.err.println(text);
                    if (HiCGlobals.guiIsCurrentlyActive)
                        JOptionPane.showMessageDialog(MainWindow.getInstance(), text, "Error", JOptionPane.ERROR_MESSAGE);
                    throw new IOException(text);
                }


                Color c = tokens.length > 6 ? ColorUtilities.stringToColor(tokens[6].trim()) : Color.black;

                Map<String, String> attrs = new LinkedHashMap<String, String>();
                if (loadAttributes) {
                    for (int i = attCol; i < tokens.length; i++) {
                        attrs.put(headers[i], tokens[i]);
                    }
                }

                Chromosome chr1 = HiCFileTools.getChromosomeNamed(chr1Name, chromosomes);
                Chromosome chr2 = HiCFileTools.getChromosomeNamed(chr2Name, chromosomes);
                if (chr1 == null || chr2 == null) {
                    if (HiCGlobals.printVerboseComments) {
                        if (errorCount < 100) {
                            System.out.println("Skipping line: " + nextLine);
                        } else if (errorCount == 100) {
                            System.out.println("Maximum error count exceeded.  Further errors will not be logged");
                        }
                    }

                    errorCount++;
                    continue;
                }

                Feature2D.FeatureType featureType = parseTitleForFeatureType(path);

                // Convention is chr1 is lowest "index". Swap if necessary
                Feature2D feature;
                if (useFeature2DWithMotif) {
                    feature = chr1.getIndex() <= chr2.getIndex() ?
                            new Feature2DWithMotif(featureType, chr1Name, chr1.getIndex(), start1, end1, chr2Name, chr2.getIndex(), start2, end2, c, attrs) :
                            new Feature2DWithMotif(featureType, chr2Name, chr2.getIndex(), start2, end2, chr1Name, chr1.getIndex(), start1, end1, c, attrs);
                } else {
                    feature = chr1.getIndex() <= chr2.getIndex() ?
                            new Feature2D(featureType, chr1Name, start1, end1, chr2Name, start2, end2, c, attrs) :
                            new Feature2D(featureType, chr2Name, start2, end2, chr1Name, start1, end1, c, attrs);
                }

                newList.add(chr1.getIndex(), chr2.getIndex(), feature);

            }

            br.close();
        } catch (Exception ec) {
            if (HiCGlobals.guiIsCurrentlyActive) {
                ec.printStackTrace();
            } else {
                System.err.println("File " + path + " could not be parsed");
            }
        }

        if (featureFilter != null)
            newList.filterLists(featureFilter);

        return newList;
    }

    private static Feature2D.FeatureType parseTitleForFeatureType(String path) {
        if (path.contains("block") || path.contains("domain")) {
            return Feature2D.FeatureType.DOMAIN;
        } else if (path.contains("peak") || path.contains("loop")) {
            return Feature2D.FeatureType.PEAK;
        } else {
            return Feature2D.FeatureType.GENERIC;
        }
    }

    public static Feature2DList parseHighScoreList(int chrIndex, String chrName, int resolution, List<HighScore> binnedScores) {
        Feature2DList feature2DList = new Feature2DList();

        for (HighScore score : binnedScores) {
            feature2DList.add(chrIndex, chrIndex, score.toFeature2D(chrName, resolution));
        }

        return feature2DList;
    }

    public static Feature2DList parseArrowheadScoreList(int chrIndex, String chrName,
                                                        ArrowheadScoreList scoreList) {
        Feature2DList feature2DList = new Feature2DList();
        feature2DList.add(scoreList.toFeature2DList(chrIndex, chrName));
        return feature2DList;
    }

    private static Feature2DList parseHiCCUPSLoopFile(String path, List<Chromosome> chromosomes,
                                                      boolean loadAttributes, FeatureFilter featureFilter) {
        Feature2DList newList = new Feature2DList();
        int attCol = 4;

        try {
            //BufferedReader br = ParsingUtils.openBufferedReader(path);
            BufferedReader br = new BufferedReader(new InputStreamReader(ParsingUtils.openInputStream(path)), HiCGlobals.bufferSize);
            String nextLine;

            // header
            nextLine = br.readLine();
            String[] headers = getHeaders(nextLine);

            int errorCount = 0;
            int lineNum = 1;
            while ((nextLine = br.readLine()) != null) {
                lineNum++;
                String[] tokens = Globals.tabPattern.split(nextLine);
                if (tokens.length > headers.length) {
                    String text = "Improperly formatted file: \nLine " + lineNum + " has " + tokens.length + " entries" +
                            " while header has " + headers.length;
                    System.err.println(text);
                    if (HiCGlobals.guiIsCurrentlyActive)
                        JOptionPane.showMessageDialog(MainWindow.getInstance(), text, "Error", JOptionPane.ERROR_MESSAGE);
                    throw new IOException(text);
                }
                if (tokens.length < attCol) { // attcol because no color
                    continue;
                }

                String chr1Name, chr2Name;
                int start1, end1, start2, end2;
                try {
                    chr1Name = tokens[0];
                    start1 = Integer.parseInt(tokens[1]);
                    end1 = start1 + 5000;

                    chr2Name = tokens[2];
                    start2 = Integer.parseInt(tokens[3]);
                    end2 = start2 + 5000;
                } catch (Exception e) {
                    String text = "Line " + lineNum + " improperly formatted in <br>" +
                            path + "<br>Line format should start with:  CHR1  X1  CHR2  Y1";
                    System.err.println(text);
                    if (HiCGlobals.guiIsCurrentlyActive)
                        JOptionPane.showMessageDialog(MainWindow.getInstance(), text, "Error", JOptionPane.ERROR_MESSAGE);
                    throw new IOException(text);
                }


                Color c = tokens.length > 4 ? ColorUtilities.stringToColor(tokens[4].trim()) : Color.black;


                Map<String, String> attrs = new LinkedHashMap<String, String>();
                if (loadAttributes) {
                    for (int i = attCol + 1; i < tokens.length; i++) {
                        attrs.put(headers[i], tokens[i]);
                    }
                }

                Chromosome chr1 = HiCFileTools.getChromosomeNamed(chr1Name, chromosomes);
                Chromosome chr2 = HiCFileTools.getChromosomeNamed(chr2Name, chromosomes);
                if (chr1 == null || chr2 == null) {
                    if (HiCGlobals.printVerboseComments) {
                        if (errorCount < 100) {
                            System.err.println("Skipping line: " + nextLine);
                        } else if (errorCount == 100) {
                            System.err.println("Maximum error count exceeded.  Further errors will not be logged");
                        }
                    }

                    errorCount++;
                    continue;
                }

                Feature2D.FeatureType featureType = parseTitleForFeatureType(path);

                // Convention is chr1 is lowest "index". Swap if necessary
                Feature2D feature = chr1.getIndex() <= chr2.getIndex() ?
                        new Feature2D(featureType, chr1Name, start1, end1, chr2Name, start2, end2, c, attrs) :
                        new Feature2D(featureType, chr2Name, start2, end2, chr1Name, start1, end1, c, attrs);

                newList.add(chr1.getIndex(), chr2.getIndex(), feature);

            }

            br.close();
        } catch (Exception ec) {
            if (HiCGlobals.guiIsCurrentlyActive) {
                ec.printStackTrace();
            } else {
                System.err.println("File " + path + " could not be parsed");
            }
        }

        if (featureFilter != null)
            newList.filterLists(featureFilter);

        return newList;
    }

    private static Feature2DList parseDomainFile(String path, List<Chromosome> chromosomes,
                                                 boolean loadAttributes, FeatureFilter featureFilter) {
        Feature2DList newList = new Feature2DList();
        int attCol = 3;

        try {
            //BufferedReader br = ParsingUtils.openBufferedReader(path);
            BufferedReader br = new BufferedReader(new InputStreamReader(ParsingUtils.openInputStream(path)), HiCGlobals.bufferSize);
            String nextLine;

            // header
            nextLine = br.readLine();
            String[] headers = getHeaders(nextLine);

            int errorCount = 0;
            int lineNum = 1;
            while ((nextLine = br.readLine()) != null) {
                lineNum++;
                String[] tokens = Globals.tabPattern.split(nextLine);
                if (tokens.length > headers.length) {
                    String text = "Improperly formatted file: \nLine " + lineNum + " has " + tokens.length + " entries" +
                            " while header has " + headers.length;
                    System.err.println(text);
                    if (HiCGlobals.guiIsCurrentlyActive)
                        JOptionPane.showMessageDialog(MainWindow.getInstance(), text, "Error", JOptionPane.ERROR_MESSAGE);
                    throw new IOException(text);
                }
                if (tokens.length < attCol) { // attcol because no color
                    continue;
                }

                String chrAName;
                int startA, endA;
                try {
                    chrAName = tokens[0];
                    startA = Integer.parseInt(tokens[1]);
                    endA = Integer.parseInt(tokens[2]);
                } catch (Exception e) {
                    String text = "Line " + lineNum + " improperly formatted in <br>" +
                            path + "<br>Line format should start with:  CHR1  X1  X2";
                    System.err.println(text);
                    if (HiCGlobals.guiIsCurrentlyActive)
                        JOptionPane.showMessageDialog(MainWindow.getInstance(), text, "Error", JOptionPane.ERROR_MESSAGE);
                    throw new IOException(text);
                }

                Color c = Color.black;
                Map<String, String> attrs = new LinkedHashMap<String, String>();
                if (loadAttributes) {
                    for (int i = attCol; i < tokens.length; i++) {
                        attrs.put(headers[i], tokens[i]);
                    }
                }

                Chromosome chrA = HiCFileTools.getChromosomeNamed(chrAName, chromosomes);
                if (chrA == null) {
                    if (HiCGlobals.printVerboseComments) {
                        if (errorCount < 100) {
                            System.err.println("Skipping line: " + nextLine);
                        } else if (errorCount == 100) {
                            System.err.println("Maximum error count exceeded.  Further errors will not be logged");
                        }
                    }

                    errorCount++;
                    continue;
                }

                Feature2D.FeatureType featureType = parseTitleForFeatureType(path);

                // Convention is chr1 is lowest "index". Swap if necessary
                Feature2D feature = new Feature2D(featureType, chrAName, startA, endA, chrAName, startA, endA, c, attrs);

                newList.add(chrA.getIndex(), chrA.getIndex(), feature);
            }

            br.close();
        } catch (Exception ec) {
            if (HiCGlobals.guiIsCurrentlyActive) {
                ec.printStackTrace();
            } else {
                System.err.println("File " + path + " could not be parsed");
            }
        }

        if (featureFilter != null)
            newList.filterLists(featureFilter);

        return newList;
    }
    /**
     * Backwards compatibility with original loops list
     * @param line Header token, usually not converted but old ones will be
     * @return  Appropriate header
     */
    private static String[] getHeaders(String line) {
        String[] tmpHeaders = Globals.tabPattern.split(line);
        String[] headers = new String[tmpHeaders.length];

        for (int i=0; i<tmpHeaders.length; i++) {
            if (tmpHeaders[i].equals("o")) headers[i] = "observed";
            else if (tmpHeaders[i].equals("e_bl")) headers[i] = "expectedBL";
            else if (tmpHeaders[i].equals("e_donut")) headers[i] = "expectedDonut";
            else if (tmpHeaders[i].equals("e_h")) headers[i] = "expectedH";
            else if (tmpHeaders[i].equals("e_v")) headers[i] = "expectedV";
            else if (tmpHeaders[i].equals("fdr_bl")) headers[i] = "fdrBL";
            else if (tmpHeaders[i].equals("fdr_donut")) headers[i] = "fdrDonut";
            else if (tmpHeaders[i].equals("fdr_h")) headers[i] = "fdrH";
            else if (tmpHeaders[i].equals("fdr_v")) headers[i] = "fdrV";
            else if (tmpHeaders[i].equals("num_collapsed")) headers[i] = "numCollapsed";
            else headers[i] = tmpHeaders[i];
        }
        return headers;
    }

}