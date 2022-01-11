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

package juicebox.track.feature;

import juicebox.HiCGlobals;
import juicebox.MainWindow;
import juicebox.data.ChromosomeHandler;
import juicebox.data.HiCFileTools;
import juicebox.data.basics.Chromosome;
import juicebox.tools.utils.juicer.arrowhead.ArrowheadScoreList;
import juicebox.tools.utils.juicer.arrowhead.HighScore;
import org.broad.igv.Globals;
import org.broad.igv.ui.color.ColorUtilities;
import org.broad.igv.util.ParsingUtils;

import javax.swing.*;
import java.awt.*;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
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

    public static Feature2DList loadFeatures(String path, ChromosomeHandler handler, boolean loadAttributes,
                                             FeatureFilter featureFilter, final boolean useFeature2DWithMotif) {
        Feature2DList newList;
        String lowerCaseEnding = path.toLowerCase();
        if (lowerCaseEnding.endsWith(".bedpe")) {
            newList = parseGeneralFile(path, handler, loadAttributes, featureFilter, new SpecificParser() {
                @Override
                void parseAndAddToList(String path, String[] headers, String[] tokens, int lineNum, boolean loadAttributes,
                                       String nextLine, ChromosomeHandler handler,
                                       Feature2DList newList, Feature2D.FeatureType featureType) throws IOException {
                    parseBEDPEAndAddToList(path, headers, tokens, lineNum, loadAttributes, nextLine,
                            handler, newList, featureType, useFeature2DWithMotif);
                }
            });
        } else if (lowerCaseEnding.endsWith(".px")) {
            newList = parseGeneralFile(path, handler, loadAttributes, featureFilter, new SpecificParser() {
                @Override
                void parseAndAddToList(String path, String[] headers, String[] tokens, int lineNum, boolean loadAttributes,
                                       String nextLine, ChromosomeHandler handler,
                                       Feature2DList newList, Feature2D.FeatureType featureType) throws IOException {
                    parsePxLoopsAndAddToList(path, headers, tokens, lineNum, loadAttributes, nextLine, handler, newList, featureType);
                }
            });
        } else if (lowerCaseEnding.endsWith(".px2")) {
            newList = parseGeneralFile(path, handler, loadAttributes, featureFilter, new SpecificParser() {
                @Override
                void parseAndAddToList(String path, String[] headers, String[] tokens, int lineNum, boolean loadAttributes,
                                       String nextLine, ChromosomeHandler handler,
                                       Feature2DList newList, Feature2D.FeatureType featureType) throws IOException {
                    parseDomainsAndAddToList(path, headers, tokens, lineNum, loadAttributes, nextLine, handler, newList, featureType);
                }
            });
        } else {
            newList = parseGeneralFile(path, handler, loadAttributes, featureFilter, new SpecificParser() {
                @Override
                void parseAndAddToList(String path, String[] headers, String[] tokens, int lineNum, boolean loadAttributes,
                                       String nextLine, ChromosomeHandler handler,
                                       Feature2DList newList, Feature2D.FeatureType featureType) throws IOException {
                    parseLegacyLoopsAndAddToList(path, headers, tokens, lineNum, loadAttributes, nextLine,
                            handler, newList, featureType, useFeature2DWithMotif);
                }
            });
        }
        newList.removeDuplicates();
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


    private static Feature2DList parseGeneralFile(String path, ChromosomeHandler handler, boolean loadAttributes,
                                                  FeatureFilter featureFilter, SpecificParser parser) {
        Feature2DList newList = new Feature2DList();
        try {
            //BufferedReader br = ParsingUtils.openBufferedReader(path);
            BufferedReader br = new BufferedReader(new InputStreamReader(ParsingUtils.openInputStream(path)), HiCGlobals.bufferSize);
            String nextLine;

            // header
            nextLine = br.readLine();
            String[] headers = getHeaders(nextLine);
            Feature2D.FeatureType featureType = parseTitleForFeatureType(path);

            int lineNum = 1;
            while ((nextLine = br.readLine()) != null) {
                lineNum++;
                if (nextLine.startsWith("#")) continue;
                String[] tokens = Globals.tabPattern.split(nextLine);
                if (tokens.length > headers.length) {
                    String text = "Improperly formatted file: \nLine " + lineNum + " has " + tokens.length + " entries" +
                            " while header has " + headers.length;
                    System.err.println(text);
                    if (HiCGlobals.guiIsCurrentlyActive) {
                        JOptionPane.showMessageDialog(MainWindow.getInstance(), text, "Error", JOptionPane.ERROR_MESSAGE);
                    }
                    throw new IOException(text);
                }

                parser.parseAndAddToList(path, headers, tokens, lineNum, loadAttributes, nextLine, handler, newList, featureType);
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
        String[] tmpHeaders = Globals.tabPattern.split(line.replaceAll("#", "").trim());
        String[] headers = new String[tmpHeaders.length];

        Map<String, String> translator = new HashMap<>();
        translator.put("o", "observed");
        translator.put("e_bl", "expectedBL");
        translator.put("e_donut", "expectedDonut");
        translator.put("e_h", "expectedH");
        translator.put("e_v", "expectedV");
        translator.put("fdr_bl", "fdrBL");
        translator.put("fdr_donut", "fdrDonut");
        translator.put("fdr_h", "fdrH");
        translator.put("fdr_v", "fdrV");
        translator.put("num_collapsed", "numCollapsed");

        for (int i = 0; i < tmpHeaders.length; i++) {
            if (translator.containsKey(tmpHeaders[i])) {
                headers[i] = translator.get(tmpHeaders[i]);
            } else {
                headers[i] = tmpHeaders[i];
            }
        }
        return headers;
    }

    private static abstract class SpecificParser {
        private static final int errorLimit = 100;
        private static int errorCount = 0;

        SpecificParser() {
            errorCount = 0;
        }

        private static Map<String, String> parseAttributes(boolean loadAttributes, int attCol, String[] headers, String[] tokens) {
            Map<String, String> attrs = new LinkedHashMap<>();
            if (loadAttributes && tokens.length > attCol) {
                for (int i = attCol; i < tokens.length; i++) {
                    attrs.put(headers[i], tokens[i]);
                }
            }
            return attrs;
        }

        private static void addToList(String chr1Name, String chr2Name, ChromosomeHandler handler, String nextLine,
                                      Feature2DList newList, boolean useFeature2DWithMotif, Feature2D.FeatureType featureType,
                                      int start1, int end1, int start2, int end2, Color c, Map<String, String> attrs) {
            Chromosome chr1 = handler.getChromosomeFromName(chr1Name);
            Chromosome chr2 = handler.getChromosomeFromName(chr2Name);
            if (chr1 == null || chr2 == null) {
                handleError(nextLine);
                return;
            }

            // Convention is chr1 is lowest "index". Swap if necessary
            if (useFeature2DWithMotif) {
                if (chr1.getIndex() <= chr2.getIndex()) {
                    newList.add(chr1.getIndex(), chr2.getIndex(), new Feature2DWithMotif(featureType, chr1Name, start1, end1, chr2Name, start2, end2, c, attrs));
                } else {
                    newList.add(chr2.getIndex(), chr1.getIndex(), new Feature2DWithMotif(featureType, chr2Name, start2, end2, chr1Name, start1, end1, c, attrs));
                }
            } else {
                if (chr1.getIndex() <= chr2.getIndex()) {
                    newList.add(chr1.getIndex(), chr2.getIndex(), new Feature2D(featureType, chr1Name, start1, end1, chr2Name, start2, end2, c, attrs));
                } else {
                    newList.add(chr2.getIndex(), chr1.getIndex(), new Feature2D(featureType, chr2Name, start2, end2, chr1Name, start1, end1, c, attrs));
                }
            }
        }

        private static void addToList(String chrAName, Feature2D feature, ChromosomeHandler handler, String nextLine,
                                      Feature2DList newList) {
            Chromosome chrA = handler.getChromosomeFromName(chrAName);
            if (chrA == null) {
                handleError(nextLine);
                return;
            }
            newList.add(chrA.getIndex(), chrA.getIndex(), feature);
        }

        private static void handleError(String nextLine) {
            if (HiCGlobals.printVerboseComments) {
                if (errorCount < errorLimit) {
                    System.err.println("Skipping line: " + nextLine);
                } else if (errorCount == errorLimit) {
                    System.err.println("Maximum error count exceeded.  Further errors will not be logged");
                }
            }
            errorCount++;
        }

        abstract void parseAndAddToList(String path, String[] headers, String[] tokens, int lineNum, boolean loadAttributes, String nextLine,
                                        ChromosomeHandler handler, Feature2DList newList, Feature2D.FeatureType featureType) throws IOException;

        void parseDomainsAndAddToList(String path, String[] headers, String[] tokens, int lineNum, boolean loadAttributes, String nextLine,
                                      ChromosomeHandler handler, Feature2DList newList, Feature2D.FeatureType featureType) throws IOException {
            if (tokens.length < 3) return;

            String chrAName;
            int startA, endA;
            try {
                chrAName = tokens[0];
                startA = Integer.parseInt(tokens[1]);
                endA = Integer.parseInt(tokens[2]);
            } catch (Exception e) {
                String text = "Line " + lineNum + " improperly formatted in <br>" + path + "<br>Line format should start with:  CHR1  X1  X2";
                System.err.println(text);
                if (HiCGlobals.guiIsCurrentlyActive)
                    JOptionPane.showMessageDialog(MainWindow.getInstance(), text, "Error", JOptionPane.ERROR_MESSAGE);
                throw new IOException(text);
            }

            Color c = Color.black;
            Map<String, String> attrs = parseAttributes(loadAttributes, 3, headers, tokens);
            Feature2D feature = new Feature2D(featureType, chrAName, startA, endA, chrAName, startA, endA, c, attrs);

            addToList(chrAName, feature, handler, nextLine, newList);
        }

        void parseBEDPEAndAddToList(String path, String[] headers, String[] tokens, int lineNum,
                                    boolean loadAttributes, String nextLine, ChromosomeHandler handler,
                                    Feature2DList newList, Feature2D.FeatureType featureType,
                                    boolean useFeature2DWithMotif) throws IOException {
            // BEDPE format
            // chrom1 start1 end1 chrom2 start2 end2 name(opt) score(opt) strand1(opt) strand2(opt) ...(opt)...
            // we will check for color in 11th column
            if (tokens.length < 6) return;
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
                String text = "Line " + lineNum + " improperly formatted in \n" +
                        path + "\nLine format should start with:  CHR1  X1  X2  CHR2  Y1  Y2";
                System.err.println(text);
                if (HiCGlobals.guiIsCurrentlyActive)
                    JOptionPane.showMessageDialog(MainWindow.getInstance(), text, "Error", JOptionPane.ERROR_MESSAGE);
                throw new IOException(text);
            }

            Color c = tokens.length > 10 ? ColorUtilities.stringToColor(tokens[10].trim()) : Color.black;
            Map<String, String> attrs = parseAttributes(loadAttributes, 11, headers, tokens);

            try {
                attrs.put("score", "" + Float.parseFloat(tokens[7]));
            } catch (Exception e) {
                //ignore
            }

            addToList(chr1Name, chr2Name, handler, nextLine, newList, useFeature2DWithMotif, featureType,
                    start1, end1, start2, end2, c, attrs);
        }

        void parseLegacyLoopsAndAddToList(String path, String[] headers, String[] tokens, int lineNum,
                                          boolean loadAttributes, String nextLine, ChromosomeHandler handler,
                                          Feature2DList newList, Feature2D.FeatureType featureType,
                                          boolean useFeature2DWithMotif) throws IOException {
            if (tokens.length < 6) return;
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
            Map<String, String> attrs = parseAttributes(loadAttributes, 7, headers, tokens);

            addToList(chr1Name, chr2Name, handler, nextLine, newList, useFeature2DWithMotif, featureType,
                    start1, end1, start2, end2, c, attrs);
        }

        void parsePxLoopsAndAddToList(String path, String[] headers, String[] tokens, int lineNum,
                                      boolean loadAttributes, String nextLine, ChromosomeHandler handler,
                                      Feature2DList newList, Feature2D.FeatureType featureType) throws IOException {
            // this was the prelim output used for old hiccups debugging
            if (tokens.length < 4) return;
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
            Map<String, String> attrs = parseAttributes(loadAttributes, 5, headers, tokens);
            addToList(chr1Name, chr2Name, handler, nextLine, newList, false, featureType,
                    start1, end1, start2, end2, c, attrs);
        }
    }
}