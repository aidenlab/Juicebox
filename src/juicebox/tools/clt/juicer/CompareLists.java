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

package juicebox.tools.clt.juicer;

import juicebox.HiCGlobals;
import juicebox.data.HiCFileTools;
import juicebox.data.anchor.MotifAnchorTools;
import juicebox.tools.clt.CommandLineParserForJuicer;
import juicebox.tools.clt.JuicerCLT;
import juicebox.track.feature.*;
import org.broad.igv.feature.Chromosome;

import java.awt.*;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by muhammadsaadshamim on 9/14/15.
 */
public class CompareLists extends JuicerCLT {

    public final static String PARENT_ATTRIBUTE = "parent_list";
    public static final Color AAA = new Color(102, 0, 153);
    public static final Color BBB = new Color(255, 102, 0);
    /**
     * Arbitrary colors for comparison list
     **/
    private static final Color AB = new Color(34, 139, 34);
    private static final Color AA = new Color(0, 255, 150);
    private static final Color BB = new Color(150, 255, 0);
    private int threshold = 10000, compareTypeID = 0;
    private String genomeID, inputFileA, inputFileB, outputPath = "comparison_list";

    public CompareLists() {
        super("compare [-m threshold] [-c chromosome(s)] <compareType> <genomeID> <list1> <list2> [output_path]");
        HiCGlobals.useCache = false;
    }

    @Override
    protected void readJuicerArguments(String[] args, CommandLineParserForJuicer juicerParser) {
        if (args.length != 5 && args.length != 6) {
            printUsageAndExit();
        }

        compareTypeID = Integer.parseInt(args[1]);
        genomeID = args[2];
        inputFileA = args[3];
        inputFileB = args[4];
        if (args.length == 6) {
            outputPath = args[5];
        } else {
            if (inputFileB.endsWith(".txt")) {
                outputPath = inputFileB.substring(0, inputFileB.length() - 4) + "_comparison_results.txt";
            } else {
                outputPath = inputFileB + "_comparison_results.txt";
            }
        }

        int specifiedMatrixSize = juicerParser.getMatrixSizeOption();
        if (specifiedMatrixSize >= 0) {
            threshold = specifiedMatrixSize;
        }
    }


    @Override
    public void run() {

        List<Chromosome> chromosomes = HiCFileTools.loadChromosomes(genomeID);
        if (givenChromosomes != null)
            chromosomes = new ArrayList<Chromosome>(HiCFileTools.stringToChromosomes(givenChromosomes,
                    chromosomes));

        Feature2DList listA = null, listB = null;
        if (compareTypeID == 0) {
            listA = Feature2DParser.loadFeatures(inputFileA, chromosomes, false, null, false);
            listB = Feature2DParser.loadFeatures(inputFileB, chromosomes, false, null, false);

        } else if (compareTypeID == 1 || compareTypeID == 2) {
            Feature2DWithMotif.useSimpleOutput = true;
            listA = Feature2DParser.loadFeatures(inputFileA, chromosomes, true, null, true);
            listB = Feature2DParser.loadFeatures(inputFileB, chromosomes, true, null, true);
        }

        if (compareTypeID == 2) {
            generateHistogramMetrics(listB);
        } else {
            compareTwoLists(listA, listB, compareTypeID);
        }
    }

    private void generateHistogramMetrics(Feature2DList list) {
        final int[] metrics = MotifAnchorTools.calculateConvergenceHistogram(list);
        System.out.println("++ : " + metrics[0] + " +- : " + metrics[1] + " -+ : " + metrics[2] + " -- : " + metrics[3]);
    }

    private void compareTwoLists(Feature2DList listA, Feature2DList listB, int compareTypeID) {
        int sizeA = listA.getNumTotalFeatures();
        int sizeB = listB.getNumTotalFeatures();
        System.out.println("List Size: " + sizeA + "(A) " + sizeB + "(B)");

        if (compareTypeID == 0) {
            Feature2D.tolerance = 0;
        } else if (compareTypeID == 1) {
            Feature2D.tolerance = threshold;
        }
        Feature2DWithMotif.lenientEqualityEnabled = false;

        Feature2DList exactMatches = Feature2DList.getIntersection(listA, listB);
        int numExactMatches = exactMatches.getNumTotalFeatures();
        System.out.println("Number of exact matches: " + numExactMatches);

        Feature2D.tolerance = this.threshold;
        Feature2DWithMotif.lenientEqualityEnabled = true;
        Feature2DList matchesWithinToleranceFromA = Feature2DList.getIntersection(listA, listB);
        Feature2DList matchesWithinToleranceFromB = Feature2DList.getIntersection(listB, listA);

        if (compareTypeID == 0) {
            Feature2D.tolerance = 0;
        } else if (compareTypeID == 1) {
            Feature2D.tolerance = threshold;
        }
        Feature2DWithMotif.lenientEqualityEnabled = false;

        Feature2DList matchesWithinToleranceUniqueToA = Feature2DTools.subtract(matchesWithinToleranceFromA, exactMatches);
        Feature2DList matchesWithinToleranceUniqueToB = Feature2DTools.subtract(matchesWithinToleranceFromB, exactMatches);
        int numMatchesWithinTolA = matchesWithinToleranceUniqueToA.getNumTotalFeatures();
        int numMatchesWithinTolB = matchesWithinToleranceUniqueToB.getNumTotalFeatures();

        System.out.println("Number of matches within tolerance: " + numMatchesWithinTolA + "(A) " + numMatchesWithinTolB + "(B)");

        Feature2DList uniqueToA = Feature2DTools.subtract(listA, matchesWithinToleranceFromA);
        Feature2DList uniqueToB = Feature2DTools.subtract(listB, matchesWithinToleranceFromB);
        int numUniqueToA = uniqueToA.getNumTotalFeatures();
        int numUniqueToB = uniqueToB.getNumTotalFeatures();

        System.out.println("Number of unique features: " + numUniqueToA + "(A) " + numUniqueToB + "(B)");

        // set parent attribute
        exactMatches.addAttributeFieldToAll(PARENT_ATTRIBUTE, "Common");
        matchesWithinToleranceUniqueToA.addAttributeFieldToAll(PARENT_ATTRIBUTE, "A");
        matchesWithinToleranceUniqueToB.addAttributeFieldToAll(PARENT_ATTRIBUTE, "B");
        uniqueToA.addAttributeFieldToAll(PARENT_ATTRIBUTE, "A*");
        uniqueToB.addAttributeFieldToAll(PARENT_ATTRIBUTE, "B*");

        // set colors
        exactMatches.setColor(AB);
        matchesWithinToleranceUniqueToA.setColor(AA);
        matchesWithinToleranceUniqueToB.setColor(BB);
        uniqueToA.setColor(AAA);
        uniqueToB.setColor(BBB);

        Feature2DList finalResults = new Feature2DList(exactMatches);
        finalResults.add(matchesWithinToleranceUniqueToA);
        finalResults.add(matchesWithinToleranceUniqueToB);
        finalResults.add(uniqueToA);
        finalResults.add(uniqueToB);

        finalResults.exportFeatureList(new File(outputPath), false, Feature2DList.ListFormat.NA);

        int percentMatch = (int) Math.round(100 * ((double) (sizeB - numUniqueToB)) / ((double) sizeB));
        if (percentMatch > 95) {
            System.out.println("Test passed");
        } else {
            System.out.println("Test failed - " + percentMatch + "% match with reference list");
        }
    }
}
