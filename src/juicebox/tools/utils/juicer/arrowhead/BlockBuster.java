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

package juicebox.tools.utils.juicer.arrowhead;

import juicebox.HiCGlobals;
import juicebox.data.HiCFileTools;
import juicebox.data.MatrixZoomData;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.Feature2DParser;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.RealMatrix;

import java.util.*;

/**
 * Created by muhammadsaadshamim on 6/3/15.
 */
public class BlockBuster {

    /**
     * should be called separately for each chromosome
     *
     * @return
     */
    public static void run(int chrIndex, String chrName, int chrLength, int resolution, int matrixWidth,
                           String outputPath, MatrixZoomData zd, ArrowheadScoreList list, ArrowheadScoreList control,
                           Feature2DList contactDomainsGenomeWide, Feature2DList contactDomainListScoresGenomeWide,
                           Feature2DList contactDomainControlScoresGenomeWide) {

        double signThreshold = 0.5;
        int increment = matrixWidth / 2;
        int maxDataLengthAtResolution = (int) Math.ceil(((double) chrLength) / resolution);

        CumulativeBlockResults results = null;
        while (results == null || (results.getCumulativeResults().size() == 0 && signThreshold > 0)) {
            signThreshold = signThreshold - 0.1; // TODO error? results in negative val run?
            results = callSubBlockbuster(zd, maxDataLengthAtResolution, (double) increment, signThreshold, matrixWidth, increment, list, control);
        }

        // high variance threshold, fewer blocks, high confidence
        CumulativeBlockResults highConfidenceResults = callSubBlockbuster(zd, maxDataLengthAtResolution,
                0.2f, 0.5f, matrixWidth, increment, new ArrowheadScoreList(), new ArrowheadScoreList());

        List<HighScore> uniqueBlocks = orderedSetDifference(results.getCumulativeResults()
                , highConfidenceResults.getCumulativeResults());
        List<HighScore> filteredUniqueBlocks = filterBlocksBySize(uniqueBlocks, 60);
        appendNonConflictingBlocks(highConfidenceResults.getCumulativeResults(), filteredUniqueBlocks);

        results.setCumulativeResults(highConfidenceResults.getCumulativeResults());
        results.mergeScores();

        // prior to this point, everything should be in terms of i,j indices in a binned matrix
        results.scaleIndicesByResolution(resolution);

        if (results.getCumulativeResults().size() > 0) {

            if (HiCGlobals.printVerboseComments) {
                System.out.println("PreResults");
                for (HighScore hs : results.getCumulativeResults()) {
                    System.out.println(hs);
                }
            }
            List<HighScore> binnedScores = binScoresByDistance(results.getCumulativeResults(), 5);
            binnedScores = binScoresByDistance(binnedScores, 10);
            Collections.sort(binnedScores, Collections.reverseOrder());

            Feature2DList blockResults = Feature2DParser.parseHighScoreList(chrIndex, chrName, resolution, binnedScores);
            Feature2DList blockResultListScores = Feature2DParser.parseArrowheadScoreList(chrIndex,
                    chrName, results.getCumulativeInternalList());
            Feature2DList blockResultControlScores = Feature2DParser.parseArrowheadScoreList(chrIndex,
                    chrName, results.getCumulativeInternalControl());

            contactDomainsGenomeWide.add(blockResults);
            contactDomainListScoresGenomeWide.add(blockResultListScores);
            contactDomainControlScoresGenomeWide.add(blockResultControlScores);
        }
        else {
            System.out.println("\nNo contact domains found for chromosome " + chrName);
        }
    }

    private static List<HighScore> binScoresByDistance(List<HighScore> results, int dist) {

        List<BinnedScore> binnedScores = new ArrayList<BinnedScore>();
        for (HighScore score : results) {
            boolean scoreNotAssigned = true;
            for (BinnedScore binnedScore : binnedScores) {
                if (binnedScore.isNear(score)) {
                    binnedScore.addScoreToBin(score);
                    scoreNotAssigned = false;
                    break;
                }
            }

            if (scoreNotAssigned) {
                binnedScores.add(new BinnedScore(score, dist));
            }
        }

        return BinnedScore.convertBinnedScoresToHighScores(binnedScores);
    }

    private static void appendNonConflictingBlocks(List<HighScore> mainList, List<HighScore> possibleAdditions) {

        Map<Integer, HighScore> blockEdges = new HashMap<Integer, HighScore>();
        for (HighScore score : mainList) {
            blockEdges.put(score.getI(), score);
            blockEdges.put(score.getJ(), score);
        }

        for (HighScore score : possibleAdditions) {
            boolean doesNotConflict = true;

            for (int k = score.getI(); k <= score.getJ(); k++) {
                if (blockEdges.containsKey(k)) {
                    doesNotConflict = false;
                    break;
                }
            }

            if (doesNotConflict) {
                mainList.add(score);
                blockEdges.put(score.getI(), score);
                blockEdges.put(score.getJ(), score);
            }
        }
    }

    private static List<HighScore> filterBlocksBySize(List<HighScore> largerList, int minWidth) {
        List<HighScore> filteredList = new ArrayList<HighScore>();
        for (HighScore score : largerList) {
            if (score.getWidth() > minWidth) {
                filteredList.add(score);
            }
        }
        return filteredList;
    }

    /**
     * Set difference - returns the high scores in longerList that are not in shorterList.
     *
     * @param listA
     * @param listB
     * @return set difference
     */
    private static List<HighScore> orderedSetDifference(List<HighScore> listA, List<HighScore> listB) {
        // remove duplicates
        Set<HighScore> setA = new HashSet<HighScore>(listA);
        Set<HighScore> setB = new HashSet<HighScore>(listB);

        List<HighScore> diffSet = new ArrayList<HighScore>();
        for (HighScore score : setA) {
            if (!setB.contains(score)) {
                diffSet.add(score);
            }
        }

        return diffSet;
    }

    /**
     * Runs blockbuster for a sliding window along the diagonal of the matrix
     *
     * @param zd
     * @param chrLength
     * @param varThreshold
     * @param signThreshold
     * @param matrixWidth
     * @param increment
     * @param list
     * @param control
     * @return
     */
    private static CumulativeBlockResults callSubBlockbuster(MatrixZoomData zd, int chrLength, double varThreshold,
                                                             double signThreshold, int matrixWidth, int increment,
                                                             ArrowheadScoreList list, ArrowheadScoreList control) {

        CumulativeBlockResults cumulativeBlockResults = new CumulativeBlockResults();
        if (HiCGlobals.printVerboseComments)
            System.out.println("Loading incr " + increment + " chrLength " + chrLength);
        for (int limStart = 0; limStart < chrLength; limStart += increment) {
            int limEnd = Math.min(limStart + matrixWidth, chrLength);
            if (HiCGlobals.printVerboseComments)
                System.out.println("Reading " + limStart + ":" + limEnd);

            list.setActiveListElements(limStart, limEnd);
            control.setActiveListElements(limStart, limEnd);

            // TODO how did limStart > limEnd not cause error?
            int n = limEnd - limStart + 1;
            RealMatrix observed = HiCFileTools.extractLocalBoundedRegion(zd, limStart, limEnd,
                    limStart, limEnd, n, n, NormalizationType.KR);

            BlockResults results = new BlockResults(MatrixTools.fillLowerLeftTriangle(observed), varThreshold,
                    signThreshold, increment, list, control);
            if (HiCGlobals.printVerboseComments)
                System.out.println("Found " + results.getResults().size() + " blocks");
            results.offsetResultsIndex(limStart); // +1? because genome index should start at 1 not 0?

            cumulativeBlockResults.add(results);
            System.out.print(".");
        }
        System.out.println(".");
        return cumulativeBlockResults;
    }
}
