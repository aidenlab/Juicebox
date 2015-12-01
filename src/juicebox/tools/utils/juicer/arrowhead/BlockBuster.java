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

import java.io.IOException;
import java.util.*;

/**
 * Created by muhammadsaadshamim on 6/3/15.
 */
public class BlockBuster {

    /**
     * Actual Arrowhead algorithm - should be called separately for each chromosome
     *
     * @return contact domain list and scores for given list/control
     */
    public static void run(int chrIndex, String chrName, int chrLength, int resolution, int matrixWidth, MatrixZoomData zd,
                           NormalizationType norm,
                           ArrowheadScoreList list, ArrowheadScoreList control,
                           Feature2DList contactDomainsGenomeWide, Feature2DList contactDomainListScoresGenomeWide,
                           Feature2DList contactDomainControlScoresGenomeWide) {

        // used for sliding window across diagonal
        int increment = matrixWidth / 2;
        int maxDataLengthAtResolution = (int) Math.ceil(((double) chrLength) / resolution);

        try {
            // get large number of blocks (lower confidence)
            CumulativeBlockResults results = null;
            for (double signThreshold = 0.4; signThreshold >= 0; signThreshold -= 0.1) {
                results = callSubBlockbuster(zd, maxDataLengthAtResolution, Double.NaN, signThreshold, matrixWidth,
                        increment, list, control, norm, resolution);
                if (results.getCumulativeResults().size() > 0) {
                    break;
                }
            }

            // high variance threshold, fewer blocks, high confidence
            CumulativeBlockResults highConfidenceResults = callSubBlockbuster(zd, maxDataLengthAtResolution,
                    0.2f, 0.5f, matrixWidth, increment, new ArrowheadScoreList(resolution),
                    new ArrowheadScoreList(resolution), norm, resolution);

            List<HighScore> uniqueBlocks = orderedSetDifference(results.getCumulativeResults(),
                    highConfidenceResults.getCumulativeResults());

            // remove the blocks that are small
            List<HighScore> filteredUniqueBlocks = filterBlocksBySize(uniqueBlocks, 60);
            appendNonConflictingBlocks(highConfidenceResults.getCumulativeResults(), filteredUniqueBlocks);

            // merge the high/low confidence results
            results.setCumulativeResults(highConfidenceResults.getCumulativeResults());
            results.mergeScores();

            // prior to this point, everything should be in terms of i,j indices in a binned matrix
            results.scaleIndicesByResolution(resolution);

            // if any contact domains are found
            if (results.getCumulativeResults().size() > 0) {
                if (HiCGlobals.printVerboseComments) {
                    System.out.println("Initial # of contact domains: " + results.getCumulativeResults().size());
                }

                // merge/bin domains in very close proximity
                List<HighScore> binnedScores = binScoresByDistance(results.getCumulativeResults(), 5 * resolution);
                binnedScores = binScoresByDistance(binnedScores, 10 * resolution);
                Collections.sort(binnedScores, Collections.reverseOrder());

                // convert to Feature2DList format
                Feature2DList blockResults = Feature2DParser.parseHighScoreList(chrIndex, chrName, resolution, binnedScores);
                Feature2DList blockResultListScores = Feature2DParser.parseArrowheadScoreList(chrIndex, chrName, results.getCumulativeInternalList());
                Feature2DList blockResultControlScores = Feature2DParser.parseArrowheadScoreList(chrIndex, chrName, results.getCumulativeInternalControl());

                // add results to genome-wide accumulator
                contactDomainsGenomeWide.add(blockResults);
                contactDomainListScoresGenomeWide.add(blockResultListScores);
                contactDomainControlScoresGenomeWide.add(blockResultControlScores);
            } else {
                if (HiCGlobals.printVerboseComments) {
                    System.out.println("No contact domains found for chromosome " + chrName);
                }
            }
        } catch (IOException e) {
            System.err.println("Data not available for this chromosome.");
        }
    }

    /**
     * Runs blockbuster for a sliding window along the diagonal of the matrix
     *
     * @param zd            - zoomData from hic file
     * @param chrLength
     * @param varThreshold
     * @param signThreshold
     * @param matrixWidth
     * @param increment
     * @param list
     * @param control
     * @return contact domain results for given thresholds and parameters
     */
    private static CumulativeBlockResults callSubBlockbuster(MatrixZoomData zd, int chrLength, double varThreshold,
                                                             double signThreshold, int matrixWidth, int increment,
                                                             ArrowheadScoreList list, ArrowheadScoreList control,
                                                             NormalizationType norm, int resolution) throws IOException {

        // container for results
        CumulativeBlockResults cumulativeBlockResults = new CumulativeBlockResults(resolution);
        if (HiCGlobals.printVerboseComments) {
            System.out.println("Loading incr " + increment + " chrLength " + chrLength);
        }

        // slide across chromosome diagonal
        for (int limStart = 0; limStart < chrLength; limStart += increment) {
            // appropriate boundaries of window
            int adjustedLimStart = limStart;
            int limEnd = Math.min(limStart + matrixWidth, chrLength);
            if (limEnd == chrLength) {
                if (chrLength > increment) {
                    adjustedLimStart = limEnd - matrixWidth;
                }
            }
            if (HiCGlobals.printVerboseComments) {
                System.out.println("Reading " + limStart + ":" + limEnd);
            }

            // get data for window from hic file
            int n = limEnd - adjustedLimStart + 1;
            RealMatrix observed = HiCFileTools.extractLocalBoundedRegion(zd, limStart, limEnd, n, norm);
            observed = MatrixTools.fillLowerLeftTriangle(observed);

            // get contact domains in window
            BlockResults results = new BlockResults(observed, varThreshold, signThreshold, list, control,
                    adjustedLimStart, limEnd);

            if (HiCGlobals.printVerboseComments) {
                System.out.println("Found " + results.getResults().size() + " blocks");
            }

            // accumulate results across the windows
            results.offsetResultsIndex(limStart); // +1? because genome index should start at 1 not 0?
            cumulativeBlockResults.add(results);
            if (HiCGlobals.printVerboseComments) {
                System.out.print(".");
            }
        }
        if (HiCGlobals.printVerboseComments) {
            System.out.println(".");
        }
        return cumulativeBlockResults;
    }

    /**
     * @param scores
     * @param dist
     * @return list of scores binned within distance
     */
    private static List<HighScore> binScoresByDistance(List<HighScore> scores, int dist) {
        List<BinnedScore> binnedScores = new ArrayList<BinnedScore>();
        for (HighScore score : scores) {
            boolean scoreNotBinned = true;
            for (BinnedScore binnedScore : binnedScores) {
                if (binnedScore.isNear(score, dist)) {
                    binnedScore.addScoreToBin(score);
                    scoreNotBinned = false;
                    break;
                }
            }

            if (scoreNotBinned) {
                binnedScores.add(new BinnedScore(score));
            }
        }

        return BinnedScore.convertBinnedScoresToHighScores(binnedScores);
    }

    /**
     * Check possibleAdditions for domains which do not overlap with the mainList
     * and append them
     *
     * @param mainList
     * @param possibleAdditions
     */
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

    /**
     * @param blockList
     * @param minWidth
     * @return blockList with small blocks (i.e. < minWidth) removed
     */
    private static List<HighScore> filterBlocksBySize(List<HighScore> blockList, int minWidth) {
        List<HighScore> filteredList = new ArrayList<HighScore>();
        for (HighScore score : blockList) {
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
}
