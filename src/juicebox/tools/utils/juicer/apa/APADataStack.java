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

package juicebox.tools.utils.juicer.apa;

import juicebox.HiCGlobals;
import juicebox.tools.utils.common.MatrixTools;
import org.apache.commons.math.linear.RealMatrix;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

/**
 * Created by muhammadsaadshamim on 5/1/15.
 */
public class APADataStack {

    // genome wide variables
    private static boolean genomeWideVariablesNotSet = true;
    private static RealMatrix gwPsea;
    private static RealMatrix gwNormedPsea;
    private static RealMatrix gwCenterNormedPsea;
    private static RealMatrix gwRankPsea;
    private static List<Double> gwEnhancement;

    // saving data variables
    private static int[] axesRange;
    private static File dataDirectory;

    // chr variables
    private final List<Double> enhancement;
    private RealMatrix psea;
    private RealMatrix normedPsea;
    private RealMatrix centerNormedPsea;
    private RealMatrix rankPsea;

    /**
     * class for saving data from chromosme wide run of APA, keeps static class to store genomide data
     *
     * @param n                width of matrix
     * @param outputFolderPath location for saving data
     * @param customPrefix     optional file/folder prefix
     */
    public APADataStack(int n, String outputFolderPath, String customPrefix) {
        psea = MatrixTools.cleanArray2DMatrix(n, n);
        normedPsea = MatrixTools.cleanArray2DMatrix(n, n);
        centerNormedPsea = MatrixTools.cleanArray2DMatrix(n, n);
        rankPsea = MatrixTools.cleanArray2DMatrix(n, n);
        enhancement = new ArrayList<Double>();

        initializeGenomeWideVariables(n);
        initializeDataSaveFolder(outputFolderPath, customPrefix);
        axesRange = new int[]{-n / 2, 1, -n / 2, 1};
    }

    /**
     * Ensure that directory for saving exists
     *
     * @param path   to directory
     * @param prefix of files to be saved
     */
    private static void initializeDataSaveFolder(String path, String prefix) {
        File newDirectory = safeFolderCreation(path);
        if (prefix.length() < 1) {// no preference specied
            dataDirectory = safeFolderCreation(newDirectory.getAbsolutePath() + "/" +
                    new SimpleDateFormat("yyyy.MM.dd.HH.mm").format(new Date()));
        } else {
            dataDirectory = safeFolderCreation(newDirectory.getAbsolutePath() + "/" + prefix);
        }
    }


    private static File safeFolderCreation(String path) {
        File newFolder = new File(path);
        if (!newFolder.exists()) {
            boolean result = newFolder.mkdir();
            if (!result) {
                System.err.println("Error creating directory (data not saved): " + newFolder);
                return null;
            }
        }
        return newFolder;
    }

    private static void initializeGenomeWideVariables(int n) {
        if (genomeWideVariablesNotSet) {
            gwPsea = MatrixTools.cleanArray2DMatrix(n, n);
            gwNormedPsea = MatrixTools.cleanArray2DMatrix(n, n);
            gwCenterNormedPsea = MatrixTools.cleanArray2DMatrix(n, n);
            gwRankPsea = MatrixTools.cleanArray2DMatrix(n, n);
            //gwCoverage = APAUtils.cleanArray2DMatrix(n, n);
            gwEnhancement = new ArrayList<Double>();
            genomeWideVariablesNotSet = false;
        }
    }

    public static void exportGenomeWideData(Integer[] peakNumbers) {
        double gwNPeaksUsedInv = 1. / peakNumbers[0];
        gwNormedPsea = gwNormedPsea.scalarMultiply(gwNPeaksUsedInv);
        gwCenterNormedPsea = gwCenterNormedPsea.scalarMultiply(gwNPeaksUsedInv);
        gwRankPsea = gwRankPsea.scalarMultiply(gwNPeaksUsedInv);

        RealMatrix[] matrices = {gwPsea, gwNormedPsea, gwCenterNormedPsea, gwRankPsea};
        String[] titles = {"psea", "normedPsea", "centerNormedPsea", "rankPsea", "enhancement", "measures"};

        saveDataSet("gw", matrices, titles, gwEnhancement, peakNumbers);
    }

    private static void saveDataSet(String prefix,
                                    RealMatrix[] apaMatrices,
                                    String[] apaDataTitles,
                                    List<Double> givenEnhancement,
                                    Integer[] peakNumbers) {

        File subFolder = safeFolderCreation(dataDirectory.getAbsolutePath() + "/" + prefix);
        if (HiCGlobals.printVerboseComments) {
            System.out.println("Saving chr " + prefix + " data to " + subFolder);
        }
        String dataPath = subFolder + "/";

        for (int i = 0; i < apaMatrices.length; i++) {

            String title = "N=" + peakNumbers[0] + " (filtered) " + peakNumbers[1] + " (unique) " +
                    peakNumbers[2] + " (total)";
            APAPlotter.plot(apaMatrices[i],
                    axesRange,
                    new File(dataPath + apaDataTitles[i] + ".png"),
                    title);
            MatrixTools.saveMatrixText(dataPath + apaDataTitles[i] + ".txt", apaMatrices[i]);
        }

        APAUtils.saveListText(dataPath + apaDataTitles[4] + ".txt", givenEnhancement);
        APAUtils.saveMeasures(dataPath + apaDataTitles[5] + ".txt", apaMatrices[0]);
    }

    public static void clearAllData() {
        axesRange = null;
        dataDirectory = null;
        genomeWideVariablesNotSet = true;
        gwPsea = null;
        gwNormedPsea = null;
        gwCenterNormedPsea = null;
        gwRankPsea = null;
        gwEnhancement = null;
    }

    public void addData(RealMatrix newData) {
        psea = psea.add(newData);
        normedPsea = normedPsea.add(APAUtils.standardNormalization(newData));
        centerNormedPsea = centerNormedPsea.add(APAUtils.centerNormalization(newData));
        rankPsea = rankPsea.add(APAUtils.rankPercentile(newData));
        enhancement.add(APAUtils.peakEnhancement(newData));
    }

    public void updateGenomeWideData() {
        gwPsea = gwPsea.add(psea);
        gwNormedPsea = gwNormedPsea.add(normedPsea);
        gwCenterNormedPsea = gwCenterNormedPsea.add(centerNormedPsea);
        gwRankPsea = gwRankPsea.add(rankPsea);
        gwEnhancement.addAll(enhancement);
    }

    public void exportDataSet(String subFolderName, Integer[] peakNumbers) {
        double nPeaksUsedInv = 1. / peakNumbers[0];
        normedPsea = normedPsea.scalarMultiply(nPeaksUsedInv);
        centerNormedPsea = centerNormedPsea.scalarMultiply(nPeaksUsedInv);
        rankPsea = rankPsea.scalarMultiply(nPeaksUsedInv);

        RealMatrix[] matrices = {psea, normedPsea, centerNormedPsea, rankPsea};
        String[] titles = {"psea", "normedPsea", "centerNormedPsea", "rankPsea", "enhancement", "measures"};

        saveDataSet(subFolderName, matrices, titles, enhancement, peakNumbers);
    }

    public void thresholdPlots(int val) {
        MatrixTools.thresholdValues(psea, val);
    }
}
