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

package juicebox.tools.utils.juicer.apa;

import juicebox.HiCGlobals;
import juicebox.data.HiCFileTools;
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
    private static RealMatrix gwAPAMatrix;
    private static RealMatrix gwNormedAPAMatrix;
    private static RealMatrix gwCenterNormedAPAMatrix;
    private static RealMatrix gwRankAPAMatrix;
    private static List<Double> gwEnhancement;

    // saving data variables
    private static int[] axesRange;
    private static File dataDirectory;

    // chr variables
    private final List<Double> enhancement;
    private RealMatrix APAMatrix;
    private RealMatrix normedAPAMatrix;
    private RealMatrix centerNormedAPAMatrix;
    private RealMatrix rankAPAMatrix;

    /**
     * class for saving data from chromosme wide run of APA, keeps static class to store genomide data
     *
     * @param n                width of matrix
     * @param outputFolder location for saving data
     * @param customPrefix     optional file/folder prefix
     */
    public APADataStack(int n, File outputFolder, String customPrefix) {
        APAMatrix = MatrixTools.cleanArray2DMatrix(n, n);
        normedAPAMatrix = MatrixTools.cleanArray2DMatrix(n, n);
        centerNormedAPAMatrix = MatrixTools.cleanArray2DMatrix(n, n);
        rankAPAMatrix = MatrixTools.cleanArray2DMatrix(n, n);
        enhancement = new ArrayList<Double>();

        initializeGenomeWideVariables(n);
        initializeDataSaveFolder(outputFolder, customPrefix);
        axesRange = new int[]{-n / 2, 1, -n / 2, 1};
    }

    /**
     * Ensure that directory for saving exists
     *
     * @param outputFolderDirectory   to directory
     * @param prefix of files to be saved
     */
    private static void initializeDataSaveFolder(File outputFolderDirectory, String prefix) {
        if (prefix.length() < 1) {// no preference specified
            dataDirectory = new File(outputFolderDirectory,
                    new SimpleDateFormat("yyyy.MM.dd.HH.mm").format(new Date()));
        } else {
            dataDirectory = new File(outputFolderDirectory, prefix);
        }
        dataDirectory = HiCFileTools.createValidDirectory(dataDirectory.getAbsolutePath());
    }

    private static void initializeGenomeWideVariables(int n) {
        if (genomeWideVariablesNotSet) {
            gwAPAMatrix = MatrixTools.cleanArray2DMatrix(n, n);
            gwNormedAPAMatrix = MatrixTools.cleanArray2DMatrix(n, n);
            gwCenterNormedAPAMatrix = MatrixTools.cleanArray2DMatrix(n, n);
            gwRankAPAMatrix = MatrixTools.cleanArray2DMatrix(n, n);
            //gwCoverage = APAUtils.cleanArray2DMatrix(n, n);
            gwEnhancement = new ArrayList<Double>();
            genomeWideVariablesNotSet = false;
        }
    }

    public static void exportGenomeWideData(Integer[] peakNumbers, int currentRegionWidth) {
        double gwNPeaksUsedInv = 1. / peakNumbers[0];
        gwNormedAPAMatrix = gwNormedAPAMatrix.scalarMultiply(gwNPeaksUsedInv);
        gwCenterNormedAPAMatrix = gwCenterNormedAPAMatrix.scalarMultiply(gwNPeaksUsedInv);
        gwRankAPAMatrix = gwRankAPAMatrix.scalarMultiply(gwNPeaksUsedInv);

        RealMatrix[] matrices = {gwAPAMatrix, gwNormedAPAMatrix, gwCenterNormedAPAMatrix, gwRankAPAMatrix};
        String[] titles = {"APA", "normedAPA", "centerNormedAPA", "rankAPA", "enhancement", "measures"};

        saveDataSet("gw", matrices, titles, gwEnhancement, peakNumbers, currentRegionWidth);
    }

    private static void saveDataSet(String prefix,
                                    RealMatrix[] apaMatrices,
                                    String[] apaDataTitles,
                                    List<Double> givenEnhancement,
                                    Integer[] peakNumbers, int currentRegionWidth) {

        File subFolder = HiCFileTools.createValidDirectory(new File(dataDirectory, prefix).getAbsolutePath());
        if (HiCGlobals.printVerboseComments) {
            System.out.println("Saving chr " + prefix + " data to " + subFolder);
        }

        for (int i = 0; i < apaMatrices.length; i++) {

            String title = "N=" + peakNumbers[0] + " (filtered) " + peakNumbers[1] + " (unique) " +
                    peakNumbers[2] + " (total)";
            APARegionStatistics apaStats = new APARegionStatistics(apaMatrices[i], currentRegionWidth);
            int dimension = apaMatrices[i].getColumnDimension();
            int midPoint = dimension / 2;
            double centralVal = apaMatrices[i].getEntry(midPoint, midPoint);
            double colorMax = 5 * centralVal / apaStats.getPeak2UR();
            double colorMin = 0;
            APAPlotter.plot(apaMatrices[i],
                    axesRange,
                    new File(subFolder, apaDataTitles[i] + ".png"),
                    title, currentRegionWidth, apaDataTitles[i].equals("APA"), colorMin, colorMax);
            MatrixTools.saveMatrixText((new File(subFolder, apaDataTitles[i] + ".txt")).getAbsolutePath(),
                    apaMatrices[i]);
        }

        APAUtils.saveListText((new File(subFolder, apaDataTitles[4] + ".txt")).getAbsolutePath(),
                givenEnhancement);
        APAUtils.saveMeasures((new File(subFolder, apaDataTitles[5] + ".txt")).getAbsolutePath(),
                apaMatrices[0], currentRegionWidth);
    }

    public static void clearAllData() {
        axesRange = null;
        dataDirectory = null;
        genomeWideVariablesNotSet = true;
        gwAPAMatrix = null;
        gwNormedAPAMatrix = null;
        gwCenterNormedAPAMatrix = null;
        gwRankAPAMatrix = null;
        gwEnhancement = null;
    }

    public void addData(RealMatrix newData) {
        RealMatrix nanFilteredData = MatrixTools.cleanUpNaNs(newData);
        APAMatrix = APAMatrix.add(nanFilteredData);
        normedAPAMatrix = normedAPAMatrix.add(APAUtils.standardNormalization(nanFilteredData));
        centerNormedAPAMatrix = centerNormedAPAMatrix.add(APAUtils.centerNormalization(nanFilteredData));
        rankAPAMatrix = rankAPAMatrix.add(APAUtils.rankPercentile(nanFilteredData));
        enhancement.add(APAUtils.peakEnhancement(nanFilteredData));
    }

    public void updateGenomeWideData() {
        gwAPAMatrix = gwAPAMatrix.add(APAMatrix);
        gwNormedAPAMatrix = gwNormedAPAMatrix.add(normedAPAMatrix);
        gwCenterNormedAPAMatrix = gwCenterNormedAPAMatrix.add(centerNormedAPAMatrix);
        gwRankAPAMatrix = gwRankAPAMatrix.add(rankAPAMatrix);
        gwEnhancement.addAll(enhancement);
    }

    public void exportDataSet(String subFolderName, Integer[] peakNumbers, int currentRegionWidth) {
        double nPeaksUsedInv = 1. / peakNumbers[0];
        normedAPAMatrix = normedAPAMatrix.scalarMultiply(nPeaksUsedInv);
        centerNormedAPAMatrix = centerNormedAPAMatrix.scalarMultiply(nPeaksUsedInv);
        rankAPAMatrix = rankAPAMatrix.scalarMultiply(nPeaksUsedInv);

        RealMatrix[] matrices = {APAMatrix, normedAPAMatrix, centerNormedAPAMatrix, rankAPAMatrix};
        String[] titles = {"APA", "normedAPA", "centerNormedAPA", "rankAPA", "enhancement", "measures"};

        saveDataSet(subFolderName, matrices, titles, enhancement, peakNumbers, currentRegionWidth);
    }

    public void thresholdPlots(int val) {
        MatrixTools.thresholdValues(APAMatrix, val);
    }
}
