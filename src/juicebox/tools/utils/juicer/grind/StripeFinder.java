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

package juicebox.tools.utils.juicer.grind;

import juicebox.data.*;
import juicebox.mapcolorui.Feature2DHandler;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.RealMatrix;
import org.broad.igv.feature.Chromosome;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Set;

import static juicebox.tools.utils.juicer.grind.SectionParser.saveMatrixText2;

public class StripeFinder implements RegionFinder {

    private Integer x;
    private Integer y;
    private Integer z;
    private Dataset ds;
    private Feature2DList features;
    private String path;
    private Set<String> givenChromosomes;
    private NormalizationType norm;
    private boolean useObservedOverExpected;
    private boolean useDenseLabels;
    private Set<Integer> resolutions;

    public StripeFinder(int x, int y, int z, Dataset ds, Feature2DList features, File outputDirectory, Set<String> givenChromosomes, NormalizationType norm,
                        boolean useObservedOverExpected, boolean useDenseLabels, Set<Integer> resolutions) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.ds = ds;
        this.features = features;
        this.path = outputDirectory.getPath();
        this.givenChromosomes = givenChromosomes;
        this.norm = norm;
        this.useObservedOverExpected = useObservedOverExpected;
        this.useDenseLabels = useDenseLabels;
        this.resolutions = resolutions;
    }

    private void makeDir(String path) {
        File file = new File(path);
        if (!file.isDirectory()) {
            file.mkdir();
        }
    }

    @Override
    public void makePositiveExamples() {

        final String negPath = path + "/negative";
        final String posPath = path + "/positive";
        makeDir(negPath);
        makeDir(posPath);


        final ChromosomeHandler chromosomeHandler = ds.getChromosomeHandler();

        try {

            final Writer posWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path + "/pos_file_names.txt"), StandardCharsets.UTF_8));
            final Writer negWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path + "/neg_file_names.txt"), StandardCharsets.UTF_8));
            final Writer posLabelWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path + "/pos_label_file_names.txt"), StandardCharsets.UTF_8));

            final Feature2DHandler feature2DHandler = new Feature2DHandler(features);

            int totalCounterForExamples = 0;
            for (int res : resolutions) {
                for (String chrom_name : givenChromosomes) {

                    System.out.println("Currently on: " + chrom_name);
                    Chromosome chrom = chromosomeHandler.getChromosomeFromName(chrom_name);
                    Matrix matrix = ds.getMatrix(chrom, chrom);
                    if (matrix == null) continue;

                    HiCZoom zoom = ds.getZoomForBPResolution(res);
                    final MatrixZoomData zd = matrix.getZoomData(zoom);
                    if (zd == null) continue;

                    //
                    //double[][] labelsBinary = new double[z][1];


                    // sliding along the diagonal
                    for (int rowIndex = 0; rowIndex < chrom.getLength() / res; rowIndex += 4) {
//                                for (int colIndex = rowIndex; colIndex < chrom.getLength() / res; colIndex+=50) {
                        int colIndex = rowIndex;

                        //System.out.println("rowIndex" + rowIndex + "colIndex" + colIndex);

                        if (totalCounterForExamples >= z) {
                            System.out.println("done iterating because got all z as requested");
                            return;
                        }

                        // if far from diagonal break (continue?)
                        if (Math.abs(rowIndex - colIndex) > 300) break;

                        try {

                            // make the rectangle
                            RealMatrix localizedRegionData = HiCFileTools.extractLocalBoundedRegion(zd,
                                    rowIndex, rowIndex + x, colIndex, colIndex + y, x, y, norm);

                            //System.out.println("sum of numbers in matrix: " + MatrixTools.sum(localizedRegionData.getData()));

                            if (MatrixTools.sum(localizedRegionData.getData()) > 0) {

                                net.sf.jsi.Rectangle currentWindow = new net.sf.jsi.Rectangle(rowIndex * res,
                                        y * res, (rowIndex + x) * res, (colIndex + y) * res);

                                List<Feature2D> inputListFoundFeatures = feature2DHandler.getContainedFeatures(chrom.getIndex(), chrom.getIndex(),
                                        currentWindow);
                                //System.out.println("size of input list found features: " + inputListFoundFeatures.size());

                                // if positive
                                if (inputListFoundFeatures.size() > 0) {
                                    double[][] labelsMatrix = new double[x][y];
                                    for (Feature2D feature2D : inputListFoundFeatures) {
                                        int rowlen = feature2D.getEnd1() - feature2D.getStart1();
                                        int collen = feature2D.getEnd2() - feature2D.getStart2();
                                        rowlen = rowlen / res;
                                        collen = collen / res;


                                        //System.out.println("row length: " + rowlen + "  col length: " + collen);

                                        int startRowOf1 = feature2D.getStart1() / res - rowIndex;
                                        int starColOf1 = feature2D.getEnd2() / res - colIndex;

                                        for (int i = 0; i < rowlen + 1; i++) {          //I'm not sure about the +1
                                            for (int j = 0; j < collen + 1; j++) {
                                                labelsMatrix[startRowOf1 + i][starColOf1 + j] = 1.0;
                                            }
                                        }

                                        totalCounterForExamples += 1;
                                        //System.out.println("index positive" + totalCounterForExamples);
                                    }


                                    String exactFileName = chrom.getName() + "pos_" + "_" + rowIndex + "_" + colIndex + "_matrix.txt";
                                    //System.out.println("saved " + exactFileName);
                                    saveMatrixText2(posPath + "/" + exactFileName, localizedRegionData);
                                    posWriter.write(exactFileName + "\n");

                                    String exactPosMatrixLabelFileName = chrom.getName() + "pos" + "_" + rowIndex + "_" + colIndex + "_matrix.label.txt";
                                    //System.out.println("saved " + exactPosMatrixLabelFileName);
                                    saveMatrixText2(posPath + "/" + exactPosMatrixLabelFileName, labelsMatrix);
                                    posLabelWriter.write(exactPosMatrixLabelFileName + "\n");

                                }
                                //else if negative
                                else {

                                    String exactFileName = chrom.getName() + "neg_" + "_" + rowIndex + "_" + colIndex + "_matrix.txt";
                                    //System.out.println("saved " + exactFileName);
                                    saveMatrixText2(negPath + "/" + exactFileName, localizedRegionData);
                                    negWriter.write(exactFileName + "\n");
                                }
                                System.out.print(".");
                            }
                        } catch (Exception e) {
                        }
                    }
                }
            }
            posWriter.close();
            negWriter.close();
            posLabelWriter.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public void makeSquaresForTrainingModelToLocalize() {

        final String negPath = path + "/negative";
        final String posPath = path + "/positive";
        makeDir(negPath);
        makeDir(posPath);


        final ChromosomeHandler chromosomeHandler = ds.getChromosomeHandler();

        try {

            final Writer posWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path + "/pos_file_names.txt"), StandardCharsets.UTF_8));
            final Writer negWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path + "/neg_file_names.txt"), StandardCharsets.UTF_8));
            final Writer posLabelWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path + "/pos_label_file_names.txt"), StandardCharsets.UTF_8));

            final Feature2DHandler feature2DHandler = new Feature2DHandler(features);

            int totalCounterForPosExamples = 0;

            for (int res : resolutions) {
                for (String chrom_name : givenChromosomes) {

                    System.out.println("Currently on: " + chrom_name);
                    Chromosome chrom = chromosomeHandler.getChromosomeFromName(chrom_name);
                    Matrix matrix = ds.getMatrix(chrom, chrom);
                    if (matrix == null) continue;

                    HiCZoom zoom = ds.getZoomForBPResolution(res);
                    final MatrixZoomData zd = matrix.getZoomData(zoom);
                    if (zd == null) continue;


                    //double[][] labelsBinary = new double[z][1];


                    // sliding along the diagonal
                    for (int rowIndex = 0; rowIndex < chrom.getLength() / res; rowIndex += 4) {
                        for (int colIndex = rowIndex; colIndex < chrom.getLength() / res; colIndex += y / 10) {

                            //System.out.println("rowIndex" + rowIndex + "colIndex" + colIndex);

                            if (totalCounterForPosExamples >= z) {
                                System.out.println("done iterating because got all z as requested");
                                return;
                            }

                            // if far from diagonal break (continue?)
                            if (Math.abs(rowIndex - colIndex) > 900) break;

                            try {

                                // make the square
                                RealMatrix localizedRegionData = HiCFileTools.extractLocalBoundedRegion(zd,
                                        rowIndex, rowIndex + x, colIndex, colIndex + y, x, y, norm);

                                //System.out.println("sum of numbers in matrix: " + MatrixTools.sum(localizedRegionData.getData()));

                                if (MatrixTools.sum(localizedRegionData.getData()) > 0) {

                                    net.sf.jsi.Rectangle currentWindow = new net.sf.jsi.Rectangle(rowIndex * res,
                                            y * res, (rowIndex + x) * res, (colIndex + y) * res);

                                    List<Feature2D> inputListFoundFeatures = feature2DHandler.getContainedFeatures(chrom.getIndex(), chrom.getIndex(),
                                            currentWindow);
                                    //System.out.println("size of input list found features: " + inputListFoundFeatures.size());

                                    // if positive
                                    if (inputListFoundFeatures.size() > 0) {
                                        double[][] labelsMatrix = new double[x][y];
                                        for (Feature2D feature2D : inputListFoundFeatures) {
                                            int rowlen = feature2D.getEnd1() - feature2D.getStart1();
                                            int collen = feature2D.getEnd2() - feature2D.getStart2();
                                            rowlen = rowlen / res;
                                            collen = collen / res;


                                            //System.out.println("row length: " + rowlen + "  col length: " + collen);

                                            int startRowOf1 = feature2D.getStart1() / res - rowIndex;
                                            int starColOf1 = feature2D.getEnd2() / res - colIndex;

                                            for (int i = 0; i < rowlen + 1; i++) {          //I'm not sure about the +1
                                                for (int j = 0; j < collen + 1; j++) {
                                                    labelsMatrix[startRowOf1 + i][starColOf1 + j] = 1.0;
                                                }
                                            }

                                            totalCounterForPosExamples += 1;
                                            //System.out.println("index positive" + totalCounterForExamples);
                                        }


                                        String exactFileName = chrom.getName() + "pos_" + "_" + rowIndex + "_" + colIndex + "_matrix.txt";
                                        //System.out.println("saved " + exactFileName);
                                        saveMatrixText2(posPath + "/" + exactFileName, localizedRegionData);
                                        posWriter.write(exactFileName + "\n");

                                        String exactPosMatrixLabelFileName = chrom.getName() + "pos" + "_" + rowIndex + "_" + colIndex + "_matrix.label.txt";
                                        //System.out.println("saved " + exactPosMatrixLabelFileName);
                                        saveMatrixText2(posPath + "/" + exactPosMatrixLabelFileName, labelsMatrix);
                                        posLabelWriter.write(exactPosMatrixLabelFileName + "\n");

                                    }
                                    //else if negative
                                    else {

                                        String exactFileName = chrom.getName() + "neg_" + "_" + rowIndex + "_" + colIndex + "_matrix.txt";
                                        //System.out.println("saved " + exactFileName);
                                        saveMatrixText2(negPath + "/" + exactFileName, localizedRegionData);
                                        negWriter.write(exactFileName + "\n");
                                    }
                                    System.out.print(".");
                                }
                            } catch (Exception e) {
                            }
                        }
                    }
                }
            }
            posWriter.close();
            negWriter.close();
            posLabelWriter.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    @Override
    public void makeNegativeExamples() {

    }
}
