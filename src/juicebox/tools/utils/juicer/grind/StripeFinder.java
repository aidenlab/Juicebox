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
import juicebox.track.feature.FeatureFunction;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.RealMatrix;
import org.broad.igv.feature.Chromosome;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Random;
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
        final Random generator = new Random();

        final String negPath = path + "/negative";
        final String posPath = path + "/positive";
        makeDir(negPath);
        makeDir(posPath);


        final ChromosomeHandler chromosomeHandler = ds.getChromosomeHandler();

        try {
            final Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path + "/all_file_names.txt"), StandardCharsets.UTF_8));

            final Feature2DHandler feature2DHandler = new Feature2DHandler(features);

            //Feature2DList features = Feature2DParser.loadFeatures(loopListPath, chromosomeHandler, false, null, false);

            features.processLists(new FeatureFunction() {
                @Override
                public void process(String chr, List<Feature2D> feature2DList) {
                    int index = 0;
                    for (int res : resolutions) {
                        for (String chrom_name : givenChromosomes) {

                            System.out.println("Currently on: " + chrom_name);
                            Chromosome chrom = chromosomeHandler.getChromosomeFromName(chrom_name);
                            Matrix matrix = ds.getMatrix(chrom, chrom);
                            if (matrix == null) continue;

                            HiCZoom zoom = ds.getZoomForBPResolution(res);
                            final MatrixZoomData zd = matrix.getZoomData(zoom);
                            if (zd == null) continue;

                            double[][] labelsMatrix = new double[x][y];
                            //double[][] labelsBinary = new double[z][1];


                            // sliding along the diagonal
                            for (int rowIndex = 300; rowIndex < chrom.getLength() / res; rowIndex += 4) {
//                                for (int colIndex = rowIndex; colIndex < chrom.getLength() / res; colIndex+=50) {
                                int colIndex = rowIndex;

                                System.out.println("rowIndex" + rowIndex + "colIndex" + colIndex);

                                if (index >= z) {
                                    System.out.println("broke because got all z as requested");
                                    return;
                                    }

                                // if far from diagonal break (continue?)
                                if (Math.abs(rowIndex - colIndex) > 300) break;
                                if (Math.abs(colIndex - rowIndex) > 300) break;

                                    try {

                                        // make the rectangle
                                        RealMatrix localizedRegionData = HiCFileTools.extractLocalBoundedRegion(zd,
                                                rowIndex, rowIndex + x, colIndex, colIndex + y, x, y, norm);

                                        System.out.println("sum of numbers in matrix: " + MatrixTools.sum(localizedRegionData.getData()));

                                        if (MatrixTools.sum(localizedRegionData.getData()) > 0) {

                                            net.sf.jsi.Rectangle currentWindow = new net.sf.jsi.Rectangle(rowIndex * res,
                                                    y * res, (rowIndex + x) * res, (colIndex + y) * res);

                                            List<Feature2D> inputListFoundFeatures = feature2DHandler.getContainedFeatures(chrom.getIndex(), chrom.getIndex(),
                                                    currentWindow);


                                            System.out.println("found features" + inputListFoundFeatures);
                                            System.out.println("size of input list found features" + inputListFoundFeatures.size());

                                            // if positive
                                            if (inputListFoundFeatures.size() > 0) {

                                                for (Feature2D feature2D : inputListFoundFeatures) {
                                                    int rowlen = 0;
                                                    int collen = 0;
                                                    rowlen = feature2D.getEnd1() - feature2D.getStart1();
                                                    rowlen = rowlen / res;
                                                    collen = feature2D.getEnd2() - feature2D.getStart2();
                                                    collen = collen / res;
                                                    labelsMatrix[feature2D.getStart1() / res - rowIndex][feature2D.getEnd2() / res - colIndex] = 1.0;
                                                    System.out.println("row length" + rowlen + "col length" + collen);
                                                    for (int i = 0; i < rowlen + 1; i++) {          //I'm not sure about the +1
                                                        for (int j = 0; j < collen + 1; j++) {
                                                            System.out.println("i " + (feature2D.getStart1() / res - rowIndex + i) + " j " + (feature2D.getEnd2() / res - colIndex + j));
                                                            labelsMatrix[feature2D.getStart1() / res - rowIndex + i][feature2D.getEnd2() / res - colIndex + j] = 1.0;
                                                        }
                                                    }
                                                    //double[] tempPos = new double[1];
                                                    //tempPos[0]= 1;
                                                    //labelsBinary[index] = tempPos;

                                                    index += 1;
                                                    System.out.println("index positive" + index);


                                                }


                                                String exactFileName = chrom.getName() + "pos" + "_" + rowIndex + "_" + colIndex + ".txt";
                                                System.out.println("saved " + exactFileName);
                                                saveMatrixText2(posPath + "/" + exactFileName, localizedRegionData);
                                                writer.write(exactFileName + "\n" + "1");


                                                String exactPosMatrixLabelFileName = chrom.getName() + "pos" + "_" + rowIndex + "_" + colIndex + "matrix.label.txt";
                                                System.out.println("saved " + exactPosMatrixLabelFileName);
                                                saveMatrixText2(posPath + "/" + exactPosMatrixLabelFileName, labelsMatrix);
                                                writer.write(exactPosMatrixLabelFileName + "\n" + "1");


                                            }
                                            //else if negative
                                            else {

                                                //double[] tempNeg = new double[1];
                                                //tempNeg[0]= 0;
                                                //labelsBinary[index] = tempNeg;

                                                String exactFileName = chrom.getName() + "neg" + "_" + rowIndex + "_" + colIndex + ".txt";
                                                System.out.println("saved " + exactFileName);
                                                saveMatrixText2(negPath + "/" + exactFileName, localizedRegionData);
                                                writer.write(exactFileName + "\n" + "0");


                                            }

                                        }
                                    } catch (Exception e) {
                                    }
//                                }
                            }
                            //String exactFileName = chrom.getName() + "labels.txt";
                            //System.out.println("saved" + exactFileName);
                            //saveMatrixText2(path + "/labels/" + exactFileName, labelsBinary);

                        }
                    }

                }
            });
            writer.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    @Override
    public void makeNegativeExamples() {

    }
}
