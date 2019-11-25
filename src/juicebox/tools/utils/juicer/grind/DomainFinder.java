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

import juicebox.data.ChromosomeHandler;
import juicebox.data.HiCFileTools;
import juicebox.data.MatrixZoomData;
import juicebox.mapcolorui.Feature2DHandler;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.tools.utils.common.UNIXTools;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.FeatureFunction;
import org.apache.commons.math.linear.RealMatrix;
import org.broad.igv.feature.Chromosome;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.util.List;

public class DomainFinder extends RegionFinder {

    public DomainFinder(ParameterConfigurationContainer container) {
        super(container);
    }

    @Override
    public void makeExamples() {

        UNIXTools.makeDir(originalPath);

        final ChromosomeHandler chromosomeHandler = ds.getChromosomeHandler();
        final int resolution = (int) resolutions.toArray()[0];

        try {
            final Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(originalPath + "all_file_names.txt"), StandardCharsets.UTF_8));

            final Feature2DHandler feature2DHandler = new Feature2DHandler(inputFeature2DList);

            inputFeature2DList.processLists(new FeatureFunction() {
                @Override
                public void process(String chr, List<Feature2D> feature2DList) {

                    Chromosome chromosome = chromosomeHandler.getChromosomeFromName(chr);
                    final MatrixZoomData zd = HiCFileTools.getMatrixZoomData(ds, chromosome, chromosome, resolution);
                    if (zd == null) return;

                    for (int rowIndex = 0; rowIndex < chromosome.getLength() / resolution; rowIndex++) {
                        for (int colIndex = rowIndex; colIndex < chromosome.getLength() / resolution; colIndex++) {

                            // todo is far from diagonal, continue

                            try {
                                RealMatrix localizedRegionData = HiCFileTools.extractLocalBoundedRegion(zd,
                                        rowIndex, rowIndex + x, colIndex, colIndex + y, x, y, norm, true);

                                if (MatrixTools.sum(localizedRegionData.getData()) > 0) {

                                    net.sf.jsi.Rectangle currentWindow = new net.sf.jsi.Rectangle(rowIndex * resolution,
                                            y * resolution, (rowIndex + x) * resolution, (colIndex + y) * resolution);

                                    List<Feature2D> inputListFoundFeatures = feature2DHandler.getIntersectingFeatures(chromosome.getIndex(), chromosome.getIndex(),
                                            currentWindow, true);

                                    double[][] labelsMatrix = new double[x][y];

                                    for (Feature2D feature2D : inputListFoundFeatures) {
                                        labelsMatrix[feature2D.getStart1() / resolution - rowIndex][feature2D.getEnd2() / resolution - colIndex] = 1.0;
                                    }

                                    String exactFileName = chromosome.getName() + "_" + rowIndex + "_" + colIndex;
                                    String exactLabelFileName = chromosome.getName() + "_" + rowIndex + "_" + colIndex + "_label";

                                    GrindUtils.saveGrindMatrixDataToFile(exactFileName, originalPath, localizedRegionData, writer, useTxtInsteadOfNPY);
                                    GrindUtils.saveGrindMatrixDataToFile(exactLabelFileName, originalPath, labelsMatrix, writer, useTxtInsteadOfNPY);

                                }
                            } catch (Exception ignored) {
                            }
                        }
                    }
                }
            });
            writer.close();
        } catch (Exception ignored) {
        }
    }
}
