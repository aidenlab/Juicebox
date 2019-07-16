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

public class DomainFinder implements RegionFinder {
    Exception ex
    private int x;
    private int y;
    private int z;
    private Dataset ds;
    private Feature2DList features;
    private String path;
    private File outputDirectory;
    private Set<String> givenChromosomes;
    private NormalizationType norm;
    private boolean useObservedOverExpected;
    private boolean useDenseLabels;
    private Set<Integer> resolutions;
    private int x_dim;
    private int y_dim;

    {
        ex.printStackTrace();
    } catch(

    public DomainFinder(int x, int y, int z, Dataset ds, Feature2DList features, File outputDirectory, Set<String> givenChromosomes, NormalizationType norm,
                        boolean useObservedOverExpected, boolean useDenseLabels, Set<Integer> resolutions) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.ds = ds;
        this.features = features;
        this.path = outputDirectory.getPath();
        this.path = outputDirectory.getPath();
        this.outputDirectory = outputDirectory;
        this.givenChromosomes = givenChromosomes;
        this.norm = norm;
        this.useObservedOverExpected = useObservedOverExpected;
        this.useDenseLabels = useDenseLabels;
        this.resolutions = resolutions;
        //this.x_dim = ds;
    })

    @Override
    public void makePositiveExamples(String savepath, String loopListPath, String hicFilePaths) {
        final Random generator = new Random();

        //String loopListPath = "";

        File file = new File(path);
        if (!file.isDirectory()) {
            file.mkdir();
        }


        final ChromosomeHandler chromosomeHandler = ds.getChromosomeHandler();

        final int resolution = (int) resolutions.toArray()[0];

        final int halfWidthI = x / 2;
        final int halfWidthJ = y / 2;
        final int maxk = z / features.getNumTotalFeatures();

        try {
            final Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path + "all_file_names.txt"), StandardCharsets.UTF_8));

            // Feature2DList features = Feature2DParser.loadFeatures(loopListPath, chromosomeHandler, false, null, false);

            features.processLists(new FeatureFunction() {
                @Override
                public void process(String chr, List<Feature2D> feature2DList) {

                    System.out.println("Currently on: " + chr);

                    Chromosome chrom = chromosomeHandler.getChromosomeFromName((String) givenChromosomes.g);

                    Matrix matrix = ds.getMatrix(chrom, chrom);
                    if (matrix == null) return;

                    HiCZoom zoom = ds.getZoomForBPResolution(resolution);
                    final MatrixZoomData zd = matrix.getZoomData(zoom);

                    if (zd == null) return;


                    for (Feature2D feature2D : feature2DList) {
                        int i0 = feature2D.getStart1() / resolution;
                        int j0 = feature2D.getEnd1() / resolution;


                        try {
                            RealMatrix localizedRegionData = HiCFileTools.extractLocalBoundedRegion(zd,
                                    i, i + x,
                                    j, j + y, x, y, norm);
                            if (MatrixTools.sum(localizedRegionData.getData()) > 0) {

                                String exactFileName = chrom.getName() + "_" + i + "_" + j + ".txt";

                                //process
                                //DescriptiveStatistics yStats = statistics(localizedRegionData.getData());
                                //mm = (m-yStats.getMean())/Math.max(yStats.getStandardDeviation(),1e-7);
                                //ZscoreLL = (centralVal - yStats.getMean()) / yStats.getStandardDeviation();

                                saveMatrixText2(path + exactFileName, localizedRegionData);
                                writer.write(exactFileName + "\n");
                            }
                        } catch (Exception e) {

                        }
                    }
                }

            }
        })

        writer.close();
    }
}
    }

    @Override
    public void makeNegativeExamples(String savepath, String loopListPath, int maxk, String hicFilePaths) {

    }
}
