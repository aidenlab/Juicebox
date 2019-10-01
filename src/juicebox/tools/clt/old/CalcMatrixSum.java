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

package juicebox.tools.clt.old;

import jargs.gnu.CmdLineParser;
import juicebox.data.*;
import juicebox.tools.clt.JuiceboxCLT;
import juicebox.tools.utils.norm.NormalizationCalculations;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import org.broad.igv.feature.Chromosome;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class CalcMatrixSum extends JuiceboxCLT {

    private Dataset dataset;
    private File outputFile;
    private ChromosomeHandler chromosomeHandler;
    private PrintWriter printWriter;


    public CalcMatrixSum() {
        super("calcMatrixSum <input_hic_file>");
    }

    @Override
    public void readArguments(String[] args, CmdLineParser parser) {
        if (!(args.length == 2)) {
            printUsageAndExit();
        }
        String infile = args[1];

        dataset = HiCFileTools.extractDatasetForCLT(Arrays.asList(infile.split("\\+")), false);
        outputFile = new File(infile + "_matrix_sums.txt");
        chromosomeHandler = dataset.getChromosomeHandler();
        try {
            printWriter = new PrintWriter(new FileOutputStream(outputFile));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            System.exit(-5);
        }
    }

    private static String getKeyWithNorm(Chromosome chromosome, HiCZoom zoom, NormalizationType normalizationType) {
        return chromosome.getName() + "_" + zoom.getKey() + "_" + normalizationType.getLabel();
    }

    @Override
    public void run() {

        int numCPUThreads = Runtime.getRuntime().availableProcessors();
        ExecutorService executor = Executors.newFixedThreadPool(numCPUThreads);

        Map<String, Double> zoomToMatrixSumMap = new HashMap<>();

        for (NormalizationType normalizationType : dataset.getNormalizationTypes()) {
            for (Chromosome chromosome : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {
                for (HiCZoom zoom : dataset.getBpZooms()) {
                    Runnable worker = new Runnable() {
                        @Override
                        public void run() {
                            NormalizationVector normalizationVector = dataset.getNormalizationVector(chromosome.getIndex(), zoom, normalizationType);
                            double[] actualVector;
                            MatrixZoomData zd;
                            try {
                                actualVector = normalizationVector.getData();
                                zd = dataset.getMatrix(chromosome, chromosome).getZoomData(zoom);
                            } catch (Exception e) {
                                System.err.println("No data for " + normalizationType.getLabel() + " - " + chromosome + " at " + zoom);
                                return;
                            }

                            NormalizationCalculations calculations = new NormalizationCalculations(zd);
                            double matrixSum = calculations.getSumFactor(actualVector);

                            String key = getKeyWithNorm(chromosome, zoom, normalizationType);
                            zoomToMatrixSumMap.put(key, matrixSum);
                            System.out.println("Finished: " + key);
                        }
                    };
                    executor.execute(worker);
                }
            }
        }

        executor.shutdown();
        while (!executor.isTerminated()) {
        }

        for (NormalizationType normalizationType : dataset.getNormalizationTypes()) {
            printWriter.println("Normalization Type: " + normalizationType);
            for (Chromosome chromosome : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {
                printWriter.println("Chromsome: " + chromosome);
                for (HiCZoom zoom : dataset.getBpZooms()) {
                    String key = getKeyWithNorm(chromosome, zoom, normalizationType);
                    if (zoomToMatrixSumMap.containsKey(key)) {
                        printWriter.println("Zoom: " + zoom + " sum: " + zoomToMatrixSumMap.get(key));
                    }
                }
            }
        }

        printWriter.close();
    }
}