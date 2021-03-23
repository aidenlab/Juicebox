/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2021 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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
 *  FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.tools.clt.old;

import juicebox.data.*;
import juicebox.data.basics.Chromosome;
import juicebox.data.iterator.IteratorContainer;
import juicebox.tools.clt.CommandLineParser;
import juicebox.tools.clt.JuiceboxCLT;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.tools.utils.norm.NormalizationCalculations;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class CalcMatrixSum extends JuiceboxCLT {

    private File outputSummaryFile, outputNpyFile, outputTxtFile;
    private ChromosomeHandler chromosomeHandler;
    private PrintWriter printWriter;


    public CalcMatrixSum() {
        super("calcMatrixSum <normalizationType> <input_hic_file> <output_directory>");
    }

    private static String getKeyWithNorm(Chromosome chromosome, HiCZoom zoom, NormalizationType normalizationType) {
        return chromosome.getName() + "_" + zoom.getKey() + "_" + normalizationType.getLabel();
    }

    @Override
    public void readArguments(String[] args, CommandLineParser parser) {
        if (!(args.length == 4)) {
            printUsageAndExit();
        }

        setDatasetAndNorm(args[2], args[1], false);
        File outputDirectory = HiCFileTools.createValidDirectory(args[3]);

        outputSummaryFile = new File(outputDirectory, "matrix_sums_summary.txt");
        outputNpyFile = new File(outputDirectory, "matrix_sums_data.npy");
        outputTxtFile = new File(outputDirectory, "matrix_sums_data.txt");

        chromosomeHandler = dataset.getChromosomeHandler();
        try {
            printWriter = new PrintWriter(new FileOutputStream(outputSummaryFile));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            System.exit(-5);
        }
    }

    @Override
    public void run() {

        int numCPUThreads = Runtime.getRuntime().availableProcessors();
        ExecutorService executor = Executors.newFixedThreadPool(numCPUThreads);

        Map<String, Double[]> zoomToMatrixSumMap = new HashMap<>();

        for (Chromosome chromosome : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {
            for (HiCZoom zoom : dataset.getBpZooms()) {
                Runnable worker = new Runnable() {
                    @Override
                    public void run() {
                        NormalizationVector normalizationVector = dataset.getNormalizationVector(chromosome.getIndex(), zoom, norm);
                        double[] actualVector;
                        MatrixZoomData zd;
                        try {
                            actualVector = normalizationVector.getData().getValues().get(0);
                            zd = HiCFileTools.getMatrixZoomData(dataset, chromosome, chromosome, zoom);
                        } catch (Exception e) {
                            System.err.println("No data for " + norm.getLabel() + " - " + chromosome + " at " + zoom);
                            return;
                        }

                        if (zd == null) {
                            System.err.println("Null MatrixZoomData");
                            return;
                        }

                        NormalizationCalculations calculations = new NormalizationCalculations(zd.getIteratorContainer());
                        Double[] matrixSum = getNormMatrixSumFactor(actualVector, zd.getIteratorContainer());

                        int numValidVectorEntries = calculations.getNumberOfValidEntriesInVector(actualVector);
                        Double[] result = new Double[]{matrixSum[0], matrixSum[1],
                                (double) numValidVectorEntries};


                        String key = getKeyWithNorm(chromosome, zoom, norm);
                        synchronized (zoomToMatrixSumMap) {
                            zoomToMatrixSumMap.put(key, result);
                        }
                        System.out.println("Finished: " + key);

                        /*
                        testCode(zoom, zd.getContactRecordList(), actualVector, 1. / matrixSum[0],
                                numValidVectorEntries / matrixSum[0]);
                         */
                    }
                };
                executor.execute(worker);
            }
        }

        executor.shutdown();
        while (!executor.isTerminated()) {
        }

        printWriter.println("Normalization Type: " + norm);
        List<double[]> matrixFormat = new ArrayList<>();
        for (Chromosome chromosome : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {
            printWriter.println("Chromsome: " + chromosome + " index: " + chromosome.getIndex());
            for (HiCZoom zoom : dataset.getBpZooms()) {
                String key = getKeyWithNorm(chromosome, zoom, norm);
                if (zoomToMatrixSumMap.containsKey(key)) {
                    printWriter.println("Zoom: " + zoom + " Normalized Matrix Sum: " + zoomToMatrixSumMap.get(key)[0]
                            + " Original Matrix Sum: " + zoomToMatrixSumMap.get(key)[1]
                            + " Number of Positive Entries in Vector: " + zoomToMatrixSumMap.get(key)[2]);
                    matrixFormat.add(new double[]{
                            (double) chromosome.getIndex(),
                            (double) zoom.getBinSize(),
                            zoomToMatrixSumMap.get(key)[0],
                            zoomToMatrixSumMap.get(key)[1],
                            zoomToMatrixSumMap.get(key)[2]
                    });
                }
            }
        }
    
        printWriter.close();
    
        double[][] matrixFormatArray = new double[matrixFormat.size()][5];
        for (int i = 0; i < matrixFormat.size(); i++) {
            matrixFormatArray[i] = matrixFormat.get(i);
        }
    
        MatrixTools.saveMatrixTextV2(outputTxtFile.getAbsolutePath(), matrixFormatArray);
        MatrixTools.saveMatrixTextNumpy(outputNpyFile.getAbsolutePath(), matrixFormatArray);
    }

    public Double[] getNormMatrixSumFactor(double[] norm, IteratorContainer ic) {
        double matrix_sum = 0;
        double norm_sum = 0;

        Iterator<ContactRecord> iterator = ic.getNewContactRecordIterator();
        while (iterator.hasNext()) {
            ContactRecord cr = iterator.next();

            int x = cr.getBinX();
            int y = cr.getBinY();
            float value = cr.getCounts();
            double valX = norm[x];
            double valY = norm[y];
            if (!Double.isNaN(valX) && !Double.isNaN(valY) && valX > 0 && valY > 0) {
                // want total sum of matrix, not just upper triangle
                if (x == y) {
                    norm_sum += value / (valX * valY);
                    matrix_sum += value;
                } else {
                    norm_sum += 2 * value / (valX * valY);
                    matrix_sum += 2 * value;
                }
            }
        }
        return new Double[]{norm_sum, matrix_sum};
    }
    
    private void testCode(HiCZoom zoom, List<ContactRecord> contactRecordList, double[] actualVector, double scalar1, double scalar2) {
        
        if (zoom.getBinSize() > 100000) {
            System.out.println("No scaling");
            System.out.println(Arrays.toString(MatrixTools.getRowSums(contactRecordList,
                    1, actualVector)));
            System.out.println("Scale by 1/sum");
            System.out.println(Arrays.toString(MatrixTools.getRowSums(contactRecordList,
                    scalar1, actualVector)));
            System.out.println("Scale by N/M");
            System.out.println(Arrays.toString(MatrixTools.getRowSums(contactRecordList,
                    scalar2, actualVector)));
        }
    }
}