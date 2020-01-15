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

import juicebox.data.HiCFileTools;
import juicebox.data.MatrixZoomData;
import juicebox.mapcolorui.Feature2DHandler;
import juicebox.tools.utils.common.UNIXTools;
import org.broad.igv.feature.Chromosome;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class IterateDownDiagonalFinder extends RegionFinder {

    public IterateDownDiagonalFinder(ParameterConfigurationContainer container) {
        super(container);
    }

    @Override
    public void makeExamples() {

        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

        final Feature2DHandler feature2DHandler = new Feature2DHandler(inputFeature2DList);

        for (int resolution : resolutions) {
            for (Chromosome chrom : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {

                Runnable worker = new Runnable() {
                    @Override
                    public void run() {

                        int numFilesWrittenSoFar = 0;
                        int currentBatchNumber = 0;
                        int maxBatchSize = 10000;

                        String newFolderPath = originalPath + "/" + resolution + "_chr" + chrom.getName();
                        UNIXTools.makeDir(newFolderPath);

                        BufferedWriter[] writers = new BufferedWriter[3];
                        try {

                            final MatrixZoomData zd = HiCFileTools.getMatrixZoomData(ds, chrom, chrom, resolution);
                            if (zd == null) return;
                            System.out.println("Currently processing: " + chrom.getName());

                            generateWriters(currentBatchNumber, writers, newFolderPath);
                            String negPath = UNIXTools.makeDir(newFolderPath + "/negative_" + currentBatchNumber);
                            String posPath = UNIXTools.makeDir(newFolderPath + "/positive_" + currentBatchNumber);

                            // sliding along the diagonal
                            for (int rowIndex = 0; rowIndex < (chrom.getLength() / resolution) - y; rowIndex += stride) {
                                int startCol = Math.max(0, rowIndex - offsetOfCornerFromDiagonal);
                                int endCol = Math.min(rowIndex + offsetOfCornerFromDiagonal + 1, (chrom.getLength() / resolution) - y);
                                for (int colIndex = startCol; colIndex < endCol; colIndex += stride) {
                                    getTrainingDataAndSaveToFile(zd, chrom, rowIndex, colIndex, resolution, feature2DHandler, x, y,
                                            posPath, negPath, writers[0], writers[2], writers[1], false);
                                    numFilesWrittenSoFar++;
                                    if (numFilesWrittenSoFar > maxBatchSize) {
                                        numFilesWrittenSoFar = 0;
                                        currentBatchNumber++;

                                        generateWriters(currentBatchNumber, writers, newFolderPath);
                                        negPath = UNIXTools.makeDir(newFolderPath + "/negative_" + currentBatchNumber);
                                        posPath = UNIXTools.makeDir(newFolderPath + "/positive_" + currentBatchNumber);
                                    }
                                }
                            }
                            if (x != y) {
                                // only rectangular regions require the double traveling
                                for (int rowIndex = y; rowIndex < (chrom.getLength() / resolution); rowIndex += stride) {
                                    int startCol = Math.max(y, rowIndex - offsetOfCornerFromDiagonal);
                                    int endCol = Math.min(rowIndex + offsetOfCornerFromDiagonal + 1, (chrom.getLength() / resolution));
                                    for (int colIndex = startCol; colIndex < endCol; colIndex += stride) {
                                        getTrainingDataAndSaveToFile(zd, chrom, rowIndex, colIndex, resolution, feature2DHandler, x, y,
                                                posPath, negPath, writers[0], writers[2], writers[1], true);
                                        numFilesWrittenSoFar++;
                                        if (numFilesWrittenSoFar > maxBatchSize) {
                                            numFilesWrittenSoFar = 0;
                                            currentBatchNumber++;

                                            generateWriters(currentBatchNumber, writers, newFolderPath);
                                            negPath = UNIXTools.makeDir(newFolderPath + "/negative_" + currentBatchNumber);
                                            posPath = UNIXTools.makeDir(newFolderPath + "/positive_" + currentBatchNumber);
                                        }
                                    }
                                }
                            }

                            for (Writer writer : writers) {
                                writer.close();
                            }

                        } catch (Exception e) {
                            e.printStackTrace();
                            return;
                        }
                    }
                };
                executor.execute(worker);
            }

        }

        executor.shutdown();
        while (!executor.isTerminated()) {
        }
    }

    private void generateWriters(int currentBatchNumber, BufferedWriter[] writers, String newFolderPath) throws FileNotFoundException {
        writers[0] = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(
                newFolderPath + "/pos_file_names_" + currentBatchNumber + ".txt"), StandardCharsets.UTF_8));
        writers[1] = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(
                newFolderPath + "/neg_file_names_" + currentBatchNumber + ".txt"), StandardCharsets.UTF_8));
        writers[2] = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(
                newFolderPath + "/pos_label_file_names_" + currentBatchNumber + ".txt"), StandardCharsets.UTF_8));
    }
}
