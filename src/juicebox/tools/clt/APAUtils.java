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

package juicebox.tools.clt;

import com.sun.tools.javac.util.ArrayUtils;
import juicebox.data.Block;
import juicebox.data.ContactRecord;
import juicebox.data.MatrixZoomData;
import juicebox.track.Feature2D;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math.stat.descriptive.rank.Percentile;
import org.apache.log4j.Logger;
import org.broad.igv.Globals;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.util.ParsingUtils;

import java.io.*;
import java.util.*;

/**
 * Created by Muhammad Shamim on 1/21/15.
 */
public class APAUtils {

    public static Map<Chromosome,ArrayList<Feature2D>> loadLoopList(String path, ArrayList<Chromosome> chromosomes,
                                                                    double min_peak_dist, double max_peak_dist) throws IOException {

        Map<Chromosome,ArrayList<Feature2D>> chrToLoops = new HashMap<Chromosome,ArrayList<Feature2D>>();
        Map<String,ArrayList<Feature2D>> loopMap = loadLoopMap(path, min_peak_dist, max_peak_dist);

        Set<String> keys = loopMap.keySet();

        for(Chromosome chr : chromosomes){
            for (String chrString : keys){
                if(equivalentChromosome(chrString,chr)){
                    chrToLoops.put(chr, new ArrayList<Feature2D>(loopMap.get(chrString)));
                    keys.remove(chrString);
                    break;
                }
            }
        }

        return chrToLoops;
    }

    private static Map<String,ArrayList<Feature2D>> loadLoopMap(String path,
                                                                double min_peak_dist, double max_peak_dist) throws IOException {
        BufferedReader br = null;

        Map<String,ArrayList<Feature2D>> chrToLoopsMap = new HashMap<String, ArrayList<Feature2D>>();

        try {
            br = ParsingUtils.openBufferedReader(path);
            String nextLine;

            nextLine = br.readLine(); // header
            String[] headers = Globals.tabPattern.split(nextLine);

            int errorCount = 0;
            int lineNum = 1;
            while ((nextLine = br.readLine()) != null) {
                lineNum++;
                String[] tokens = Globals.tabPattern.split(nextLine);
                if (tokens.length > headers.length) {
                    throw new IOException("Improperly formatted loop file");
                }
                if (tokens.length < 6) {
                    continue;
                }

                String chr1Name, chr2Name;
                int start1, end1, start2, end2;
                try {
                    chr1Name = tokens[0].replaceAll("chr", "").toLowerCase();
                    start1 = Integer.parseInt(tokens[1]);
                    end1 = Integer.parseInt(tokens[2]);

                    chr2Name = tokens[3].replaceAll("chr", "").toLowerCase();
                    start2 = Integer.parseInt(tokens[4]);
                    end2 = Integer.parseInt(tokens[5]);
                } catch (Exception e) {
                    throw new IOException("Line " + lineNum + " improperly formatted in <br>" +
                            path + "<br>Line format should start with:  CHR1  X1  X2  CHR2  Y1  Y2");
                }

                if(chr1Name.equals(chr2Name)) {
                    Feature2D feature = new Feature2D("Peak", chr1Name, start1, end1, chr2Name, start2, end2, null, null);
                    if(chrToLoopsMap.containsKey(chr1Name)){
                        chrToLoopsMap.get(chr1Name).add(feature);
                    }
                    else{
                        ArrayList<Feature2D> newList = new ArrayList<Feature2D>();
                        newList.add(feature);
                        chrToLoopsMap.put(chr1Name, newList);
                    }
                }
            }
        } finally {
            if (br != null) br.close();
        }

        Map<String,ArrayList<Feature2D>> filteredChrToLoopsMap = new HashMap<String, ArrayList<Feature2D>>();

        for(String key : chrToLoopsMap.keySet()){
            filteredChrToLoopsMap.put(key, filterLoopsBySize(chrToLoopsMap.get(key), min_peak_dist, max_peak_dist));
        }

        return filteredChrToLoopsMap;
    }


    private static ArrayList<Feature2D> filterLoopsBySize(ArrayList<Feature2D> loops,
                                                          double min_peak_dist, double max_peak_dist) {

        ArrayList<Feature2D> filteredLoops = new ArrayList<Feature2D>();

        for(Feature2D loop : loops){
            int x = Math.abs(loop.getEnd1() - loop.getStart1());
            int y = Math.abs(loop.getEnd2() - loop.getStart2());

            if(x >= min_peak_dist && y >= min_peak_dist)
                if(x <= max_peak_dist && y <= max_peak_dist)
                    filteredLoops.add(loop);
        }

        return new ArrayList<Feature2D>(filteredLoops);
    }



    private static boolean equivalentChromosome(String token, Chromosome chr) {
        String token2 = token.toLowerCase().replaceAll("chr","");
        String chrName = chr.getName().toLowerCase().replaceAll("chr", "");
        return token2.equals(chrName);
    }

    public static int[] range(int start, int stop) {
        int[] result = new int[stop - start];
        for (int i = 0; i < stop - start; i++)
            result[i] = start + i;
        return result;
    }

    //TODO
    public static double[][] uniqueRows(double[][] iMatrix) {
        //unique_a = unique(a.view([('', a.dtype)]*a.shape[1]))
        //return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
        return new double[2][2];
    }

    public static void saveMeasures(Array2DRowRealMatrix x, int mdpt, String filename) {
        int width = 6; //width of boxes
        Writer writer = null;
        //int totWidth = x.getColumnDimension();
        //float widthp5 = width + .5f;
        //float half_width = width/2f;


        try {
            writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "utf-8"));
            writer.write("P2M" + '\t' + peak2mean(x, mdpt, width) + '\n');
            writer.write("P2UL" + '\t' + peak2UL(x, mdpt, width) + '\n');
            writer.write("P2UR" + '\t' + peak2UR(x, mdpt, width) + '\n');
            writer.write("P2LL" + '\t' + peak2LL(x, mdpt, width) + '\n');
            writer.write("P2LR" + '\t' + peak2LR(x, mdpt, width) + '\n');
            writer.write("ZscoreLL" + '\t' + ZscoreLL(x, mdpt, width));
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            try {
                if(writer != null)
                    writer.close();
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    }


    public static void saveMatrixText(String filename, Array2DRowRealMatrix realMatrix) {
        Writer writer = null;
        try {
            writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "utf-8"));
            double[][] matrix = realMatrix.getData();
            for (double[] row : matrix) {
                for (double val : row) {
                    writer.write(val + " ");
                }
                writer.write("\n");
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            try {
                if(writer != null)
                    writer.close();
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    }

    public static void saveListText(String filename, List<Double> array) {
        Writer writer = null;
        try {
            writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "utf-8"));
            for (double val : array) {
                writer.write(val + " ");
            }
            writer.write("\n");
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            try {
                if(writer != null)
                    writer.close();
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    }

    // NOTE - indices are inclusive in java
    // but in python the second index is not inclusive

    private static double peak2mean(Array2DRowRealMatrix x, int mdpt, int width) {
        double centralVal = x.getEntry(mdpt, mdpt);
        return centralVal / ((sum(x.getData()) - centralVal) / (x.getColumnDimension() - 1));
    }

    private static double peak2UL(Array2DRowRealMatrix x, int mdpt, int width) {
        double centralVal = x.getEntry(mdpt, mdpt);
        double avg = mean(x.getSubMatrix(0, width - 1, 0, width - 1).getData());
        return centralVal / avg;
    }

    private static double peak2UR(Array2DRowRealMatrix x, int mdpt, int width) {
        double centralVal = x.getEntry(mdpt, mdpt);
        int max = x.getColumnDimension();
        double avg = mean(x.getSubMatrix(0, width - 1, max - width, max - 1).getData());
        return centralVal / avg;
    }

    private static double peak2LL(Array2DRowRealMatrix x, int mdpt, int width) {
        double centralVal = x.getEntry(mdpt, mdpt);
        int max = x.getColumnDimension();
        double avg = mean(x.getSubMatrix(max - width, max - 1, 0, width - 1).getData());
        return centralVal / avg;
    }

    private static double peak2LR(Array2DRowRealMatrix x, int mdpt, int width) {
        double centralVal = x.getEntry(mdpt, mdpt);
        int max = x.getColumnDimension();
        double avg = mean(x.getSubMatrix(max - width, max - 1, max - width, max - 1).getData());
        return centralVal / avg;
    }

    private static double ZscoreLL(Array2DRowRealMatrix x, int mdpt, int width) {
        double centralVal = x.getEntry(mdpt, mdpt);
        int max = x.getColumnDimension();
        DescriptiveStatistics yStats = statistics(x.getSubMatrix(max - width, max - 1, 0, width - 1).getData());
        return (centralVal - yStats.getMean()) / yStats.getStandardDeviation();
    }

    private static double sum(double[][] x) {
        double total = 0;
        for (double[] row : x)
            for (double val : row)
                total += val;
        return total;
    }

    private static DescriptiveStatistics statistics(double[][] x) {
        DescriptiveStatistics stats = new DescriptiveStatistics();
        for (double[] row : x)
            for (double val : row)
                stats.addValue(val);
        return stats;
    }

    private static double mean(double[][] x) {
        return statistics(x).getMean();
    }

    private static double mean(Array2DRowRealMatrix x) {
        return statistics(x.getData()).getMean();
    }

    public static Array2DRowRealMatrix standardNormalization(Array2DRowRealMatrix matrix) {
        Array2DRowRealMatrix normeddata = cleanArray2DMatrix(matrix.getRowDimension(),
                matrix.getColumnDimension()).add(matrix);
        normeddata.scalarMultiply(1./Math.max(1.,APAUtils.mean(matrix)));
        return normeddata;
    }

    public static Array2DRowRealMatrix centerNormalization(Array2DRowRealMatrix matrix) {

        int center = matrix.getRowDimension()/2;
        double centerVal = matrix.getEntry(center, center);

        if(centerVal == 0){
            centerVal = minimumPositive(matrix.getData());
            if (centerVal == 0)
                centerVal = 1;
        }

        Array2DRowRealMatrix normeddata = cleanArray2DMatrix(matrix.getRowDimension(),
                matrix.getColumnDimension()).add(matrix);
        normeddata.scalarMultiply(1./centerVal);
        return normeddata;
    }

    public static double peakEnhancement(Array2DRowRealMatrix matrix){
        int rows = matrix.getRowDimension();
        int center = rows/2;
        double centerVal = matrix.getEntry(center, center);
        double remainingSum = sum(matrix.getData()) - centerVal;
        double remainingAverage = remainingSum/(rows*rows-1);
        return centerVal / remainingAverage;
    }

    private static double minimumPositive(double[][] data) {
        double minVal = Double.MAX_VALUE;
        for(double[] row : data){
            for(double val : row) {
                if(val > 0 && val < minVal)
                    minVal = val;
            }
        }
        if(minVal == Double.MAX_VALUE)
            minVal = 0;
        return minVal;
    }


    public static Array2DRowRealMatrix cleanArray2DMatrix(int rows, int cols){
        Array2DRowRealMatrix matrix = new Array2DRowRealMatrix(rows,cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                matrix.setEntry(r,c,0);
        return matrix;
    }


    public static Array2DRowRealMatrix extractLocalizedData(MatrixZoomData zd, Feature2D loop,
                                                            int L, int resolution, int window) {
        long time = System.nanoTime();
        int loopX = loop.getStart1();
        int loopY = loop.getStart2();
        int binXStart = loopX - resolution*(window+1);
        int binXEnd = loopX + resolution*(window+1);
        int binYStart = loopY - resolution*(window+1);
        int binYEnd = loopY + resolution*(window+1);

        System.out.println((System.nanoTime()-time)/1000000000.);

        List<Block> blocks = zd.getNormalizedBlocksOverlapping(binXStart, binYStart, binXEnd, binYEnd,
                NormalizationType.NONE);

        System.out.println((System.nanoTime()-time)/1000000000.);

        Array2DRowRealMatrix data = APAUtils.cleanArray2DMatrix(L, L);

        System.out.println((System.nanoTime()-time)/1000000000.);

        for (Block b : blocks) {
            for (ContactRecord rec : b.getContactRecords()) {
                //, rec.getBinY(), rec.getCounts()

                int relativeX = window + (rec.getBinX() - loopX)/resolution;
                int relativeY = window + (rec.getBinY() - loopY)/resolution;

                if(relativeX >= 0 && relativeX < L){
                    if(relativeY >= 0 && relativeY < L){
                        data.addToEntry(relativeX, relativeY, rec.getCounts());
                    }
                }
            }
        }
        System.out.println((System.nanoTime()-time)/1000000000.);
        return data;
    }

    public static Array2DRowRealMatrix rankPercentile(Array2DRowRealMatrix data) {
        int n = data.getColumnDimension();
        Percentile percentile = new Percentile();
        percentile.setData(flattenSquareMatrix(data));

        Array2DRowRealMatrix matrix = new Array2DRowRealMatrix(n,n);
        for (int r = 0; r < n; r++){
            for (int c = 0; c < n; c++) {
                double currValue = data.getEntry(r, c);
                if(currValue == 0){
                    matrix.setEntry(r, c, 0);
                }
                else {
                    matrix.setEntry(r, c, percentile.evaluate(currValue));
                }
                //matrix.setEntry(r, c, percentile.evaluate());
            }
        }
        return matrix;
    }

    private static double[] flattenSquareMatrix(Array2DRowRealMatrix matrix){
        int n = matrix.getColumnDimension();
        int numElements = n*n;
        double[] flatMatrix = new double[numElements];

        int index = 0;
        for (double[] row : matrix.getData()){
            System.arraycopy(row, 0, flatMatrix, index, n);
            index += n;
        }
        return flatMatrix;
    }
}