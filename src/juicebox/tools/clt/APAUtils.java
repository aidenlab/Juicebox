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

import juicebox.track.Feature2D;
import juicebox.track.Feature2DList;
import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;
import org.broad.igv.Globals;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.ui.color.ColorUtilities;
import org.broad.igv.util.ParsingUtils;

import java.awt.*;
import java.io.*;
import java.util.*;

/**
 * Created by Muhammad Shamim on 1/21/15.
 */
public class APAUtils {

    private static final Logger log = Logger.getLogger(APA.class);

    public static Map<Chromosome,ArrayList<Feature2D>> loadLoopList(String path, ArrayList<Chromosome> chromosomes) throws IOException {

        Map<Chromosome,ArrayList<Feature2D>> chrToLoops = new HashMap<Chromosome,ArrayList<Feature2D>>();
        Map<String,ArrayList<Feature2D>> loopMap = loadLoopMap(path);

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

    public static Map<String,ArrayList<Feature2D>> loadLoopMap(String path) throws IOException {
        BufferedReader br = null;

        Map<String,ArrayList<Feature2D>> chrToLoopsMap = new HashMap<String, ArrayList<Feature2D>>();


        try {
            br = ParsingUtils.openBufferedReader(path);
            String nextLine;

            // header
            nextLine = br.readLine();
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
        return chrToLoopsMap;
    }

    private static boolean equivalentChromosome(String token, Chromosome chr) {
        if (token.toLowerCase().equals(chr.getName().toLowerCase()) || String.valueOf("chr").concat(token.toLowerCase()).equals(chr.getName().toLowerCase()) || token.toLowerCase().equals(String.valueOf("chr").concat(chr.getName().toLowerCase())))
            return true;
        return false;
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
                writer.close();
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    }

    public static void saveArrayText(String filename, double[] array) {
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
                writer.close();
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    }

    // NOTE - indices are inclusive in java
    // but in python the second index is not inclusive

    public static double peak2mean(Array2DRowRealMatrix x, int mdpt, int width) {
        double centralVal = x.getEntry(mdpt, mdpt);
        return centralVal / ((sum(x.getData()) - centralVal) / (x.getColumnDimension() - 1));
    }

    public static double peak2UL(Array2DRowRealMatrix x, int mdpt, int width) {
        double centralVal = x.getEntry(mdpt, mdpt);
        double avg = mean(x.getSubMatrix(0, width - 1, 0, width - 1).getData());
        return centralVal / avg;
    }

    public static double peak2UR(Array2DRowRealMatrix x, int mdpt, int width) {
        double centralVal = x.getEntry(mdpt, mdpt);
        int max = x.getColumnDimension();
        double avg = mean(x.getSubMatrix(0, width - 1, max - width, max - 1).getData());
        return centralVal / avg;
    }

    public static double peak2LL(Array2DRowRealMatrix x, int mdpt, int width) {
        double centralVal = x.getEntry(mdpt, mdpt);
        int max = x.getColumnDimension();
        double avg = mean(x.getSubMatrix(max - width, max - 1, 0, width - 1).getData());
        return centralVal / avg;
    }

    public static double peak2LR(Array2DRowRealMatrix x, int mdpt, int width) {
        double centralVal = x.getEntry(mdpt, mdpt);
        int max = x.getColumnDimension();
        double avg = mean(x.getSubMatrix(max - width, max - 1, max - width, max - 1).getData());
        return centralVal / avg;
    }

    public static double ZscoreLL(Array2DRowRealMatrix x, int mdpt, int width) {
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


}
