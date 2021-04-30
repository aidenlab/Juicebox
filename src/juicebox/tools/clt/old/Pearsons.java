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

import htsjdk.tribble.util.LittleEndianInputStream;
import htsjdk.tribble.util.LittleEndianOutputStream;
import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.data.ChromosomeHandler;
import juicebox.data.ExpectedValueFunction;
import juicebox.data.HiCFileTools;
import juicebox.data.MatrixZoomData;
import juicebox.data.basics.Chromosome;
import juicebox.matrix.BasicMatrix;
import juicebox.matrix.DiskResidentBlockMatrix;
import juicebox.matrix.InMemoryMatrix;
import juicebox.tools.clt.CommandLineParser;
import juicebox.tools.clt.JuiceboxCLT;
import juicebox.tools.utils.dev.PearsonCorrelationMetric;
import juicebox.windowui.HiCZoom;
import org.broad.igv.util.ParsingUtils;

import java.io.*;
import java.util.BitSet;

/**
 * Class for calculating Pearsons (separated out from Dump)
 * @author Neva Durand
 */
public class Pearsons extends JuiceboxCLT {

    private static final int BLOCK_TILE = 500;
    private String ofile = null;
    private HiC.Unit unit = null;
    private int binSize = 0;
    private Chromosome chromosome1;


    public Pearsons() {
        super(getBasicUsage() + "\n\t-p, --pearsons_all_resolutions: calculate Pearson's at all resolutions");
    }

    public static String getBasicUsage(){
        return "pearsons [-p] <NONE/VC/VC_SQRT/KR> <hicFile(s)> <chr> <BP/FRAG> <binsize> [outfile]";
    }

    public static BasicMatrix readPearsons(String path) throws IOException {

        // Peak at file to determine version
        BufferedInputStream bis = null;
        int magic;
        try {
            InputStream is = ParsingUtils.openInputStream(path);
            bis = new BufferedInputStream(is);
            LittleEndianInputStream les = new LittleEndianInputStream(bis);

            magic = les.readInt();

            if (magic != 6515048) {
                System.err.println("Problem reading Pearson's " + path);
                return null;
            }
        } finally {
            if (bis != null)
                bis.close();
        }

        return new DiskResidentBlockMatrix(path);

    }

    public static BasicMatrix computePearsons(double[][] columns, int dim) {

        BasicMatrix pearsons = new InMemoryMatrix(dim);

        BitSet bitSet = new BitSet(dim);
        for (int i = 0; i < dim; i++) {
            for (int j = i + 1; j < dim; j++) {
                double[] v1 = columns[i];
                double[] v2 = columns[j];
                if (v1 == null || v2 == null) {
                    pearsons.setEntry(i, j, Float.NaN);
                    pearsons.setEntry(j, i, Float.NaN);
                } else {
                    double corr = PearsonCorrelationMetric.corr(columns[i], columns[j]);
                    pearsons.setEntry(i, j, (float) corr);
                    pearsons.setEntry(j, i, (float) corr);
                    bitSet.set(i);
                    bitSet.set(j);
                }
            }
        }
        // Set diagonal to 1, set centromere to NaN
        for (int i = 0; i < dim; i++) {
            if (bitSet.get(i)) pearsons.setEntry(i, i, 1.0f);
            else pearsons.setEntry(i, i, Float.NaN);
        }
        return pearsons;
    }

    @Override
    public void readArguments(String[] args, CommandLineParser parser) {
        if (args.length != 7 && args.length != 6) {
            printUsageAndExit();
        }

        //HiCGlobals.MAX_PEARSON_ZOOM = 500000;
        setDatasetAndNorm(args[2], args[1], true);
        ChromosomeHandler chromosomeHandler = dataset.getChromosomeHandler();

        if (chromosomeHandler.doesNotContainChromosome(args[3])) {
            System.err.println("Unknown chromosome: " + args[3]);
            System.exit(18);
        }
        chromosome1 = chromosomeHandler.getChromosomeFromName(args[3]);

        try {
            unit = HiC.valueOfUnit(args[4]);
        } catch (IllegalArgumentException error) {
            System.err.println("Unit must be in BP or FRAG.");
            System.exit(20);
        }

        String binSizeSt = args[5];

        try {
            binSize = Integer.parseInt(binSizeSt);
        } catch (NumberFormatException e) {
            System.err.println("Integer expected for bin size.  Found: " + binSizeSt + ".");
            System.exit(21);
        }

        if ((unit == HiC.Unit.BP && binSize < HiCGlobals.MAX_PEARSON_ZOOM) ||
                (unit == HiC.Unit.FRAG && binSize < HiCGlobals.MAX_PEARSON_ZOOM / 1000)) {
            System.out.println("Pearson's and Eigenvector are not calculated for high resolution datasets");
            System.out.println("To override this limitation, send in the \"-p\" flag.");
            System.exit(0);
        }

        if (args.length == 7) {
            ofile = args[6];
        }

    }

    @Override
    public void run() {
        HiCZoom zoom = new HiCZoom(unit, binSize);
        MatrixZoomData zd = HiCFileTools.getMatrixZoomData(dataset, chromosome1, chromosome1, zoom);
        if (zd == null) {
            System.err.println("No reads in " + chromosome1);
            System.err.println("Unknown resolution: " + zoom);
            System.err.println("This data set has the following bin sizes (in bp): ");
            for (int zoomIdx = 0; zoomIdx < dataset.getNumberZooms(HiC.Unit.BP); zoomIdx++) {
                System.err.print(dataset.getZoom(HiC.Unit.BP, zoomIdx).getBinSize() + " ");
            }
            System.err.println("\nand the following bin sizes (in frag): ");
            for (int zoomIdx = 0; zoomIdx < dataset.getNumberZooms(HiC.Unit.FRAG); zoomIdx++) {
                System.err.print(dataset.getZoom(HiC.Unit.FRAG, zoomIdx).getBinSize() + " ");
            }
            System.exit(13);
        }
        ExpectedValueFunction df = dataset.getExpectedValuesOrExit(zd.getZoom(), norm, chromosome1, true);

        BasicMatrix pearsons = zd.getPearsons(df);
        if (pearsons == null) {
            System.err.println("Pearson's not available at zoom " + zoom  + ". For high resolution, try again with -p");
            System.exit(15);
        }

        LittleEndianOutputStream les = null;
        PrintWriter txtWriter = null;
        if (ofile != null) {
            try {
                if (ofile.endsWith(".bin")) {
                    BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(ofile));
                    les = new LittleEndianOutputStream(bos);
                } else {
                    txtWriter = new PrintWriter(new FileOutputStream(ofile));
                }
            } catch (IOException error) {
                System.err.println("Cannot write to " + ofile);
                error.printStackTrace();
                System.exit(22);
            }
        }
        else {
            txtWriter = new PrintWriter(System.out);
        }

        if (les == null) {
            int dim = pearsons.getRowDimension();
            for (int i = 0; i < dim; i++) {
                for (int j = 0; j < dim; j++) {
                    float output = pearsons.getEntry(i, j);
                    txtWriter.print(output + " ");
                }
                txtWriter.println();
            }
            txtWriter.flush();
        }
        else {
            try {
                int dim = pearsons.getRowDimension();
                writeHeader(les, dim, pearsons.getLowerValue(), pearsons.getUpperValue());
                int block_side = (int) Math.ceil((float) dim / (float) BLOCK_TILE);
                for (int i = 0; i < block_side; i++) {
                    int block_row_start = i * BLOCK_TILE;
                    int block_row_end = Math.min(block_row_start + BLOCK_TILE, dim);
                    int row_len = block_row_end - block_row_start;
                    for (int j = 0; j < block_side; j++) {
                        int block_col_start = j * BLOCK_TILE;
                        int block_col_end = Math.min(block_col_start + BLOCK_TILE, dim);
                        int col_len = block_col_end - block_col_start;
                        for (int ui = 0; ui < row_len; ui++) {
                            for (int uj = 0; uj < col_len; uj++) {
                                int now_i = ui + block_row_start;
                                int now_j = uj + block_col_start;
                                float output = pearsons.getEntry(now_i, now_j);
                                les.writeFloat(output);
                            }
                        }

                    }
                }
                les.close();
            }
            catch (IOException error) {
                System.err.println("Problem when writing Pearson's");
                error.printStackTrace();
                System.exit(1);
            }
        }
    }

    private void writeHeader(LittleEndianOutputStream les, int dim, float lower, float upper) throws IOException {

        // Magic number - 4 bytes
        les.writeByte('h');
        les.writeByte('i');
        les.writeByte('c');
        les.writeByte(0);

        // Version number
        les.writeInt(1);

        // Genome --
        les.writeString(dataset.getGenomeId());

        // Chromosomes
        les.writeString(chromosome1.getName());
        les.writeString(chromosome1.getName());

        // Resolution (bin size)
        les.writeInt(binSize);

        // Statistics, other attributes
        les.writeFloat(lower);  // this is supposed to be lower quartile
        les.writeFloat(upper);  // this is supposed to be upper quartile
        les.writeInt(dim);  // # rows
        les.writeInt(dim);  // # cols
        les.writeInt(BLOCK_TILE);
    }
}
