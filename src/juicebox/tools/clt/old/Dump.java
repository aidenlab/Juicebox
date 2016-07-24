/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2016 Broad Institute, Aiden Lab
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

import htsjdk.tribble.util.LittleEndianOutputStream;
import jargs.gnu.CmdLineParser;
import juicebox.HiC;
import juicebox.data.*;
import juicebox.tools.clt.JuiceboxCLT;
import juicebox.tools.utils.original.ExpectedValueCalculation;
import juicebox.tools.utils.original.NormalizationCalculations;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.MatrixType;
import juicebox.windowui.NormalizationType;
import org.broad.igv.Globals;
import org.broad.igv.feature.Chromosome;

import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

public class Dump extends JuiceboxCLT {


    private static int[] regionIndices = new int[]{-1, -1, -1, -1};
    private static boolean useRegionIndices = false;
    private final List<String> files = new ArrayList<String>();
    private HiC.Unit unit = null;
    private NormalizationType norm = null;
    private String chr1, chr2;
    private Dataset dataset = null;
    private List<Chromosome> chromosomeList;
    private Map<String, Chromosome> chromosomeMap;
    private int binSize = 0;
    private MatrixType matrixType = null;
    private String ofile = null;
    private boolean includeIntra = false;

    public Dump() {
        super(getUsage());
    }

    public static String getUsage(){
        return "dump <observed/oe/norm/expected> <NONE/VC/VC_SQRT/KR> <hicFile(s)> <chr1>[:x1:x2] <chr2>[:y1:y2] <BP/FRAG> <binsize> <outfile>";
    }

    private static void dumpGenomeWideData(Dataset dataset, List<Chromosome> chromosomeList,
                                           boolean includeIntra, HiCZoom zoom, NormalizationType norm,
                                           MatrixType matrixType, int binSize) {
        if (zoom.getUnit() == HiC.Unit.FRAG) {
            System.err.println("All versus All currently not supported on fragment resolution");
            System.exit(8);
        }

        // Build a "whole-genome" matrix
        ArrayList<ContactRecord> recordArrayList = createWholeGenomeRecords(dataset, chromosomeList, zoom, includeIntra);

        int totalSize = 0;
        for (Chromosome c1 : chromosomeList) {
            if (c1.getName().equals(Globals.CHR_ALL)) continue;
            totalSize += c1.getLength() / zoom.getBinSize() + 1;
        }

        NormalizationCalculations calculations = new NormalizationCalculations(recordArrayList, totalSize);
        double[] vector = calculations.getNorm(norm);

        if (matrixType == MatrixType.NORM) {

            ExpectedValueCalculation evKR = new ExpectedValueCalculation(chromosomeList, zoom.getBinSize(), null, NormalizationType.GW_KR);
            int addY = 0;
            // Loop through chromosomes
            for (Chromosome chr : chromosomeList) {

                if (chr.getName().equals(Globals.CHR_ALL)) continue;
                final int chrIdx = chr.getIndex();
                Matrix matrix = dataset.getMatrix(chr, chr);

                if (matrix == null) continue;
                MatrixZoomData zd = matrix.getZoomData(zoom);
                Iterator<ContactRecord> iter = zd.contactRecordIterator();
                while (iter.hasNext()) {
                    ContactRecord cr = iter.next();
                    int x = cr.getBinX();
                    int y = cr.getBinY();
                    final float counts = cr.getCounts();
                    if (vector[x + addY] > 0 && vector[y + addY] > 0 && !Double.isNaN(vector[x + addY]) && !Double.isNaN(vector[y + addY])) {
                        double value = counts / (vector[x + addY] * vector[y + addY]);
                        evKR.addDistance(chrIdx, x, y, value);
                    }
                }

                addY += chr.getLength() / zoom.getBinSize() + 1;
            }
            evKR.computeDensity();
            double[] exp = evKR.getDensityAvg();
            System.out.println(binSize + "\t" + vector.length + "\t" + exp.length);
            for (double aVector : vector) {
                System.out.println(aVector);
            }

            for (double aVector : exp) {
                System.out.println(aVector);
            }
        } else {   // type == "observed"

            for (ContactRecord cr : recordArrayList) {
                int x = cr.getBinX();
                int y = cr.getBinY();
                float value = cr.getCounts();

                if (vector[x] != 0 && vector[y] != 0 && !Double.isNaN(vector[x]) && !Double.isNaN(vector[y])) {
                    value = (float) (value / (vector[x] * vector[y]));
                } else {
                    value = Float.NaN;
                }

                System.out.println(x + "\t" + y + "\t" + value);
            }
        }
    }

    private static void dumpGeneralVector(Dataset dataset, String chr, Chromosome chromosome,
                                          NormalizationType norm, HiCZoom zoom, MatrixType matrixType, String ofile,
                                          int binSize, HiC.Unit unit) throws IOException {
        PrintWriter pw;

        if (ofile != null && ofile.length() > 0) {
            pw = new PrintWriter(new FileOutputStream(ofile));
        } else {
            pw = new PrintWriter(System.out);
        }


        if (matrixType == MatrixType.NORM) {
            NormalizationVector nv = dataset.getNormalizationVector(chromosome.getIndex(), zoom, norm);
            if (nv == null) {
                System.err.println("Norm not available at " + chr + " " + binSize + " " + unit + " " + norm);
                System.exit(9);
            }

            dumpVector(pw, nv.getData(), false);

        } else if (matrixType == MatrixType.EXPECTED) {
            final ExpectedValueFunction df = dataset.getExpectedValues(zoom, norm);
            if (df == null) {
                System.err.println("Expected not available at " + chr + " " + binSize + " " + unit + " " + norm);
                System.exit(10);
            }
            int length = df.getLength();

            if (HiCFileTools.isAllChromosome(chr)) { // removed cast to ExpectedValueFunctionImpl
                dumpVector(pw, df.getExpectedValues(), false);
            } else {
                for (int i = 0; i < length; i++) {
                    pw.println((float) df.getExpectedValue(chromosome.getIndex(), i));
                }
                pw.flush();
                pw.close();
            }
        } else if (matrixType == MatrixType.EIGENVECTOR) {
            dumpVector(pw, dataset.getEigenvector(chromosome, zoom, 0, norm), true);
        }
    }

    private static ArrayList<ContactRecord> createWholeGenomeRecords(Dataset dataset, List<Chromosome> tmp, HiCZoom zoom, boolean includeIntra) {
        ArrayList<ContactRecord> recordArrayList = new ArrayList<ContactRecord>();
        int addX = 0;
        int addY = 0;
        for (Chromosome c1 : tmp) {
            if (c1.getName().equals(Globals.CHR_ALL)) continue;
            for (Chromosome c2 : tmp) {
                if (c2.getName().equals(Globals.CHR_ALL)) continue;
                if (c1.getIndex() < c2.getIndex() || (c1.equals(c2) && includeIntra)) {
                    Matrix matrix = dataset.getMatrix(c1, c2);
                    if (matrix != null) {
                        MatrixZoomData zd = matrix.getZoomData(zoom);
                        if (zd != null) {
                            Iterator<ContactRecord> iter = zd.contactRecordIterator();
                            while (iter.hasNext()) {
                                ContactRecord cr = iter.next();
                                int binX = cr.getBinX() + addX;
                                int binY = cr.getBinY() + addY;
                                recordArrayList.add(new ContactRecord(binX, binY, cr.getCounts()));
                            }
                        }
                    }
                }
                addY += c2.getLength() / zoom.getBinSize() + 1;
            }
            addX += c1.getLength() / zoom.getBinSize() + 1;
            addY = 0;
        }
        return recordArrayList;
    }

    /**
     * Prints out a vector to the given print stream.  Mean centers if center is set
     *
     * @param pw     Stream to print to
     * @param vector Vector to print out
     * @param center Mean centers if true
     */
    static public void dumpVector(PrintWriter pw, double[] vector, boolean center) {
        double mean = 0;
        if (center) {
            int count = 0;
            double total = 0;

            for (double element : vector) {
                if (!Double.isNaN(element)) {
                    total += element;
                    count++;
                }
            }

            mean = total / count; // sum is now mean
        }
        // print out vector
        for (double element : vector) {
            pw.println(element - mean);
        }
        pw.flush();
        pw.close();
    }

    /**
     * Dumps the matrix.  Does more argument checking, thus this should not be called outside of this class.
     *
     * @param dataset    Dataset
     * @param chr1       Chromosome 1
     * @param chr2       Chromosome 2
     * @param norm       Normalization
     * @param zoom       Zoom level
     * @param matrixType observed/oe/pearson
     * @param ofile      Output file string (binary output), possibly null (then prints to standard out)
     * @throws java.io.IOException
     */
    static private void dumpMatrix(Dataset dataset, Chromosome chr1, Chromosome chr2, NormalizationType norm,
                                   HiCZoom zoom, MatrixType matrixType, String ofile) throws IOException {
        LittleEndianOutputStream les = null;
        BufferedOutputStream bos = null;
        PrintWriter txtWriter = null;

        // TODO should add check to ensure directory for file exists otherwise mkdir?
        if (ofile != null) {
            if (ofile.endsWith(".bin")) {
                bos = new BufferedOutputStream(new FileOutputStream(ofile));
                les = new LittleEndianOutputStream(bos);
            } else {
                txtWriter = new PrintWriter(new FileOutputStream(ofile));
            }
        }

        if (MatrixType.isOnlyIntrachromosomalType(matrixType) || matrixType == MatrixType.OE) {
            if (!chr1.equals(chr2)) {
                System.err.println("Chromosome " + chr1 + " not equal to Chromosome " + chr2);
                System.err.println("Currently only intrachromosomal O/E, Pearson's, and VS are supported.");
                System.exit(11);
            }
        }

        Matrix matrix = dataset.getMatrix(chr1, chr2);
        if (matrix == null) {
            System.err.println("No reads in " + chr1 + " " + chr2);
            System.exit(12);
        }

        if (chr2.getIndex() < chr1.getIndex()) {
            regionIndices = new int[]{regionIndices[2], regionIndices[3], regionIndices[0], regionIndices[1]};
        }

        MatrixZoomData zd = matrix.getZoomData(zoom);
        if (zd == null) {

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

        try {
            ExpectedValueFunction df = null;
            if (MatrixType.isExpectedValueType(matrixType)) {
                df = dataset.getExpectedValues(zd.getZoom(), norm);
                if (df == null) {
                    System.err.println(matrixType + " not available at " + chr1 + " " + zoom + " " + norm);
                    System.exit(14);
                }
            }
            zd.dump(txtWriter, les, norm, matrixType, useRegionIndices, regionIndices, df);
        } finally {
            if (les != null) les.close();
            if (bos != null) bos.close();
        }
    }

    @Override
    public void readArguments(String[] args, CmdLineParser parser) {

        if (args.length < 7) {
            printUsageAndExit();
        }

        String mType = args[1].toLowerCase();

        matrixType = MatrixType.enumValueFromString(mType);
        if (matrixType == null) {
            System.err.println("Matrix or vector must be one of \"observed\", \"oe\", \"pearson\", \"norm\", " +
                    "\"expected\", or \"eigenvector\".");
            System.exit(15);
        }

        try {
            norm = NormalizationType.valueOf(args[2]);
        } catch (IllegalArgumentException error) {
            System.err.println("Normalization must be one of \"NONE\", \"VC\", \"VC_SQRT\", \"KR\", \"GW_KR\"," +
                    " \"GW_VC\", \"INTER_KR\", or \"INTER_VC\".");
            System.exit(16);
        }

        if (!args[3].endsWith("hic")) {
            System.err.println("Only 'hic' files are supported");
            System.exit(17);
        }

        int idx = 3;
        // TODO make this consistent with rest of summing hic maps syntax
        while (idx < args.length && args[idx].endsWith("hic")) {
            files.add(args[idx]);
            idx++;
        }

        // idx is now the next argument.  following arguments should be chr1, chr2, unit, binsize, [outfile]
        if (args.length != idx + 4 && args.length != idx + 5) {
            System.err.println("Incorrect number of arguments to \"dump\"");
            printUsageAndExit();
        }

        // initialize chromosome map
        dataset = HiCFileTools.extractDatasetForCLT(files, false);
        chromosomeList = dataset.getChromosomes();
        chromosomeMap = new HashMap<String, Chromosome>();
        for (Chromosome c : chromosomeList) {
            chromosomeMap.put(c.getName(), c);
        }

        // retrieve input chromosomes / regions
        chr1 = args[idx];
        chr2 = args[idx + 1];
        extractChromosomeRegionIndices(); // at the end of this, chr1&2 will just be the chr key names

        if (!chromosomeMap.containsKey(chr1)) {
            System.err.println("Unknown chromosome: " + chr1);
            System.exit(18);
        }
        if (!chromosomeMap.containsKey(chr2)) {
            System.err.println("Unknown chromosome: " + chr2);
            System.exit(19);
        }


        try {
            unit = HiC.valueOfUnit(args[idx + 2]);
        } catch (IllegalArgumentException error) {
            System.err.println("Unit must be in BP or FRAG.");
            System.exit(20);
        }

        String binSizeSt = args[idx + 3];

        try {
            binSize = Integer.parseInt(binSizeSt);
        } catch (NumberFormatException e) {
            System.err.println("Integer expected for bin size.  Found: " + binSizeSt + ".");
            System.exit(21);
        }


        if ((matrixType == MatrixType.OBSERVED || matrixType == MatrixType.NORM) && chr1.equals(Globals.CHR_ALL) && chr2.equals(Globals.CHR_ALL)) {

            if (args.length == idx + 5) {
                includeIntra = true;
            }
        } else {

            if (args.length == idx + 5) {
                ofile = args[idx + 4];
            }
        }
    }

    /**
     * Added so that subregions could be dumped without dumping the full chromosome
     */
    private void extractChromosomeRegionIndices() {
        if (chr1.contains(":")) {
            String[] regionComponents = chr1.split(":");
            if (regionComponents.length != 3) {
                System.err.println("Invalid number of indices for chr1: " + regionComponents.length +
                        ", should be 3 --> chromosome_name:start_index:end_index");
                printUsageAndExit();
            } else {
                try {
                    chr1 = regionComponents[0];
                    regionIndices[0] = Integer.parseInt(regionComponents[1]);
                    regionIndices[1] = Integer.parseInt(regionComponents[2]);
                    useRegionIndices = true;
                } catch (Exception e) {
                    System.err.println("Invalid indices for chr1: " + chr1);
                    printUsageAndExit();
                }
            }
        } else {
            Chromosome chromosome1 = chromosomeMap.get(chr1);
            regionIndices[0] = 0;
            regionIndices[1] = chromosome1.getLength();
        }

        if (chr2.contains(":")) {
            String[] regionComponents = chr2.split(":");
            if (regionComponents.length != 3) {
                System.err.println("Invalid number of indices for chr2 : " + regionComponents.length +
                        ", should be 3 --> chromosome_name:start_index:end_index");
                printUsageAndExit();
            } else {
                try {
                    chr2 = regionComponents[0];
                    regionIndices[2] = Integer.parseInt(regionComponents[1]);
                    regionIndices[3] = Integer.parseInt(regionComponents[2]);
                    useRegionIndices = true;
                } catch (Exception e) {
                    System.err.println("Invalid indices for chr2:  " + chr2);
                    printUsageAndExit();
                }
            }
        } else {
            Chromosome chromosome2 = chromosomeMap.get(chr2);
            regionIndices[2] = 0;
            regionIndices[3] = chromosome2.getLength();
        }
    }

    @Override
    public void run() {
        HiCZoom zoom = new HiCZoom(unit, binSize);

        //*****************************************************
        if ((matrixType == MatrixType.OBSERVED || matrixType == MatrixType.NORM)
                && chr1.equals(Globals.CHR_ALL)
                && chr2.equals(Globals.CHR_ALL)) {
            dumpGenomeWideData(dataset, chromosomeList, includeIntra, zoom, norm, matrixType, binSize);
        } else if (MatrixType.isDumpMatrixType(matrixType)) {
            try {
                dumpMatrix(dataset, chromosomeMap.get(chr1), chromosomeMap.get(chr2), norm, zoom, matrixType, ofile);
            } catch (Exception e) {
                System.err.println("Unable to dump matrix");
                e.printStackTrace();
            }
        } else if (MatrixType.isDumpVectorType(matrixType)) {
            try {
                dumpGeneralVector(dataset, chr1, chromosomeMap.get(chr1), norm, zoom, matrixType, ofile, binSize, unit);
            } catch (Exception e) {
                System.err.println("Unable to dump vector");
                e.printStackTrace();
            }
        }

    }
}