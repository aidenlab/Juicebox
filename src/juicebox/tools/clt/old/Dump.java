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

import htsjdk.tribble.util.LittleEndianOutputStream;
import jargs.gnu.CmdLineParser;
import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.data.*;
import juicebox.tools.clt.CommandLineParser;
import juicebox.tools.clt.JuiceboxCLT;
import juicebox.tools.utils.original.ExpectedValueCalculation;
import juicebox.tools.utils.original.norm.GenomeWideNormalizationVectorUpdater;
import juicebox.tools.utils.original.norm.NormalizationCalculations;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.MatrixType;
import juicebox.windowui.NormalizationHandler;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.util.ParsingUtils;
import org.broad.igv.util.ResourceLocator;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;

public class Dump extends JuiceboxCLT {


    private static int[] regionIndices = new int[]{-1, -1, -1, -1};
    private static boolean useRegionIndices = false;
    private HiC.Unit unit = null;
    private String chr1, chr2;
    private ChromosomeHandler chromosomeHandler;
    private int binSize = 0;
    private MatrixType matrixType = null;
    private PrintWriter pw = null;
    private LittleEndianOutputStream les = null;
    private HiCZoom zoom = null;
    private boolean includeIntra = false;
    private boolean dense = false;
    private String feature = null;

    public Dump() {
        super(getUsage());
    }

    public static String getUsage(){
        return "dump <observed/oe> <NONE/VC/VC_SQRT/KR> <hicFile(s)> <chr1>[:x1:x2] <chr2>[:y1:y2] <BP/FRAG> <binsize> [outfile]\n" +
                "\tdump <norm/expected> <NONE/VC/VC_SQRT/KR> <hicFile(s)> <chr> <BP/FRAG> <binsize> [outfile]\n" +
                "\tdump <loops/domains> <hicFile URL> [outfile]";
    }

    private void dumpGenomeWideData() {

        if (unit == HiC.Unit.FRAG) {
            System.err.println("All versus All currently not supported on fragment resolution");
            System.exit(8);
        }

        // This is for internal purposes only - to print the All vs All matrix
        // This matrix should not in general be exposed since it is arbitrarily binned
        // If in the future we wish to expose, we should use a more reasonable flag.
        if (zoom.getBinSize() == 6197 || zoom.getBinSize() == 6191) {
            Chromosome chr = chromosomeHandler.getChromosomeFromName("All");
            Matrix matrix =  dataset.getMatrix(chr, chr);
            if (matrix == null) {
                System.err.println("No All vs. All matrix");
                System.exit(1);
            }
            MatrixZoomData zd = matrix.getZoomData(zoom);
            if (zd == null){
                System.err.println("No All vs. All matrix; be sure zoom is correct");
                System.exit(1);
            }
            Iterator<ContactRecord> iter = zd.getNewContactRecordIterator();
            while (iter.hasNext()) {
                ContactRecord cr = iter.next();
                pw.println(cr.getBinX() + "\t" + cr.getBinY() + "\t" + cr.getCounts());
            }
            pw.close();
            return;
        }

        // Build a "whole-genome" matrix
        ArrayList<ContactRecord> recordArrayList = GenomeWideNormalizationVectorUpdater.createWholeGenomeRecords(dataset, chromosomeHandler, zoom, includeIntra);

        if (recordArrayList.isEmpty()) {
            System.err.println("No reads found at " +  zoom +". Include intra is " + includeIntra);
            return;
        }
        int totalSize = 0;
        for (Chromosome c1 : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {
            totalSize += c1.getLength() / binSize + 1;
        }

        NormalizationCalculations calculations = new NormalizationCalculations(recordArrayList, totalSize);
        double[] vector = calculations.getNorm(norm);


        if (matrixType == MatrixType.NORM) {

            ExpectedValueCalculation evKR = new ExpectedValueCalculation(chromosomeHandler, binSize, null, NormalizationHandler.GW_KR);
            int addY = 0;
            // Loop through chromosomes
            for (Chromosome chr : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {
                final int chrIdx = chr.getIndex();
                Matrix matrix = dataset.getMatrix(chr, chr);

                if (matrix == null) continue;
                MatrixZoomData zd = matrix.getZoomData(zoom);
                Iterator<ContactRecord> iter = zd.getNewContactRecordIterator();
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

                addY += chr.getLength() / binSize + 1;
            }
            evKR.computeDensity();
            double[] exp = evKR.getDensityAvg();
            pw.println(binSize + "\t" + vector.length + "\t" + exp.length);
            for (double aVector : vector) {
                pw.println(aVector);
            }

            for (double aVector : exp) {
                pw.println(aVector);
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

                pw.println(x + "\t" + y + "\t" + value);
            }
        }

        pw.close();
    }

    private void dumpGeneralVector() {

        Chromosome chromosome = chromosomeHandler.getChromosomeFromName(chr1);

        if (matrixType == MatrixType.NORM) {
            NormalizationVector nv = dataset.getNormalizationVector(chromosome.getIndex(), zoom, norm);
            if (nv == null) {
                System.err.println("Norm not available at " + zoom + " " + norm);
                System.exit(9);
            }
            int count=0;
            int total = (chromosome.getLength()/zoom.getBinSize())+1;
            // print out vector
            for (double element : nv.getData()) {
                pw.println(element);
                count++;
                // Do not print out more entries than length of chromosome
                if (zoom.getUnit() == HiC.Unit.BP) {
                    if (count >= total) break;
                }
            }
            pw.close();

        } else if (matrixType == MatrixType.EXPECTED) {
            final ExpectedValueFunction df = dataset.getExpectedValues(zoom, norm);
            if (df == null) {
                System.err.println("Expected not available at " + zoom + " " + norm);
                System.exit(10);
            }
            int length = df.getLength();

            if (ChromosomeHandler.isAllByAll(chromosome)) { // removed cast to ExpectedValueFunctionImpl
                // print out vector
                for (double element : df.getExpectedValues()) {
                    pw.println(element);
                }
                pw.close();
            } else {
                for (int i = 0; i < length; i++) {
                    pw.println((float) df.getExpectedValue(chromosome.getIndex(), i));
                }

                pw.close();
            }
        }
    }

    /**
     * Dumps the matrix.  Does more argument checking, thus this should not be called outside of this class.
     *
     * @throws java.io.IOException   In case of problems writing out matrix
     */
    private void dumpMatrix() throws IOException {

        Chromosome chromosome1 = chromosomeHandler.getChromosomeFromName(chr1);
        Chromosome chromosome2 = chromosomeHandler.getChromosomeFromName(chr2);

        Matrix matrix = dataset.getMatrix(chromosome1, chromosome2);
        if (matrix == null) {
            System.err.println("No reads in " + chr1 + " " + chr2);
            return;
        }

        if (chromosome2.getIndex() < chromosome1.getIndex()) {
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


        ExpectedValueFunction df = null;
        if (MatrixType.isExpectedValueType(matrixType)) {
            df = dataset.getExpectedValues(zd.getZoom(), norm);
            if (df == null) {
                System.err.println(matrixType + " not available at " + chr1 + " " + zoom + " " + norm);
                System.exit(14);
            }
        }
        zd.dump(pw, les, norm, matrixType, useRegionIndices, regionIndices, df, dense);

    }

    private void dumpFeature() throws IOException {
        ResourceLocator locator;
        if (feature.equals("loops")) {
            locator = dataset.getPeaks();
        }
        else locator = dataset.getBlocks();

        if (locator == null) {
            System.err.println("Sorry, " + feature + " is not available for this dataset");
            System.exit(0);
        }


        BufferedReader br = new BufferedReader(new InputStreamReader(ParsingUtils.openInputStream(locator.getPath())), HiCGlobals.bufferSize);
        String nextLine;

        // header
        while ((nextLine = br.readLine()) != null) {
            pw.println(nextLine);
        }

        pw.close();
    }

    @Override
    public void readArguments(String[] args, CmdLineParser parser) {
        String ofile = null;

        if (args.length == 3 || args.length == 4) {
            if (!(args[1].equals("loops") || args[1].equals("domains"))) {
                printUsageAndExit();
            }
            if (!args[2].endsWith("hic") || !args[2].startsWith("http")) {
                printUsageAndExit();
            }
            dataset = HiCFileTools.extractDatasetForCLT(Arrays.asList(args[2].split("\\+")), false);
            feature = args[1];
            if (args.length == 4) {
                ofile = args[3];
            }
        }
        else {
            // -d in pre means diagonal, in dump means dense
            dense = ((CommandLineParser) parser).getDiagonalsOption();
            // -n in pre means no norm, in dump means includeIntra for the whole genome
            includeIntra = ((CommandLineParser) parser).getNoNormOption();

            if (args.length < 7) {
                printUsageAndExit();
            }

            String mType = args[1].toLowerCase();

            matrixType = MatrixType.enumValueFromString(mType);
            if (!(MatrixType.isDumpMatrixType(matrixType) || MatrixType.isDumpVectorType(matrixType))) {
                System.err.println("Matrix or vector must be one of \"observed\", \"oe\",  \"norm\", or \"expected\".");
                System.exit(15);
            }

            setDatasetAndNorm(args[3], args[2], false);

            chromosomeHandler = dataset.getChromosomeHandler();

            // retrieve input chromosomes / regions
            chr1 = args[4];
            if (MatrixType.isDumpMatrixType(matrixType)) {
                chr2 = args[5];

            } else chr2 = chr1;

            extractChromosomeRegionIndices(); // at the end of this, chr1&2 will just be the chr key names

            if (chromosomeHandler.doesNotContainChromosome(chr1)) {
                System.err.println("Unknown chromosome: " + chr1);
                System.exit(18);
            }

            if (chromosomeHandler.doesNotContainChromosome(chr2)) {
                System.err.println("Unknown chromosome: " + chr2);
                System.exit(19);
            }


            try {
                if (MatrixType.isDumpMatrixType(matrixType)) {
                    unit = HiC.valueOfUnit(args[6]);
                } else {
                    unit = HiC.valueOfUnit(args[5]);
                }
            } catch (IllegalArgumentException error) {
                System.err.println("Unit must be in BP or FRAG.");
                System.exit(20);
            }

            String binSizeSt;
            if (MatrixType.isDumpMatrixType(matrixType)) {
                binSizeSt = args[7];
            } else {
                binSizeSt = args[6];
            }

            try {
                binSize = Integer.parseInt(binSizeSt);
            } catch (NumberFormatException e) {
                System.err.println("Integer expected for bin size.  Found: " + binSizeSt + ".");
                System.exit(21);
            }

            zoom = new HiCZoom(unit, binSize);

            if (MatrixType.isDumpMatrixType(matrixType)) {
                if (args.length == 9) ofile = args[8];
            } else {
                if (args.length == 8) ofile = args[7];
            }
        }

        try {
            if (ofile != null && ofile.length() > 0) {
                if (ofile.endsWith(".bin")) {
                    BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(args[6]));
                    les = new LittleEndianOutputStream(bos);
                }
                else pw = new PrintWriter(new FileOutputStream(ofile));
            } else {
                pw = new PrintWriter(System.out);
            }
        }
        catch (IOException error) {
            System.err.println("Unable to open " + ofile + " for writing.");
            System.exit(22);
        }


    }

    /**
     * Added for benchmark
     */
    public void setQuery(String chr1, String chr2, int binSize) {
        this.chr1 = chr1;
        this.chr2 = chr2;
        this.binSize = binSize;
        extractChromosomeRegionIndices();
    }

    public int[] getBpBinSizes() {
        int[] bpBinSizes = new int[dataset.getNumberZooms(HiC.Unit.BP)];
        for (int zoomIdx = 0; zoomIdx < bpBinSizes.length; zoomIdx++) {
            bpBinSizes[zoomIdx] = dataset.getZoom(HiC.Unit.BP, zoomIdx).getBinSize();
        }
        return bpBinSizes;
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
            Chromosome chromosome1 = chromosomeHandler.getChromosomeFromName(chr1);
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
            Chromosome chromosome2 = chromosomeHandler.getChromosomeFromName(chr2);
            regionIndices[2] = 0;
            regionIndices[3] = chromosome2.getLength();
        }
    }

    @Override
    public void run() {
        try {
            if (feature != null) {
                dumpFeature();
            }
            else if ((matrixType == MatrixType.OBSERVED || matrixType == MatrixType.NORM)
                    && ChromosomeHandler.isAllByAll(chr1)
                    && ChromosomeHandler.isAllByAll(chr2)) {
                    dumpGenomeWideData();
            } else if (MatrixType.isDumpMatrixType(matrixType)) {
                dumpMatrix();
            } else if (MatrixType.isDumpVectorType(matrixType)) {
                dumpGeneralVector();
            }
        }
        catch (Exception e) {
            System.err.println("Unable to dump");
            e.printStackTrace();
        }
    }

    public ChromosomeHandler getChromosomeHandler() {
        return chromosomeHandler;
    }
}