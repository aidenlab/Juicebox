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

import htsjdk.tribble.util.LittleEndianOutputStream;
import juicebox.HiC;
import juicebox.data.*;
import juicebox.tools.utils.original.ExpectedValueCalculation;
import juicebox.tools.HiCTools;
import juicebox.tools.utils.original.NormalizationCalculations;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import org.broad.igv.Globals;
import org.broad.igv.feature.Chromosome;

import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

public class Dump extends JuiceboxCLT {


    private HiC.Unit unit = null;
    private NormalizationType norm = null;

    private final List<String> files = new ArrayList<String>();
    private String chr1, chr2;
    private Dataset dataset = null;
    private List<Chromosome> chromosomeList;
    private Map<String, Chromosome> chromosomeMap;
    private int binSize = 0;
    private String type;
    private String ofile = null;
    private boolean includeIntra = false;

    public Dump(){
        super("dump <observed/oe/norm/expected> <NONE/VC/VC_SQRT/KR> <hicFile(s)> <chr1> <chr2> <BP/FRAG> <binsize> [outfile]");
    }

    @Override
    public void readArguments(String[] args, HiCTools.CommandLineParser parser) throws IOException {
        //juicebox dump <observed/oe/pearson/norm/expected/eigenvector> <NONE/VC/VC_SQRT/KR> <hicFile> <chr1> <chr2> <BP/FRAG> <binsize> [outfile]")

        String mType = args[1].toLowerCase();
        if (!(mType.equals("observed") || mType.equals("oe") ||
                mType.equals("pearson") || mType.equals("norm") ||
                mType.equals("expected") || mType.equals("eigenvector"))) {
            System.err.println("Matrix or vector must be one of \"observed\", \"oe\", \"pearson\", \"norm\", " +
                    "\"expected\", or \"eigenvector\".");
            throw new IOException("-1");
        }

        try {
            norm = NormalizationType.valueOf(args[2]);
        } catch (IllegalArgumentException error) {
            System.err.println("Normalization must be one of \"NONE\", \"VC\", \"VC_SQRT\", \"KR\", \"GW_KR\", \"GW_VC\", \"INTER_KR\", or \"INTER_VC\".");
            throw new IOException("-1");
        }

        if (!args[3].endsWith("hic")) {
            System.err.println("Only 'hic' files are supported");
            throw new IOException("-1");
        }

        int idx = 3;
        while (idx < args.length && args[idx].endsWith("hic")) {
            files.add(args[idx]);
            idx++;
        }

        // idx is now the next argument.  following arguments should be chr1, chr2, unit, binsize, [outfile]
        if (args.length != idx + 4 && args.length != idx + 5) {
            System.err.println("Incorrect number of arguments to \"dump\"");
            throw new IOException("-1");
        }
        chr1 = args[idx];
        chr2 = args[idx + 1];


        if (files.size() == 1) {
            String magicString = DatasetReaderV2.getMagicString(files.get(0));

            DatasetReader reader = null;
            if (magicString.equals("HIC")) {
                reader = new DatasetReaderV2(files.get(0));
            } else {
                System.err.println("This version of HIC is no longer supported");
                System.exit(-1);
            }
            dataset = reader.read();
        } else {
            DatasetReader reader = DatasetReaderFactory.getReader(files);
            if (reader == null) {
                System.err.println("Error while reading files");
                System.exit(-1);
            } else {
                dataset = reader.read();
            }
        }

        chromosomeList = dataset.getChromosomes();

        chromosomeMap = new HashMap<String, Chromosome>();
        for (Chromosome c : chromosomeList) {
            chromosomeMap.put(c.getName(), c);
        }

        if (!chromosomeMap.containsKey(chr1)) {
            System.err.println("Unknown chromosome: " + chr1);
            throw new IOException("-1");
        }
        if (!chromosomeMap.containsKey(chr2)) {
            System.err.println("Unknown chromosome: " + chr2);
            throw new IOException("-1");
        }


        try {
            unit = HiC.Unit.valueOf(args[idx + 2]);
        } catch (IllegalArgumentException error) {
            System.err.println("Unit must be in BP or FRAG.");
            throw new IOException("1");
        }

        String binSizeSt = args[idx + 3];

        try {
            binSize = Integer.parseInt(binSizeSt);
        } catch (NumberFormatException e) {
            System.err.println("Integer expected for bin size.  Found: " + binSizeSt + ".");
            throw new IOException("1");
        }


        type = args[1];

        if ((type.equals("observed") || type.equals("norm")) && chr1.equals(Globals.CHR_ALL) && chr2.equals(Globals.CHR_ALL)) {

            if (args.length == idx + 5) {
                includeIntra = true;
            }
        } else{// if (type.equals("oe") || type.equals("pearson") || type.equals("observed")) {

            if (args.length == idx + 5) {
                ofile = args[idx + 4];
            }
        }
    }


    @Override
    public void run() throws IOException {
        HiCZoom zoom = new HiCZoom(unit, binSize);

        //*****************************************************
        if ((type.equals("observed") || type.equals("norm")) && chr1.equals(Globals.CHR_ALL) && chr2.equals(Globals.CHR_ALL)) {
            dumpGenomeWideData(dataset, chromosomeList, includeIntra, zoom, norm, type, binSize);
        }
        else if (type.equals("oe") || type.equals("pearson") || type.equals("observed")) {
            dumpMatrix(dataset, chromosomeMap.get(chr1), chromosomeMap.get(chr2), norm, zoom, type, ofile);
        } else if (type.equals("norm") || type.equals("expected") || type.equals("eigenvector")) {
            dumpGeneralVector(dataset, chr1, chromosomeMap.get(chr1), norm, zoom, type, ofile, binSize, unit);
        }

    }

    private static void dumpGenomeWideData(Dataset dataset, List<Chromosome> chromosomeList,
                                           boolean includeIntra, HiCZoom zoom, NormalizationType norm,
                                           String type, int binSize) {
        if (zoom.getUnit() == HiC.Unit.FRAG) {
            System.err.println("All versus All currently not supported on fragment resolution");
            System.exit(1);
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

        if (type.equals("norm")) {

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
                                   NormalizationType norm, HiCZoom zoom, String type, String ofile,
                                   int binSize, HiC.Unit unit) throws IOException{
        PrintWriter pw;

        if (ofile != null && ofile.length() > 0) {
            pw = new PrintWriter(new FileOutputStream(ofile));
        }
        else {
            pw = new PrintWriter(System.out);
        }


        if (type.equals("norm")) {
            NormalizationVector nv = dataset.getNormalizationVector(chromosome.getIndex(), zoom, norm);
            if (nv == null) {
                System.err.println("Norm not available at " + chr + " " + binSize + " " + unit + " " + norm);
                System.exit(-1);
            }
            dumpVector(pw, nv.getData(), false);

        } else if (type.equals("expected")) {
            final ExpectedValueFunction df = dataset.getExpectedValues(zoom, norm);
            if (df == null) {
                System.err.println("Expected not available at " + chr + " " + binSize + " " + unit + " " + norm);
                System.exit(-1);
            }
            int length = df.getLength();
            if (chr.equals("All")) { // removed cast to ExpectedValueFunctionImpl
                dumpVector(pw, df.getExpectedValues(), false);
            } else {
                for (int i = 0; i < length; i++) {
                    pw.println((float) df.getExpectedValue(chromosome.getIndex(), i));
                }
                pw.flush();
                pw.close();
            }
        } else if (type.equals("eigenvector")) {
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
     * @param dataset Dataset
     * @param chr1    Chromosome 1
     * @param chr2    Chromosome 2
     * @param norm    Normalization
     * @param zoom    Zoom level
     * @param type    observed/oe/pearson
     * @param ofile   Output file string (binary output), possibly null (then prints to standard out)
     * @throws java.io.IOException
     */
    static private void dumpMatrix(Dataset dataset, Chromosome chr1, Chromosome chr2, NormalizationType norm,
                                   HiCZoom zoom, String type, String ofile) throws IOException {
        LittleEndianOutputStream les = null;
        BufferedOutputStream bos = null;
        PrintWriter txtWriter = null;

        // TODO should add check to ensure directory for file exists otherwise mkdir?
        if (ofile != null) {
            if(ofile.endsWith(".bin")){
                bos = new BufferedOutputStream(new FileOutputStream(ofile));
                les = new LittleEndianOutputStream(bos);
            }else{
                txtWriter = new PrintWriter(new FileOutputStream(ofile));
            }
        }

        if (type.equals("oe") || type.equals("pearson")) {
            if (!chr1.equals(chr2)) {
                System.err.println("Chromosome " + chr1 + " not equal to Chromosome " + chr2);
                System.err.println("Currently only intrachromosomal O/E and Pearson's are supported.");
                throw new IOException("-1");
            }
        }

        Matrix matrix = dataset.getMatrix(chr1, chr2);
        if (matrix == null) {
            System.err.println("No reads in " + chr1 + " " + chr2);
            System.exit(-1);
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
            System.exit(-1);
        }

        if (type.equals("oe") || type.equals("pearson")) {
            final ExpectedValueFunction df = dataset.getExpectedValues(zd.getZoom(), norm);
            if (df == null) {
                System.err.println(type + " not available at " + chr1 + " " + zoom + " " + norm);
                System.exit(-1);
            }
            try {
                zd.dumpOE(df, type, norm, les, null);
            } finally {
                if (les != null)
                    les.close();
                if (bos != null)
                    bos.close();
            }
        } else if (type.equals("observed")) {
            double[] nv1 = null;
            double[] nv2 = null;
            if (norm != NormalizationType.NONE) {
                NormalizationVector nv = dataset.getNormalizationVector(chr1.getIndex(), zd.getZoom(), norm);
                if (nv == null) {
                    System.err.println(type + " not available at " + chr1 + " " + zoom + " " + norm);
                    System.exit(-1);
                } else {
                    nv1 = nv.getData();
                }
                if (!chr1.equals(chr2)) {
                    nv = dataset.getNormalizationVector(chr2.getIndex(), zd.getZoom(), norm);
                    if (nv == null) {
                        System.err.println(type + " not available at " + chr2 + " " + zoom + " " + norm);
                        System.exit(-1);
                    } else {
                        nv2 = nv.getData();
                    }
                } else {
                    nv2 = nv1;
                }
            }
            if(txtWriter != null){
                zd.dump(txtWriter, nv1, nv2);
            }
            else if (les == null) {
                zd.dump(new PrintWriter(System.out), nv1, nv2);
            } else {
                try {
                    zd.dump(les, nv1, nv2);
                } finally {
                    les.close();
                    bos.close();
                }
            }
        }
    }
}