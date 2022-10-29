package juicebox;

import javastraw.reader.Dataset;
import javastraw.reader.basics.Chromosome;
import javastraw.reader.block.Block;
import javastraw.reader.block.ContactRecord;
import javastraw.reader.mzd.Matrix;
import javastraw.reader.mzd.MatrixZoomData;
import javastraw.reader.norm.NormalizationPicker;
import javastraw.reader.type.HiCZoom;
import javastraw.reader.type.NormalizationType;
import javastraw.tools.HiCFileTools;

import java.util.Iterator;
import java.util.List;

public class AnnotatedExample {
    public static void main(String[] args) {
        // do you want to cache portions of the file?
        // this uses more RAM, but if you want to repeatedly
        // query nearby regions, it can improve speed by a lot
        boolean useCache = false;
        String filename = "https://www.dropbox.com/s/a6ykz8ajgszv0b6/Trachops_cirrhosus.rawchrom.hic";

        // create a hic dataset object

        long s1 = System.nanoTime();
        Dataset ds = HiCFileTools.extractDatasetForCLT(filename, false, useCache, false);
        long s2 = System.nanoTime();
        System.out.println((s2-s1)*1e-9);
        System.out.println("^^^^ line 28 execution line");

        // pick the normalization we would like
        // this line will check multiple possible norms
        // and pick whichever is available (in order of preference)

        long s3 = System.nanoTime();
        NormalizationType norm = NormalizationPicker.getFirstValidNormInThisOrder(ds, new String[]{"KR", "SCALE", "VC", "VC_SQRT", "NONE"});
        long s4 = System.nanoTime();
        System.out.println((s4-s3)*1e-9);
        System.out.println("^^^^ line 40 execution line");
        System.out.println("Norm being used: " + norm.getLabel());

        // let's set our resolution
        int resolution = 5000;

        // let's grab the chromosomes

        long s5 = System.nanoTime();
        Chromosome[] chromosomes = ds.getChromosomeHandler().getChromosomeArrayWithoutAllByAll();
        long s6 = System.nanoTime();
        System.out.println((s6-s5)*1e-9);
        System.out.println("^^^^ line 52 execution line");


        // now let's iterate on every chromosome (only intra-chromosomal regions for now)
        for (Chromosome chromosome : chromosomes) {
            long s7 = System.nanoTime();
            Matrix matrix = ds.getMatrix(chromosome, chromosome);
            long s8 = System.nanoTime();
            System.out.println((s8-s7)*1e-9);
            System.out.println("^^^^ line 59 execution line");


            if (matrix == null) continue;
            long s9 = System.nanoTime();

            MatrixZoomData zd = matrix.getZoomData(new HiCZoom(resolution));
            long s10 = System.nanoTime();
            System.out.println((s10-s9)*1e-9);
            System.out.println("^^^^ line 70 execution line");

            if (zd == null) continue;

            // zd is now a data structure that contains pointers to the data
            // *** Let's show 2 different ways to access data ***

            // OPTION 2
            // just grab sparse data for a specific region

            // choose your setting for when the diagonal is in the region
            boolean getDataUnderTheDiagonal = true;

            // our bounds will be binXStart, binYStart, binXEnd, binYEnd
            // these are in BIN coordinates, not genome coordinates
            int binXStart = 500, binYStart = 600, binXEnd = 1000, binYEnd = 1200;
            long s11 = System.nanoTime();
            List<Block> blocks = zd.getNormalizedBlocksOverlapping(binXStart, binYStart, binXEnd, binYEnd, norm, getDataUnderTheDiagonal);
            long s12 = System.nanoTime();
            System.out.println((s12-s11)*1e-9);
            System.out.println("^^^^ line 88 execution line");


            long s13 = System.nanoTime();
            for (Block b : blocks) {
                if (b != null) {
                    for (ContactRecord rec : b.getContactRecords()) {
                        if (rec.getCounts() > 0) { // will skip NaNs
                            // can choose to use the BIN coordinates
                            int binX = rec.getBinX();
                            int binY = rec.getBinY();

                            // you could choose to use relative coordinates for the box given
                            int relativeX = rec.getBinX() - binXStart;
                            int relativeY = rec.getBinY() - binYStart;

                            float counts = rec.getCounts();
                        }
                    }
                }
            }
            long s14 = System.nanoTime();
            System.out.println((s14-s13)*1e-9);
            System.out.println("^^^^ for loop execution line");
        }
    }
}