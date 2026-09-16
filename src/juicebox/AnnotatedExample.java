/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2022 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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


package juicebox;
import  java.io.*;

import com.opencsv.CSVWriter;
import juicebox.data.*;
import juicebox.data.basics.Chromosome;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationHandler;
import juicebox.windowui.NormalizationType;

import java.io.File;
import java.util.ArrayList;
import java.util.List;


public class AnnotatedExample {
    public static void main(String[] args) throws IOException {
        // do you want to cache portions of the file?
        // this uses more RAM, but if you want to repeatedly
        // query nearby regions, it can improve speed by a lot

        String[] timedata = { "80", "10", "620","100", "230" };

        boolean useCache = false;
        String filename = "https://www.dropbox.com/s/a6ykz8ajgszv0b6/Trachops_cirrhosus.rawchrom.hic";
        filename = juicebox.data.HiCFileTools.cleanUpDropboxURL(filename);

        // create a hic dataset object

        long s1 = System.nanoTime();

        List<String> files = new ArrayList<>();
        files.add(filename);
        Dataset ds = HiCFileTools.extractDatasetForCLT(files, false);
        long s2 = System.nanoTime();
        System.out.println((s2 - s1) * 1e-9);
        System.out.println("^^^^ line 29 execution line");

        timedata[0]=String.valueOf((s2 - s1) * 1e-9);

        // pick the normalization we would like
        // this line will check multiple possible norms
        // and pick whichever is available (in order of preference)

        NormalizationType norm = NormalizationHandler.NONE;

        System.out.println("Norm being used: " + norm.getLabel());

        int resolution = 5000;


        juicebox.data.basics.Chromosome[] chromosomes = ds.getChromosomeHandler().getChromosomeArrayWithoutAllByAll();



        // now let's iterate on every chromosome (only intra-chromosomal regions for now)
        for (Chromosome chromosome : chromosomes) {
            if (chromosome.getIndex() > 6) continue;
            long s7 = System.nanoTime();
            Matrix matrix = ds.getMatrix(chromosome, chromosome);
            long s8 = System.nanoTime();
            System.out.println((s8 - s7) * 1e-9);
            System.out.println("^^^^ line 49 execution line");

            timedata[1]=String.valueOf((s8 - s7) * 1e-9);



            if (matrix == null) continue;
            long s9 = System.nanoTime();

            MatrixZoomData zd = matrix.getZoomData(new HiCZoom(HiC.Unit.BP, resolution));
            long s10 = System.nanoTime();
            System.out.println((s10 - s9) * 1e-9);
            System.out.println("^^^^ line 57 execution line");

            timedata[2]=String.valueOf((s10 - s9) * 1e-9);




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
            List<Block> blocks = zd.getNormalizedBlocksOverlapping(binXStart, binYStart, binXEnd, binYEnd, norm,false, getDataUnderTheDiagonal);
            long s12 = System.nanoTime();
            System.out.println((s12 - s11) * 1e-9);
            System.out.println("^^^^ line 77 execution line");

            timedata[3]=String.valueOf((s12 - s11) * 1e-9);


            binXStart = 1500;
            binYStart = 1600;
            binXEnd = 2000;
            binYEnd = 2200;
            s11 = System.nanoTime();
            blocks = zd.getNormalizedBlocksOverlapping(binXStart, binYStart, binXEnd, binYEnd, norm,false, getDataUnderTheDiagonal);
            s12 = System.nanoTime();
            System.out.println((s12 - s11) * 1e-9);
            System.out.println("^^^^ line 88 execution line");

            timedata[4]=String.valueOf((s12 - s11) * 1e-9);


            //change file path for the different versions of juicebox you are testing, and to wherever you want to store the file. I had two different paths.
            String csvFilePath = "C:\\Users\\Mahdi\\Downloads\\Juicebox2.csv";
            File f = new File(csvFilePath);

            if (f.exists()){
                CSVWriter writer = new CSVWriter(new FileWriter(csvFilePath, true));

                writer.writeNext(timedata);

                writer.close();
            }

            else{
                FileWriter outputfile = new FileWriter(csvFilePath);

                // create CSVWriter object filewriter object as parameter
                CSVWriter writer = new CSVWriter(outputfile);

                // adding header to csv
                String[] header = { "L29", "L49", "L57", "L77", "L88" };
                writer.writeNext(header);

                // add data to csv
                writer.writeNext(timedata);
                writer.close();
            }

        }
    }
}