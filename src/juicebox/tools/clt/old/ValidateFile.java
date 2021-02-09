/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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

import juicebox.HiCGlobals;
import juicebox.data.*;
import juicebox.data.basics.Chromosome;
import juicebox.tools.clt.CommandLineParser;
import juicebox.tools.clt.JuiceboxCLT;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * Created by muhammadsaadshamim on 6/2/16.
 */
public class ValidateFile extends JuiceboxCLT {

    private String filePath;

    public ValidateFile() {
        super(getUsage());
    }

    public static String getUsage() {
        return "validate <hicFile>";
    }

    @Override
    public void readArguments(String[] args, CommandLineParser parser) {
        if (args.length != 2) {
            printUsageAndExit();
        }
        filePath = args[1];
    }

    @Override
    public void run() {
        try {
            DatasetReader reader = HiCFileTools.extractDatasetReaderForCLT(Arrays.asList(filePath.split("\\+")), true);
            Dataset ds = reader.read();
            HiCGlobals.verifySupportedHiCFileVersion(reader.getVersion());
            assert ds.getGenomeId() != null;
            assert ds.getChromosomeHandler().size() > 0;
            List<HiCZoom> zooms = ds.getBpZooms();
            List<NormalizationType> norms = ds.getNormalizationTypes();

            for (NormalizationType type : norms) {
                System.out.println("File has normalization: " + type.getLabel());
                System.out.println("Description: " + type.getDescription());
            }

            for (HiCZoom zoom : zooms) {
                System.out.println("File has zoom: " + zoom);
            }

            Chromosome[] array = ds.getChromosomeHandler().getChromosomeArrayWithoutAllByAll();
            for (Chromosome chr: array)  {
                for (Chromosome chr2: array) {
                    System.out.print(".");
                    Matrix matrix = ds.getMatrix(chr, chr2);
                    if (matrix == null) {
                        System.err.println("Warning: no reads in " + chr.getName() + " " + chr2.getName());
                    }
                    else {
                        for (HiCZoom zoom: zooms) {
                            MatrixZoomData zd = matrix.getZoomData(zoom);
                            for (NormalizationType type: norms) {
                                reader.readNormalizedBlock(0, zd, type);
                            }
                        }
                    }
                }
                System.out.println();
            }
            System.out.println("(-: Validation successful");
            System.exit(0);
            throw new IOException("t");
        }
        catch (IOException error) {
            System.err.println(":( Validation failed");
            error.printStackTrace();
            System.exit(1);
        }
    }
}