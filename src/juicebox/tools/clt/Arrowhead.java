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

import juicebox.HiC;
import juicebox.data.Dataset;
import juicebox.data.HiCFileTools;
import juicebox.data.Matrix;
import juicebox.data.MatrixZoomData;
import juicebox.tools.HiCTools;
import juicebox.tools.utils.juicer.arrowhead.ArrowheadScoreList;
import juicebox.tools.utils.juicer.arrowhead.BlockBuster;
import juicebox.windowui.HiCZoom;
import org.broad.igv.Globals;
import org.broad.igv.feature.Chromosome;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * Created by nchernia on 1/9/15.
 */
public class Arrowhead extends JuiceboxCLT {

    private String file, outputPath;
    private int resolution = -100;

    public Arrowhead() {
        super("arrowhead <input_HiC_file(s)> <output_file> <resolution>");
    }

    @Override
    public void readArguments(String[] args, HiCTools.CommandLineParser parser) throws IOException {
        //setUsage("juicebox arrowhead hicFile resolution");
        if (args.length != 4) {
            throw new IOException("1");
        }
        file = args[1];
        outputPath = args[2];
        try {
            resolution = Integer.valueOf(args[3]);
        } catch (NumberFormatException error) {
            throw new IOException("1");
        }
    }

    @Override
    public void run() throws IOException {

        // might need to catch OutofMemory errors.  10Kb => 8GB, 5Kb => 12GB in original script
        Dataset ds = HiCFileTools.extractDatasetForCLT(Arrays.asList(file.split("\\+")), true);

        List<Chromosome> chromosomes = ds.getChromosomes();

        // Note: could make this more general if we wanted, to arrowhead calculation at any BP or FRAG resolution
        HiCZoom zoom = new HiCZoom(HiC.Unit.BP, resolution);

        for (Chromosome chr : chromosomes) {
            if (chr.getName().equals(Globals.CHR_ALL)) continue;

            Matrix matrix = ds.getMatrix(chr, chr);

            if (matrix == null) continue;
            System.out.println("\nProcessing " + chr.getName());
            MatrixZoomData zd = matrix.getZoomData(zoom);
            ArrowheadScoreList list = new ArrowheadScoreList();
            ArrowheadScoreList control = new ArrowheadScoreList();
            BlockBuster.run(chr.getIndex(), chr.getName(), chr.getLength(), resolution, outputPath + resolution,
                    zd, list, control);
        }
    }
}