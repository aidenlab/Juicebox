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

import juicebox.HiCGlobals;
import juicebox.data.Dataset;
import juicebox.data.DatasetReaderV2;
import juicebox.data.Matrix;
import juicebox.data.MatrixZoomData;
import juicebox.tools.HiCTools;
import juicebox.tools.utils.Common.HiCFileTools;
import juicebox.tools.utils.Juicer.*;
import juicebox.tools.utils.Juicer.APA.APADataStack;
import juicebox.tools.utils.Juicer.APA.APAUtils;
import juicebox.track.Feature2D;
import juicebox.windowui.HiCZoom;
import org.apache.commons.math.linear.RealMatrix;
import org.broad.igv.Globals;
import org.broad.igv.feature.Chromosome;
import java.io.IOException;
import java.util.List;
import java.util.Set;

/**
 * TODO - once fully debugged, change notation convention from underscore to camelcase (match the rest of the files)
 */
public class APA extends JuiceboxCLT {

    private String[] files;

    public APA() {
        super("apa [-n minval] [-x maxval] [-w window]  [-r resolution] <hic file> <PeaksFile> <SaveFolder> [SavePrefix]");
    }

    //defaults
    private double min_peak_dist = 30; // distance between two bins, can be changed in opts
    private double max_peak_dist = Double.POSITIVE_INFINITY;
    private int window = 10;
    int width = 6; //size of boxes
    int peakwidth = 2; //for enrichment calculation of crosshair norm
    private int resolution = 10000;
    private final boolean saveAllData = true;

    private final String workingdirectory = System.getProperty("user.dir");

    @Override
    public void readArguments(String[] args, HiCTools.CommandLineParser parser) throws IOException {

        System.out.println(args);

        if (!(args.length > 3 && args.length < 6)) {
            throw new IOException("1");
        }

        files = new String[4];
        files[3] = "";
        System.arraycopy(args, 1, files, 0, args.length-1);


        for (String s : files)
            System.out.println("--- " + s);

        //if (files.length > 4)
        //    restrictionSiteFilename = files[4];

        //                 [min value, max value, window, resolution]
        Number[] optionalAPAFlags = parser.getAPAOptions();

        if (optionalAPAFlags[0] != null)
            min_peak_dist = optionalAPAFlags[0].doubleValue();

        if (optionalAPAFlags[1] != null)
            max_peak_dist = optionalAPAFlags[1].doubleValue();

        if (optionalAPAFlags[2] != null)
            window = optionalAPAFlags[2].intValue();

        if (optionalAPAFlags[3] != null)
            resolution = optionalAPAFlags[3].intValue();

    }

    @Override
    public void run() {

        //Calculate parameters that will need later
        int L = 2 * window + 1;
        Integer[] gwPeakNumbers = new Integer[3];
        for(int i = 0; i < gwPeakNumbers.length; i++)
            gwPeakNumbers[i] = 0;

        try {
            System.out.println("Accessing " + files[0]);
            DatasetReaderV2 reader = new DatasetReaderV2(files[0]);
            Dataset ds = reader.read();
            HiCGlobals.verifySupportedHiCFileVersion(reader.getVersion());

            // select zoom level closest to the requested one
            HiCZoom zoom = HiCFileTools.getZoomLevel(ds, resolution);
            resolution = zoom.getBinSize();



            List<Chromosome> chromosomes = ds.getChromosomes();
            LoopContainer loopContainer = LoopListParser.parseList(files[1], chromosomes,
                    min_peak_dist, max_peak_dist, resolution);
            Set<Chromosome> commonChromosomes = loopContainer.getCommonChromosomes(chromosomes);

            // Loop through chromosomes
            // TODO this should be rewritten to use chrIdx/Name instead of the whole chromosome
            // TODO the arraylists and maps don't need the entire chr class, just a unique identifier
            for (Chromosome chr : commonChromosomes) {
                APADataStack apaDataStack = new APADataStack(L, files[2], files[3]);

                if (chr.getName().equals(Globals.CHR_ALL)) continue;
                Matrix matrix = ds.getMatrix(chr, chr);
                if (matrix == null) continue;
                MatrixZoomData zd = matrix.getZoomData(zoom);

                Set<Feature2D> loops = loopContainer.getUniqueFilteredLoopList(chr);
                Integer[] peakNumbers = loopContainer.getUniqueFilteredLoopNumbers(chr);

                if(loops.size() != peakNumbers[0])
                    System.out.println("Error reading stat numbers fro "+chr);

                for(int i = 0; i < peakNumbers.length; i++){
                    System.out.println(chr+" "+i + " " + peakNumbers[i]+" "+gwPeakNumbers[i]);
                    gwPeakNumbers[i] += peakNumbers[i];
                }

                for (Feature2D loop : loops) {
                    RealMatrix newData = APAUtils.extractLocalizedData(zd, loop, L, resolution, window);
                    apaDataStack.addData(newData);
                }

                apaDataStack.updateGenomeWideData();
                if (saveAllData)
                    apaDataStack.exportDataSet(chr.getName(), peakNumbers);

            }
        } catch (IOException e) {
            System.out.println("Unable to extract APA data");
            e.printStackTrace();
            System.exit(-3);
        }


        APADataStack.exportGenomeWideData(gwPeakNumbers);

        System.exit(7);
    }
}