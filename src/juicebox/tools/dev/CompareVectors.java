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

package juicebox.tools.dev;

import juicebox.HiC;
import juicebox.data.ChromosomeHandler;
import juicebox.data.NormalizationVector;
import juicebox.data.basics.Chromosome;
import juicebox.tools.clt.CommandLineParser;
import juicebox.tools.clt.JuiceboxCLT;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.tools.utils.common.UNIXTools;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;

import java.io.File;

public class CompareVectors extends JuiceboxCLT {

    ChromosomeHandler chromosomeHandler;
    NormalizationType norm2;
    File outputFolder;
    HiCZoom zoom;

    public CompareVectors() {
        super(getUsage());
    }

    private static String getUsage() {
        return "compare-vectors <NONE/VC/VC_SQRT/KR/SCALE> <NONE/VC/VC_SQRT/KR/SCALE> <hicFile(s)> <binsize> <directory>";
    }

    @Override
    public void readArguments(String[] args, CommandLineParser parser) {
        if (args.length != 7 && args.length != 6) {
            printUsageAndExit();
        }

        setDatasetAndNorm(args[3], args[1], false);

        norm2 = dataset.getNormalizationHandler().getNormTypeFromString(args[2]);

        chromosomeHandler = dataset.getChromosomeHandler();

        try {
            zoom = new HiCZoom(HiC.Unit.BP, Integer.parseInt(args[4]));
        } catch (NumberFormatException e) {
            System.err.println("Integer expected for bin size.  Found: " + args[4] + ".");
            System.exit(21);
        }

        outputFolder = new File(args[5]);
        UNIXTools.makeDir(outputFolder);
    }

    @Override
    public void run() {

        for (Chromosome chromosome : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {
			NormalizationVector nv1 = dataset.getNormalizationVector(chromosome.getIndex(), zoom, norm);
			NormalizationVector nv2 = dataset.getNormalizationVector(chromosome.getIndex(), zoom, norm2);
			if (nv2 == null) {
				System.err.println(norm2 + " not available for " + chromosome.getName());
			}
	
			if (chromosome.getLength() / zoom.getBinSize() + 1 < Integer.MAX_VALUE) {
				int numElements = (int) (chromosome.getLength() / zoom.getBinSize()) + 1;
				double[][] result = new double[2][numElements];
		
				fillInVector(result, nv1, 0, norm, chromosome);
				fillInVector(result, nv2, 1, norm2, chromosome);
		
				File rOut = new File(outputFolder, chromosome.getName() + "_" + norm.getLabel() + "_vs_" + norm2.getLabel() + ".npy");
				MatrixTools.saveMatrixTextNumpy(rOut.getAbsolutePath(), result);
			} else {
				System.err.println("long vector support not currently available");
			}
        }
    }

    private void fillInVector(double[][] array, NormalizationVector nv, int rowIdx, NormalizationType normalizationType, Chromosome chromosome) {
        if (nv == null) {
            System.err.println(normalizationType + " not available for " + chromosome.getName());
        } else {
            System.arraycopy(nv.getData().getValues().get(0), 0, array[rowIdx], 0, array[rowIdx].length);
        }
    }
}
