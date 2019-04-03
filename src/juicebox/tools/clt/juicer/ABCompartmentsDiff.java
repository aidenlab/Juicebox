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

package juicebox.tools.clt.juicer;

import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.data.ChromosomeHandler;
import juicebox.data.Dataset;
import juicebox.data.HiCFileTools;
import juicebox.tools.clt.CommandLineParserForJuicer;
import juicebox.tools.clt.JuicerCLT;
import juicebox.tools.utils.common.ArrayTools;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import org.broad.igv.feature.Chromosome;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;

/**
 * Created by muhammadsaadshamim on 6/2/16.
 */
public class ABCompartmentsDiff extends JuicerCLT {

    private final HiCZoom highZoom = new HiCZoom(HiC.Unit.BP, 500000);
    private ChromosomeHandler chromosomeHandler;
    private Dataset ds1, ds2;
    private PrintWriter diffFileWriter, simFileWriter;

    public ABCompartmentsDiff() {
        super("ab_compdiff [-c chromosome(s)] <firstHicFile> <secondHicFile> <outputDirectory>");

    }

    @Override
    protected void readJuicerArguments(String[] args, CommandLineParserForJuicer juicerParser) {
        if (args.length != 4) {
            printUsageAndExit();
        }

        File outputDirectory = HiCFileTools.createValidDirectory(args[3]);
        File diffFile = new File(outputDirectory, "diff_AB_compartments.wig");
        File simFile = new File(outputDirectory, "similar_AB_compartments.wig");

        try {
            diffFileWriter = new PrintWriter(diffFile);
            simFileWriter = new PrintWriter(simFile);
        } catch (IOException e) {
            System.err.println("Unable to create files in output directory");
            System.exit(1);
        }

        ds1 = HiCFileTools.extractDatasetForCLT(Arrays.asList(args[1].split("\\+")), true);
        ds2 = HiCFileTools.extractDatasetForCLT(Arrays.asList(args[2].split("\\+")), true);

        if (!(ds1.getGenomeId().equals(ds2.getGenomeId()))) {
            System.err.println("Hi-C maps must be from the same genome");
            System.exit(2);
        }
        chromosomeHandler = ds1.getChromosomeHandler();

        if (givenChromosomes != null)
            chromosomeHandler = HiCFileTools.stringToChromosomes(givenChromosomes, chromosomeHandler);

        NormalizationType preferredNorm = juicerParser.getNormalizationTypeOption(ds1.getNormalizationHandler());
        if (preferredNorm != null)
            norm = preferredNorm;

        System.out.println("Running differential A/B compartments at resolution " + highZoom.getBinSize());
    }

    @Override
    public void run() {

        double maxProgressStatus = determineHowManyChromosomesWillActuallyRun(ds1, chromosomeHandler);
        int currentProgressStatus = 0;

        for (Chromosome chromosome : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {

            if (HiCGlobals.printVerboseComments) {
                System.out.println("\nProcessing " + chromosome.getName());
            }

            double[] eigenvector1, eigenvector2;
            try {
                eigenvector1 = ds1.getEigenvector(chromosome, highZoom, 0, norm);
                eigenvector2 = ds2.getEigenvector(chromosome, highZoom, 0, norm);
            } catch (Exception e) {
                System.err.println("Unable to get eigenvector for " + chromosome);
                continue;
            }

            int n = eigenvector1.length, n2 = eigenvector2.length;
            if (n != n2) {
                System.err.println(chromosome + " eigenvector lengths do not match: L1=" + n + " L2=" + n2);
                n = Math.min(n, n2);
                System.err.println("Using length " + n);
            }

            // first determine orientation (sign) of control eigenvector relative to observed
            // eigenvectors can be multiplied by any scalar and remain an eigenvector of the matrix
            // default sign is arbitrary
            int scalarMultipleForControlEigenvector = 1;
            int numSimilarities = 0, numDifferences = 0;
            for (int i = 0; i < n; i++) {
                double a = eigenvector1[i];
                double b = eigenvector2[i];

                if ((a > 0 && b > 0) || (a < 0 && b < 0)) {
                    numSimilarities++;
                } else if ((a > 0 && b < 0) || (a < 0 && b > 0)) {
                    numDifferences++;
                }
            }

            if (numDifferences > numSimilarities) {
                // unlikely outcome unless vastly different species are being studied
                // ctrl_eigenvector probably should be multiplied by -1
                scalarMultipleForControlEigenvector = -1;
            }
            if (HiCGlobals.printVerboseComments) {
                System.out.println("\nScalar " + scalarMultipleForControlEigenvector);
            }

            // Now actually find the differences
            double[] differencesA2B = new double[n];
            double[] similaritiesA2B = new double[n];
            for (int i = 0; i < n; i++) {
                double a = eigenvector1[i];
                double b = scalarMultipleForControlEigenvector * eigenvector2[i];

                if ((a > 0 && b > 0) || (a < 0 && b < 0)) {
                    similaritiesA2B[i] = Math.copySign(a - b, a);
                } else if ((a > 0 && b < 0) || (a < 0 && b > 0)) {
                    differencesA2B[i] = Math.copySign(a - b, a);
                }
            }

            ArrayTools.exportChr1DArrayToWigFormat(differencesA2B, diffFileWriter, chromosome.getName(), highZoom.getBinSize());
            ArrayTools.exportChr1DArrayToWigFormat(similaritiesA2B, simFileWriter, chromosome.getName(), highZoom.getBinSize());
            System.out.println(((int) Math.floor((100.0 * ++currentProgressStatus) / maxProgressStatus)) + "% ");
        }

        diffFileWriter.close();
        simFileWriter.close();

        System.out.println("Differential A/B Compartments Complete");
    }
}