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

package juicebox.tools.clt.juicer;

import jargs.gnu.CmdLineParser;
import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.data.Dataset;
import juicebox.data.ExpectedValueFunction;
import juicebox.data.HiCFileTools;
import juicebox.data.Matrix;
import juicebox.tools.clt.CommandLineParserForJuicer;
import juicebox.tools.clt.JuicerCLT;
import juicebox.tools.utils.juicer.arrowhead.ArrowheadScoreList;
import juicebox.tools.utils.juicer.arrowhead.BlockBuster;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.Feature2DParser;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import org.broad.igv.Globals;
import org.broad.igv.feature.Chromosome;

import java.io.PrintWriter;
import java.util.*;

/**
 * Arrowhead
 * <p/>
 * Developed by Neva Durand
 * Implemented by Muhammad Shamim
 * <p/>
 * -------
 * Arrowhead
 * -------
 * <p/>
 * arrowhead [-c chromosome(s)] [-m matrix size] <NONE/VC/VC_SQRT/KR> <input_HiC_file(s)> <output_file>
 * <resolution> [feature_list] [control_list]
 * *
 * The required arguments are:
 * <p/>
 * <NONE/VC/VC_SQRT/KR> One of the normalizations must be selected (case sensitive). Generally, KR (Knight-Ruiz)
 * balancing should be used.
 * <p/>
 * <input_HiC_file(s)>: Address of HiC File(s) which should end with .hic.  This is the file you will
 * load into Juicebox. URLs or local addresses may be used. To sum multiple HiC Files together,
 * use the '+' symbol between the addresses (no whitespace between addresses).
 * <p/>
 * <output_file>: Final list of all contact domains found by Arrowhead. Can be visualized directly in Juicebox
 * as a 2D annotation.
 * <p/>
 * <resolution>: Integer resolution for which Arrowhead will be run. Generally, 5kB (5000) or 10kB (10000)
 * resolution is used depending on the depth of sequencing in the hic file(s).
 * <p/>
 * -- NOTE -- If you want to find scores for a feature and control list, both must be provided:
 * <p/>
 * [feature_list]: Feature list of loops/domains for which block scores are to be calculated
 * <p/>
 * [control_list]: Control list of loops/domains for which block scores are to be calculated
 * <p/>
 * <p/>
 * The optional arguments are:
 * <p/>
 * -m <int> Size of the sliding window along the diagonal in which contact domains will be found. Must be an even
 * number as (m/2) is used as the increment for the sliding window. (Default 2000)
 * <p/>
 * -c <String(s)> Chromosome(s) on which Arrowhead will be run. The number/letter for the chromosome can be used with or
 * without appending the "chr" string. Multiple chromosomes can be specified using commas (e.g. 1,chr2,X,chrY)
 * <p/>
 * <p/>
 * ----------------
 * Arrowhead Examples
 * ----------------
 * <p/>
 * arrowhead -m 2000 KR ch12-lx-b-lymphoblasts_mapq_30.hic contact_domains_list 10000
 * This command will run Arrowhead on a mouse cell line HiC map and save all contact domains to the
 * contact_domains_list file. These are the settings used to generate the official contact domain list on the
 * ch12-lx-b-lymphoblast cell line.
 * <p/>
 * arrowhead KR GM12878_mapq_30.hic contact_domains_list 5000
 * This command will run Arrowhead on the GM12878 HiC map and save all contact domains to the contact_domains_list
 * file. These are the settings used to generate the official GM12878 contact domain list.
 */
public class Arrowhead extends JuicerCLT {

    private static int matrixSize = 2000;
    private boolean configurationsSetByUser = false;
    private Set<String> givenChromosomes = null;
    private boolean controlAndListProvided = false;
    private String featureList, controlList;

    // must be passed via command line
    private int resolution = 10000;
    private String file, outputPath;

    public Arrowhead() {
        super("arrowhead [-c chromosome(s)] [-m matrix size] [-r resolution] [-k normalization (NONE/VC/VC_SQRT/KR)] " +
                "<HiC file(s)> <output_file> [feature_list] [control_list]");
        HiCGlobals.useCache = false;
    }

    @Override
    public void readArguments(String[] args, CmdLineParser parser) {

        CommandLineParserForJuicer juicerParser = (CommandLineParserForJuicer) parser;
        if (args.length != 3 && args.length != 5) {
            // 3 - standard, 5 - when list/control provided
            printUsage();
            System.exit(0);
        }

        NormalizationType preferredNorm = juicerParser.getNormalizationTypeOption();
        if (preferredNorm != null)
            norm = preferredNorm;

        file = args[1];
        outputPath = args[2];

        List<String> potentialResolution = juicerParser.getMultipleResolutionOptions();
        if (potentialResolution != null) {
            resolution = Integer.parseInt(potentialResolution.get(0));
            configurationsSetByUser = true;
        }

        if (args.length == 5) {
            controlAndListProvided = true;
            featureList = args[3];
            controlList = args[4];
        }

        List<String> potentialChromosomes = juicerParser.getChromosomeOption();
        if (potentialChromosomes != null) {
            givenChromosomes = new HashSet<String>(potentialChromosomes);
        }

        int specifiedMatrixSize = juicerParser.getMatrixSizeOption();
        if (specifiedMatrixSize > 1) {
            matrixSize = specifiedMatrixSize;
        }
    }


    @Override
    public void run() {

        Dataset ds = HiCFileTools.extractDatasetForCLT(Arrays.asList(file.split("\\+")), true);

        final ExpectedValueFunction df = ds.getExpectedValues(new HiCZoom(HiC.Unit.BP, 2500000), NormalizationType.NONE);
        double firstExpected = df.getExpectedValues()[0]; // expected value on diagonal
        // From empirical testing, if the expected value on diagonal at 2.5Mb is >= 100,000
        // then the map had more than 300M contacts.
        // If map has less than 300M contacts, we will not run Arrowhead or HiCCUPs
        if (firstExpected < 100000) {
            System.err.println("HiC contact map is too sparse to run Arrowhead, exiting.");
            System.exit(0);
        }

        // high quality (IMR90, GM12878) maps have different settings
        if (!configurationsSetByUser) {
            matrixSize = 2000;
            if (firstExpected > 250000) {
                resolution = 5000;
                System.out.println("Default settings for 5kb being used");
            } else {
                resolution = 10000;
                System.out.println("Default settings for 10kb being used");
            }
        }

        List<Chromosome> chromosomes = ds.getChromosomes();

        Feature2DList contactDomainsGenomeWide = new Feature2DList();
        Feature2DList contactDomainListScoresGenomeWide = new Feature2DList();
        Feature2DList contactDomainControlScoresGenomeWide = new Feature2DList();

        Feature2DList inputList = new Feature2DList();
        Feature2DList inputControl = new Feature2DList();
        if (controlAndListProvided) {
            inputList.add(Feature2DParser.loadFeatures(featureList, chromosomes, true, null, false));
            inputControl.add(Feature2DParser.loadFeatures(controlList, chromosomes, true, null, false));
        }

        PrintWriter outputBlockFile = HiCFileTools.openWriter(outputPath + "_" + resolution + "_blocks");
        PrintWriter outputListFile = null;
        PrintWriter outputControlFile = null;
        if (controlAndListProvided) {
            outputListFile = HiCFileTools.openWriter(outputPath + "_" + resolution + "_list_scores");
            outputControlFile = HiCFileTools.openWriter(outputPath + "_" + resolution + "_control_scores");
        }

        // chromosome filtering must be done after input/control created
        // because full set of chromosomes required to parse lists
        if (givenChromosomes != null)
            chromosomes = new ArrayList<Chromosome>(HiCFileTools.stringToChromosomes(givenChromosomes,
                    chromosomes));

        HiCZoom zoom = new HiCZoom(HiC.Unit.BP, resolution);

        double maxProgressStatus = determineHowManyChromosomesWillActuallyRun(ds, chromosomes);
        int currentProgressStatus = 0;

        for (Chromosome chr : chromosomes) {

            if (chr.getName().equals(Globals.CHR_ALL)) continue;

            Matrix matrix = ds.getMatrix(chr, chr);
            if (matrix == null) continue;

            ArrowheadScoreList list = new ArrowheadScoreList(inputList.get(chr.getIndex(), chr.getIndex()), resolution);
            ArrowheadScoreList control = new ArrowheadScoreList(inputControl.get(chr.getIndex(), chr.getIndex()), resolution);

            if (HiCGlobals.printVerboseComments) {
                System.out.println("\nProcessing " + chr.getName());
            }

            // actual Arrowhead algorithm
            BlockBuster.run(chr.getIndex(), chr.getName(), chr.getLength(), resolution, matrixSize,
                    matrix.getZoomData(zoom), norm, list, control, contactDomainsGenomeWide,
                    contactDomainListScoresGenomeWide, contactDomainControlScoresGenomeWide);

            System.out.println(((int) Math.floor((100.0 * ++currentProgressStatus) / maxProgressStatus)) + "% ");
        }

        // save the data on local machine
        contactDomainsGenomeWide.exportFeatureList(outputBlockFile, true, "arrowhead");
        System.out.println(contactDomainsGenomeWide.getNumTotalFeatures() + " domains written to file: " +
                outputPath + "_" + resolution + "_blocks");
        if (controlAndListProvided) {
            contactDomainListScoresGenomeWide.exportFeatureList(outputListFile, false, "NA");
            contactDomainControlScoresGenomeWide.exportFeatureList(outputControlFile, false, "NA");
        }
        System.out.println("Arrowhead complete");
    }
}