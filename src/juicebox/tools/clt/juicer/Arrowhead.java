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

import java.util.*;

/**
 * HiC Computational Unbiased Peak Search
 *
 * Developed by Neva Durand
 * Implemented by Muhammad Shamim
 *
 * -------
 * Arrowhead
 * -------
 *
 * arrowhead [-c chromosome(s)] [-m matrix size] <NONE/VC/VC_SQRT/KR> <input_HiC_file(s)> <output_file>
 *   <resolution> [feature_list] [control_list]
 **
 * The required arguments are:
 *
 * <NONE/VC/VC_SQRT/KR> One of the normalizations must be selected (case sensitive). Generally, KR (Knight-Ruiz)
 *   balancing should be used.
 *
 * <input_HiC_file(s)>: Address of HiC File(s) which should end with .hic.  This is the file you will
 *   load into Juicebox. URLs or local addresses may be used. To sum multiple HiC Files together,
 *   use the '+' symbol between the addresses (no whitespace between addresses).
 *
 * <output_file>: Final list of all contact domains found by Arrowhead. Can be visualized directly in Juicebox
 *   as a 2D annotation.
 *
 * <resolution>: Integer resolution for which Arrowhead will be run. Generally, 5kB (5000) or 10kB (10000)
 *   resolution is used depending on the depth of sequencing in the hic file(s).
 *
 *   -- NOTE -- If you want to find scores for a feature and control list, both must be provided:
 *
 * [feature_list]: Feature list of loops/domains for which block scores are to be calculated
 *
 * [control_list]: Control list of loops/domains for which block scores are to be calculated
 *
 *
 * The optional arguments are:
 *
 * -m <int> Size of the sliding window along the diagonal in which contact domains will be found. Must be an even
 *   number as (m/2) is used as the increment for the sliding window. (Default 2000)
 *
 * -c <String(s)> Chromosome(s) on which Arrowhead will be run. The number/letter for the chromosome can be used with or
 *   without appending the "chr" string. Multiple chromosomes can be specified using commas (e.g. 1,chr2,X,chrY)
 *
 *
 * ----------------
 * Arrowhead Examples
 * ----------------
 *
 * arrowhead -m 2000 KR ch12-lx-b-lymphoblasts_mapq_30.hic contact_domains_list 10000
 *   This command will run Arrowhead on a mouse cell line HiC map and save all contact domains to the
 *   contact_domains_list file. These are the settings used to generate the official contact domain list on the
 *   ch12-lx-b-lymphoblast cell line.
 *
 * arrowhead KR GM12878_mapq_30.hic contact_domains_list 5000
 *   This command will run Arrowhead on the GM12878 HiC map and save all contact domains to the contact_domains_list
 *   file. These are the settings used to generate the official GM12878 contact domain list.
 *
 */
public class Arrowhead extends JuicerCLT {

    private static int matrixSize = 2000;
    private NormalizationType norm = NormalizationType.KR;
    private Set<String> givenChromosomes = null;
    private boolean controlAndListProvided = false;
    private String featureList, controlList;

    // must be passed via command line
    private int resolution = -100;
    private String file, outputPath;

    public Arrowhead() {
        super("arrowhead [-c chromosome(s)] [-m matrix size] <NONE/VC/VC_SQRT/KR> <input_HiC_file(s)> <output_file> " +
                "<resolution> [feature_list] [control_list]");
        HiCGlobals.useCache = false;
    }

    @Override
    public void readArguments(String[] args, CmdLineParser parser) {

        CommandLineParserForJuicer juicerParser = (CommandLineParserForJuicer) parser;
        if (args.length != 5 && args.length != 7) {
            // 5 - standard, 7 - when list/control provided
            printUsage();
        }

        norm = retrieveNormalization(args[1]);
        file = args[2];
        outputPath = args[3];

        try {
            resolution = Integer.valueOf(args[4]);
        } catch (NumberFormatException error) {
            printUsage();
        }

        if (args.length == 7) {
            controlAndListProvided = true;
            featureList = args[5];
            controlList = args[6];
        }

        List<String> potentialChromosomes = juicerParser.getChromosomeOption();
        if (potentialChromosomes != null)
            givenChromosomes = new HashSet<String>(potentialChromosomes);
        int specifiedMatrixSize = juicerParser.getMatrixSizeOption();
        if (specifiedMatrixSize % 2 == 1)
            specifiedMatrixSize += 1;
        if (specifiedMatrixSize > 50)
            matrixSize = specifiedMatrixSize;

    }


    @Override
    public void run() {

        Dataset ds = HiCFileTools.extractDatasetForCLT(Arrays.asList(file.split("\\+")), true);

        List<Chromosome> chromosomes = ds.getChromosomes();

        Feature2DList contactDomainsGenomeWide = new Feature2DList();
        Feature2DList contactDomainListScoresGenomeWide = new Feature2DList();
        Feature2DList contactDomainControlScoresGenomeWide = new Feature2DList();

        Feature2DList inputList = new Feature2DList();
        Feature2DList inputControl = new Feature2DList();
        if (controlAndListProvided) {
            inputList.add(Feature2DParser.loadFeatures(featureList, chromosomes, true, null));
            inputControl.add(Feature2DParser.loadFeatures(controlList, chromosomes, true, null));
        }

        // chromosome filtering must be done after input/control created
        // because full set of chromosomes required to parse lists
        if (givenChromosomes != null)
            chromosomes = new ArrayList<Chromosome>(HiCFileTools.stringToChromosomes(givenChromosomes,
                    chromosomes));

        HiCZoom zoom = new HiCZoom(HiC.Unit.BP, resolution);

        for (Chromosome chr : chromosomes) {

            if (chr.getName().equals(Globals.CHR_ALL)) continue;

            Matrix matrix = ds.getMatrix(chr, chr);
            if (matrix == null) continue;

            ArrowheadScoreList list = new ArrowheadScoreList(inputList.get(chr.getIndex(), chr.getIndex()), resolution);
            ArrowheadScoreList control = new ArrowheadScoreList(inputControl.get(chr.getIndex(), chr.getIndex()), resolution);

            // actual Arrowhead algorithm
            System.out.println("\nProcessing " + chr.getName());
            BlockBuster.run(chr.getIndex(), chr.getName(), chr.getLength(), resolution, matrixSize,
                    matrix.getZoomData(zoom), norm, list, control, contactDomainsGenomeWide,
                    contactDomainListScoresGenomeWide, contactDomainControlScoresGenomeWide);
        }

        // save the data on local machine
        contactDomainsGenomeWide.exportFeatureList(outputPath + "_" + resolution + "_blocks", false);
        contactDomainListScoresGenomeWide.exportFeatureList(outputPath + "_" + resolution + "_list_scores", false);
        contactDomainControlScoresGenomeWide.exportFeatureList(outputPath + "_" + resolution + "_control_scores", false);
    }
}