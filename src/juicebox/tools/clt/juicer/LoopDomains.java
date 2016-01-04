/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2016 Broad Institute, Aiden Lab
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
import juicebox.HiCGlobals;
import juicebox.data.HiCFileTools;
import juicebox.tools.clt.CommandLineParserForJuicer;
import juicebox.tools.clt.JuicerCLT;
import juicebox.track.feature.*;
import org.broad.igv.feature.Chromosome;

import java.awt.*;
import java.util.*;
import java.util.List;

/**
 * Created by muhammadsaadshamim on 12/29/15.
 */
public class LoopDomains extends JuicerCLT {


    private static Color C_LOOP = new Color(34, 149, 34);
    private static Color C_CONTACT = new Color(0, 105, 0);
    private static Color C_LONE = new Color(102, 0, 153);

    /**
     * For every domain boundary, find the overlapping loop anchors.
     * this is just an intersection of domain boundary and loop anchor tracks,
     * but you'll need to give the domain boundaries a bit of width since there
     * are offsets and the exact boundary positions returned by arrowhead aren't
     * entirely reliable. There should be one or two loop anchors associated
     * with a domain boundary (two is the case where there are loops in
     * both directions)
     * <p/>
     * For every loop anchor at a domain boundary, find the motifs required to
     * disrupt the loop. this is (almost) returned by the motif finder
     * (if we had completed the enhancements). but in the case of a unique motif or
     * inferred motif, that should be returned; otherwise, the collection of motifs
     * should be returned. this gives the set of motifs required to disrupt the domain boundary.
     **/

    private int threshold = 25000;
    private String genomeID, loopListPath, domainListPath, outputPath = "loop_domains_list";
    private Set<String> givenChromosomes = null;

    /*
     * Assumes that the loop list provided already has associated motifs
     */
    public LoopDomains() {
        super("loop_domains [-m threshold] [-c chromosome(s)] <genomeID> " +
                "<loop_list_with_motifs> <contact_domains_list> [output_path]");
        HiCGlobals.useCache = false;
    }


    @Override
    public void readArguments(String[] args, CmdLineParser parser) {
        CommandLineParserForJuicer juicerParser = (CommandLineParserForJuicer) parser;

        if (args.length != 4 && args.length != 5) {
            printUsage();
        }

        genomeID = args[1];
        loopListPath = args[2];
        domainListPath = args[3];
        if (args.length == 5) {
            outputPath = args[4];
        }

        int specifiedMatrixSize = juicerParser.getMatrixSizeOption();
        if (specifiedMatrixSize > 0) {
            threshold = specifiedMatrixSize;
        }

        List<String> potentialChromosomes = juicerParser.getChromosomeOption();
        if (potentialChromosomes != null) {
            givenChromosomes = new HashSet<String>(potentialChromosomes);
        }
    }


    @Override
    public void run() {

        List<Chromosome> chromosomes = HiCFileTools.loadChromosomes(genomeID);
        if (givenChromosomes != null) {
            chromosomes = new ArrayList<Chromosome>(HiCFileTools.stringToChromosomes(givenChromosomes,
                    chromosomes));
        }

        // need to keep motifs for loop list
        final Feature2DList loopList = Feature2DParser.loadFeatures(loopListPath, chromosomes, true, null, true);
        // domains have to use the extended with motif class, but they'll just have null values for that
        Feature2DList domainList = Feature2DParser.loadFeatures(domainListPath, chromosomes, false, null, true);

        loopList.clearAllAttributes();
        domainList.clearAllAttributes();
        // only motifs are preserved due to usage of feature2DWithMotif class (the last boolean
        // parameter iin feature load line)

        // not necessary as size of feature makes it pretty obvious...
        loopList.addAttributeFieldToAll("Type", "Loop");
        domainList.addAttributeFieldToAll("Type", "Domain");

        final Feature2DList loopDomainList = new Feature2DList();

        domainList.processLists(new FeatureFunction() {
            @Override
            public void process(String chr, List<Feature2D> feature2DList) {
                List<Feature2D> domains = new ArrayList<Feature2D>(feature2DList);

                List<Feature2D> loops = new ArrayList<Feature2D>();
                if (loopList.containsKey(chr)) {
                    loops.addAll(loopList.get(chr));
                }


                Collections.sort(domains);
                Collections.sort(loops);

                for (Feature2D domain : domains) {
                    boolean domainHasNoLoops = true;
                    for (Feature2D loop : loops) {
                        // since we've sorted, avoid assessing loops too far away
                        if (Feature2DTools.loopIsUpstreamOfDomain(loop, domain, threshold)) {
                            continue;
                        }
                        if (Feature2DTools.loopIsDownstreamOfDomain(loop, domain, threshold)) {
                            break;// since list is sorted, all valid options should have been considered
                        }

                        if (Feature2DTools.domainContainsLoopWithinExpandedTolerance(loop, domain, threshold)) {
                            // inside larger box
                            if (!Feature2DTools.domainContainsLoopWithinExpandedTolerance(loop, domain, -threshold)) {
                                // not inside smaller box, i.e. along boundary
                                domainHasNoLoops = false;
                                loop.setColor(C_LOOP);
                                loopDomainList.addByKey(chr, loop);
                            }
                        }
                    }

                    if (domainHasNoLoops) {
                        domain.setColor(C_LONE);
                    } else {
                        domain.setColor(C_CONTACT);
                    }
                    loopDomainList.addByKey(chr, domain);
                }
            }
        });

        loopDomainList.exportFeatureList(outputPath, false, "NA");
    }

}
