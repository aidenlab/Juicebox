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

import juicebox.data.ChromosomeHandler;
import juicebox.data.HiCFileTools;
import juicebox.data.anchor.*;
import juicebox.data.feature.FeatureFunction;
import juicebox.data.feature.GenomeWideList;
import juicebox.tools.clt.CommandLineParserForJuicer;
import juicebox.tools.clt.JuicerCLT;
import juicebox.tools.utils.juicer.GeneTools;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.Feature2DParser;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Created by muhammadsaadshamim on 8/3/16.
 */
public class GeneFinder extends JuicerCLT {

    private String genomeID, bedFilePath, loopListPath;
    private File outFile;

    public GeneFinder() {
        super("genes <genomeID> <bed_file> <looplist> [output]");
    }

    @Override
    protected void readJuicerArguments(String[] args, CommandLineParserForJuicer juicerParser) {
        if (args.length != 3 && args.length != 4 && args.length != 5) {
            printUsageAndExit();
        }
        genomeID = args[1];
        bedFilePath = args[2];
        if (args.length >= 4) loopListPath = args[3];

        String outputPath = "active_genes";
        if (args.length == 5) {
            outputPath = args[4];
        }
        outFile = new File(outputPath);

    }

    @Override
    public void run() {
        ChromosomeHandler handler = HiCFileTools.loadChromosomes(genomeID);


        try {

            GenomeWideList<GenericLocus> genes = GeneTools.parseGenome(genomeID, handler);
            System.out.println("Starting with " + genes.size() + " genes");

            if (loopListPath != null) {
                final Feature2DList allLoops = Feature2DParser.loadFeatures(loopListPath, handler, false, null, false);
                GenomeWideList<GenericLocus> allAnchors = MotifAnchorTools.extractAnchorsFromIntrachromosomalFeatures(allLoops, false, handler, 15000
                );
                final Feature2DList filteredLoops = new Feature2DList();

                if ((new File(bedFilePath)).exists()) {
                    GenomeWideList<GenericLocus> proteins = GenericLocusParser.loadFromBEDFile(handler, bedFilePath);
                    MotifAnchorTools.preservativeIntersectLists(allAnchors, proteins, false);

                    allAnchors.processLists(new FeatureFunction<GenericLocus>() {
                        @Override
                        public void process(String chr, List<GenericLocus> anchors) {
                            List<Feature2D> restoredLoops = new ArrayList<>();
                            for (GenericLocus anchor : anchors) {
                                restoredLoops.addAll(anchor.getOriginalFeatures1());
                                restoredLoops.addAll(anchor.getOriginalFeatures2());
                            }

                            filteredLoops.addByKey(chr + "_" + chr, restoredLoops);
                        }
                    });
                } else {
                    System.err.println("No bed file provided, all loops being assessed.");
                    filteredLoops.add(allLoops);
                }


                // note, this is NOT identical to all anchors after preservative intersect
                // because this restores both of the loops anchors even if one was eliminated
                // in the previous intersection as long as one of its anchors hit the protein
                GenomeWideList<GenericLocus> filteredAnchors = MotifAnchorTools.extractAnchorsFromIntrachromosomalFeatures(filteredLoops, false, handler, 15000);
                MotifAnchorTools.preservativeIntersectLists(genes, filteredAnchors, false);

            } else {
                if ((new File(bedFilePath)).exists()) {
                    System.out.println("Just using bed file");
                    GenomeWideList<GenericLocus> proteins = GenericLocusParser.loadFromBEDFile(handler, bedFilePath);
                    //MotifAnchorTools.mergeAndExpandSmallAnchors(proteins, 1000);
                    MotifAnchorTools.preservativeIntersectLists(genes, proteins, false);
                }
            }

            final Set<String> geneNames = new HashSet<>();
            genes.processLists(new FeatureFunction<GenericLocus>() {
                @Override
                public void process(String chr, List<GenericLocus> featureList) {
                    for (GenericLocus anchor : featureList) {
                        geneNames.add(anchor.getName());
                    }
                }
            });

            BufferedWriter writer = new BufferedWriter(new FileWriter(outFile));
            StringBuilder sb = new StringBuilder();
            for (String s : geneNames) {
                sb.append(s);
                sb.append("\n");
            }
            writer.write(sb.toString());
            writer.flush();
            writer.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
