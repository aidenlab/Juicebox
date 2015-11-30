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
import juicebox.data.HiCFileTools;
import juicebox.data.anchor.MotifAnchor;
import juicebox.data.anchor.MotifAnchorParser;
import juicebox.data.anchor.MotifAnchorTools;
import juicebox.data.feature.GenomeWideList;
import juicebox.tools.clt.JuicerCLT;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.Feature2DParser;
import org.broad.igv.feature.Chromosome;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by muhammadsaadshamim on 9/4/15.
 *
 */
public class MotifFinder extends JuicerCLT {

    private String outputPath;
    private String loopListPath;
    private String genomeID;
    private List<String> proteinsForUniqueMotifPaths, proteinsForInferredMotifPaths;
    private String bedFileDirPath;
    private String globalMotifListPath;

    public MotifFinder() {
        super("motifs <genomeID> <bed_file_dir> <looplist> [custom_global_motif_list]");
        MotifAnchor.uniquenessShouldSupercedeConvergentRule = false;
    }

    @Override
    public void readArguments(String[] args, CmdLineParser parser) {

        if (args.length != 4 && args.length != 5) {
            this.printUsage();
        }

        int i = 1;
        genomeID = args[i++];
        bedFileDirPath = args[i++];
        //proteinsForUniqueMotifPaths = args[i++].split(",");
        //proteinsForInferredMotifPaths = args[i++].split(",");
        loopListPath = args[i++];
        if (args.length == 5) {
            globalMotifListPath = args[i++];
        }

        if (loopListPath.endsWith(".txt")) {
            outputPath = loopListPath.substring(0, loopListPath.length() - 4) + "_with_motifs.txt";
        } else {
            outputPath = loopListPath + "_with_motifs.txt";
        }

        try {
            retrieveAllBEDFiles(bedFileDirPath);
        } catch (Exception e) {
            System.err.println("Unable to locate BED files");
            System.err.println("All BED files should include the '.bed' extension");
            System.err.println("BED files for locating unique motifs should be located in given_bed_file_dir/unique");
            System.err.println("BED files for locating inferred motifs should be located in given_bed_file_dir/inferred");
            //e.printStackTrace();
            System.exit(-4);
        }

    }

    @Override
    public void run() {

        //System.out.println("Found X3 " + MotifAnchorTools.searchForFeatureWithin(10, 22230000,	22240000, anchors));
        //System.out.println("Found Y1 " + MotifAnchorTools.searchForFeature(10, 4891921, 4892143, proteinsForInference));
        //Feature2D f00 = features.searchForFeature(10	,4890000,	4900000,	10,	5320000,	5330000);
        //System.out.println("Found1 " + MotifAnchorTools.searchForFeature(10, "ATGACCAACAAGGGTCGCCA", globalAnchors));


        List<Chromosome> chromosomes = HiCFileTools.loadChromosomes(genomeID);

        // anchors from loops
        Feature2DList features = Feature2DParser.loadFeatures(loopListPath, chromosomes, true, null, true);
        GenomeWideList<MotifAnchor> anchors = MotifAnchorTools.extractAnchorsFromFeatures(features, false);
        //GenomeWideList<MotifAnchor> anchors2 = MotifAnchorTools.extractAnchorsFromFeatures(features, false);

        // global anchors
        GenomeWideList<MotifAnchor> globalAnchors = loadMotifs(chromosomes);

        // intersect all the 1d tracks for unique motifs and get appropriate global anchors
        GenomeWideList<MotifAnchor> proteinsForUniqueness = getIntersectionOfBEDFiles(chromosomes, proteinsForUniqueMotifPaths);

        // try1
        MotifAnchorTools.intersectLists(globalAnchors, proteinsForUniqueness, true);
        MotifAnchorTools.intersectLists(anchors, globalAnchors, true);
        MotifAnchorTools.updateOriginalFeatures(anchors, true);

        //try 2
        //globalAnchors = loadMotifs(chromosomes);
        //anchors = MotifAnchorTools.extractAnchorsFromFeatures(features, false);
        //MotifAnchorTools.preservativeIntersectLists(anchors, proteinsForUniqueness, false);
        //MotifAnchorTools.intersectLists(anchors, globalAnchors, true);
        //MotifAnchorTools.updateOriginalFeatures(anchors, true);

        //System.out.println("Found1 " + MotifAnchorTools.searchForFeature(10, "GCTGGCAGGAGAGGCCGGCC", anchors).getOriginalFeatures2());
        //System.out.println("Found1 " + MotifAnchorTools.searchForFeature(10, "GCTGGCAGGAGAGGCCGGCC", anchors).getScore());

        //System.out.println("Found1 " + MotifAnchorTools.searchForFeature(10, "GTCACCACAAGATGGCCCCA", anchors).getOriginalFeatures2());
        //System.out.println("Found1 " + MotifAnchorTools.searchForFeature(10, "GTCACCACAAGATGGCCCCA", anchors).getScore());


        // reload motif list
        anchors = MotifAnchorTools.extractAnchorsFromFeatures(features, true);
        //anchors2 = MotifAnchorTools.extractAnchorsFromFeatures(features, true);

        // intersect all the 1d tracks for inferring motifs
        globalAnchors = loadMotifs(chromosomes);
        GenomeWideList<MotifAnchor> proteinsForInference = getIntersectionOfBEDFiles(chromosomes, proteinsForInferredMotifPaths);
        MotifAnchorTools.intersectLists(globalAnchors, proteinsForInference, true);
        MotifAnchorTools.intersectLists(anchors, globalAnchors, true);
        MotifAnchorTools.updateOriginalFeatures(anchors, false);

        // try2
        //globalAnchors = loadMotifs(chromosomes);
        //MotifAnchorTools.preservativeIntersectLists(anchors2, proteinsForInference, false);
        //MotifAnchorTools.intersectLists(anchors2, globalAnchors, true);
        //MotifAnchorTools.updateOriginalFeatures(anchors2, true);

        features.exportFeatureList(outputPath, false);
        System.out.println("Motif Finder complete");
    }

    private GenomeWideList<MotifAnchor> loadMotifs(List<Chromosome> chromosomes) {
        GenomeWideList<MotifAnchor> anchors;
        if (globalMotifListPath == null || globalMotifListPath.length() < 1) {
            anchors = MotifAnchorParser.loadGlobalMotifs(genomeID, chromosomes);
        } else {
            anchors = MotifAnchorParser.loadMotifs(globalMotifListPath, chromosomes, null);
        }
        return anchors;
    }


    private void retrieveAllBEDFiles(String path) throws IOException {
        File bedFileDir = new File(path);
        if (bedFileDir.exists()) {
            String uniqueBEDFilesPath = path + "/unique";
            String inferredBEDFilesPath = path + "/inferred";

            // if the '/' was already included
            if (path.endsWith("/")) {
                uniqueBEDFilesPath = path + "unique";
                inferredBEDFilesPath = path + "inferred";
            }

            proteinsForUniqueMotifPaths = retrieveBEDFilesByExtensionInFolder(uniqueBEDFilesPath, "Unique");
            proteinsForInferredMotifPaths = retrieveBEDFilesByExtensionInFolder(inferredBEDFilesPath, "Inferred");
        } else {
            throw new IOException("BED files directory not valid");
        }
    }

    private List<String> retrieveBEDFilesByExtensionInFolder(String directoryPath, String description) throws IOException {

        List<String> bedFiles = new ArrayList<String>();

        File folder = new File(directoryPath);
        File[] listOfFiles = folder.listFiles();

        for (File file : listOfFiles) {
            if (file.isFile()) {
                String path = file.getAbsolutePath();
                if (path.endsWith(".bed")) {
                    bedFiles.add(path);
                }
            }
        }

        if (bedFiles.size() < 1) {
            throw new IOException(description + " BED files not found");
        }

        return bedFiles;
    }

    private GenomeWideList<MotifAnchor> getIntersectionOfBEDFiles(List<Chromosome> chromosomes, List<String> bedFiles) {
        GenomeWideList<MotifAnchor> proteins = MotifAnchorParser.loadFromBEDFile(chromosomes, bedFiles.get(0));
        for (int i = 1; i < bedFiles.size(); i++) {
            GenomeWideList<MotifAnchor> nextProteinList = MotifAnchorParser.loadFromBEDFile(chromosomes, bedFiles.get(i));
            MotifAnchorTools.intersectLists(proteins, nextProteinList, false);
        }
        return proteins;
    }
}
