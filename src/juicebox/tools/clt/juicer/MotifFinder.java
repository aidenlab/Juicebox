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
import juicebox.tools.clt.JuicerCLT;
import juicebox.track.anchor.AnchorList;
import juicebox.track.anchor.AnchorParser;
import juicebox.track.anchor.AnchorTools;
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
 * the user provides (i) a loop list (ii) any number of 1-d peak tracks
 * for use in uniqueness (iii) any number of 1-d peak tracks for use in
 * inferring (iv) a genomewide list of motifs (i.e. our genomewide list
 * of motifs or their own).
 *
 * first step: all the 1-d peak tracks provided in (ii) are intersected.
 *
 * second step: peak loci that have only one 1-d peak from the intersected 1-d
 * peak track are identified (along with their corresponding unique 1-d peak)
 *
 * third step: the best motif match is identified by intersecting unique 1-d peaks
 * and the genome wide list of motifs. This gives a mapping of peak loci to unique motifs
 * (in the final loop list format, these motifs are outputted as 'u')
 *
 * fourth step: the 1-d peak tracks provided in (iii) are intersected.
 *
 * fifth step: the 1-d peak track from step 4 are intersected with the genomewide
 * motif list (best motif match) and split into a forward motif track and a reverse motif track.
 *
 * sixth step: upstream peak loci that did not have a unique motif are intersected
 * with the forward motif track from step 5, and for each peak locus if the peak
 * locus has only one forward motif, that is an inferred mapping (these motifs
 * are outputted as 'i'). downstream peak loci that did not have a unique motif
 * are intersected with the reverse motif track from step 5, and for each peak
 * locus if the peak locus has only one reverse motif, that is an inferred mapping
 * (these motifs are outputted as 'i'). Peak loci that form loops in both directions are ignored.
 *
 * the final output is the original loop list + information about the
 * motifs under each of the anchors (i.e. GEO format).
 *
 * Let me know if you have questions. I believe that these steps
 * handle things in the way that we handled them for the Dec paper
 * with the most reasonable and logical user inputs. the reason
 * that both (ii) and (iii) are necessary is because in December
 * we identified unique motifs with as much data as possible
 * (CTCF+RAD21+SMC3) but inferred motifs using only CTCF.
 *
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

        List<Chromosome> chromosomes = HiCFileTools.loadChromosomes(genomeID);

        AnchorList globalAnchors;
        if (globalMotifListPath == null || globalMotifListPath.length() < 1) {
            globalAnchors = AnchorParser.loadGlobalMotifs(genomeID, chromosomes);
        } else {
            globalAnchors = AnchorParser.loadMotifs(globalMotifListPath, chromosomes, null);
        }

        // 1st step - intersect all the 1d tracks for unique motifs
        AnchorList proteinsForUniqueness = getIntersectionOfBEDFiles(chromosomes, proteinsForUniqueMotifPaths);

        // second step: peak loci that have only one 1-d peak from the intersected 1-d
        //  peak track are identified (along with their corresponding unique 1-d peak)

        // anchors from given loop list
        Feature2DList features = Feature2DParser.loadFeatures(loopListPath, chromosomes, true, null, true);
        AnchorList anchors = AnchorList.extractAnchorsFromFeatures(features, false);
        //anchors.merge();
        //anchors.expandSmallAnchors(15000);

        // third step: the best motif match is identified by intersecting unique 1-d peaks
        // and the genome wide list of motifs. This gives a mapping of peak loci to unique motifs
        // (in the final loop list format, these motifs are outputted as 'u')

        AnchorList uniqueGlobalAnchors = AnchorTools.extractUniqueMotifs(globalAnchors, 5000);
        uniqueGlobalAnchors.intersectWith(proteinsForUniqueness, true);
        anchors.intersectWith(uniqueGlobalAnchors, true);
        anchors.updateOriginalMotifs(true);

        // 4th step - intersect all the 1d tracks for inferring motifs
        AnchorList proteinsForInference = getIntersectionOfBEDFiles(chromosomes, proteinsForInferredMotifPaths);

        // fifth step: the 1-d peak track from step 4 are intersected with the genome wide
        // motif list (best motif match) and split into a forward motif track and a reverse motif track.

        AnchorList bestGlobalAnchors = AnchorTools.extractBestMotifs(globalAnchors, 5000);
        bestGlobalAnchors.intersectWith(proteinsForInference, true);

        // sixth step: upstream peak loci that did not have a unique motif are intersected
        // with the forward motif track from step 5, and for each peak locus if the peak
        // locus has only one forward motif, that is an inferred mapping (these motifs
        // are outputted as 'i'). downstream peak loci that did not have a unique motif
        // are intersected with the reverse motif track from step 5, and for each peak
        // locus if the peak locus has only one reverse motif, that is an inferred mapping
        // (these motifs are outputted as 'i'). Peak loci that form loops in both directions are ignored.

        AnchorList remainingAnchors = AnchorList.extractAnchorsFromFeatures(features, true);
        remainingAnchors.intersectWith(bestGlobalAnchors, true);
        remainingAnchors.updateOriginalMotifs(false);

        // seventh step: the final output is the original loop list + information about the
        // motifs under each of the anchors (i.e. GEO format).

        features.exportFeatureList(outputPath, false);
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

    private AnchorList getIntersectionOfBEDFiles(List<Chromosome> chromosomes, List<String> bedFiles) {
        AnchorList proteins = AnchorParser.loadFromBEDFile(chromosomes, bedFiles.get(0));
        for (int i = 1; i < bedFiles.size(); i++) {
            AnchorList nextProteinList = AnchorParser.loadFromBEDFile(chromosomes, bedFiles.get(i));
            proteins.intersectWith(nextProteinList, false);
        }
        return proteins;
    }
}
