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
import juicebox.tools.utils.common.UNIXTools;
import juicebox.track.anchor.AnchorList;
import juicebox.track.anchor.AnchorParser;
import juicebox.track.anchor.AnchorTools;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.Feature2DParser;
import org.broad.igv.feature.Chromosome;

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

    private String outputDir;
    private String loopListPath;
    private String genomeID;
    private String[] proteinsForUniqueMotifPaths, proteinsForInferredMotifPaths;
    private String globalMotifListPath;

    public MotifFinder() {
        super("motifs <protein_files_for_unique_motifs> <protein_files_for_inferred_motifs> <genomeID> " +
                "[custom_global_motif_list] <looplist> <output directory>");
    }

    @Override
    public void readArguments(String[] args, CmdLineParser parser) {

        if (args.length != 6 && args.length != 7) {
            this.printUsage();
        }

        int i = 1;
        proteinsForUniqueMotifPaths = args[i++].split(",");
        proteinsForInferredMotifPaths = args[i++].split(",");
        genomeID = args[i++];
        if (args.length == 7) {
            globalMotifListPath = args[i++];
        }
        loopListPath = args[i++];
        outputDir = args[i++];
    }

    @Override
    public void run() {

        // create the output directory if it doesn't exist
        List<Chromosome> chromosomes = HiCFileTools.loadChromosomes(genomeID);


        // intersect all the 1d tracks for unique motifs
        AnchorList proteinsForUniqueness = AnchorParser.loadFromBEDFile(chromosomes, proteinsForUniqueMotifPaths[0]);
        for (int i = 1; i < proteinsForUniqueMotifPaths.length; i++) {
            AnchorList nextProteinList = AnchorParser.loadFromBEDFile(chromosomes, proteinsForUniqueMotifPaths[i]);
            proteinsForUniqueness.intersectWith(nextProteinList);
        }

        // intersect all the 1d tracks for inferring motifs
        AnchorList proteinsForInference = AnchorParser.loadFromBEDFile(chromosomes, proteinsForInferredMotifPaths[0]);
        for (int i = 1; i < proteinsForInferredMotifPaths.length; i++) {
            AnchorList nextProteinList = AnchorParser.loadFromBEDFile(chromosomes, proteinsForInferredMotifPaths[i]);
            proteinsForInference.intersectWith(nextProteinList);
        }

        // anchors from given loop list
        Feature2DList features = Feature2DParser.loadFeatures(loopListPath, chromosomes, true, null);
        AnchorList anchors = AnchorList.extractAnchorsFromFeatures(features);
        anchors.merge();
        anchors.expandSmallAnchors(15000);

        AnchorList globalAnchors;

        if (globalMotifListPath == null || globalMotifListPath.length() < 1) {
            globalAnchors = AnchorParser.loadGlobalMotifs(genomeID, chromosomes);
        } else {
            globalAnchors = AnchorParser.loadMotifs(globalMotifListPath, chromosomes, null);
        }

        AnchorList uniqueGlobalAnchors = AnchorTools.extractUniqueMotifs(globalAnchors, 5000);
        AnchorList bestGlobalAnchors = AnchorTools.extractBestMotifs(globalAnchors, 5000);


        String t="";
        UNIXTools.extractElement(t,2);


        UNIXTools.executeSimpleCommand("mkdir -p " + outputDir + "/CTCF/REN_fimo_out");
        UNIXTools.executeSimpleCommand("mkdir -p " + outputDir + "/CTCF/M1_fimo_out");
        UNIXTools.executeSimpleCommand("mkdir -p " + outputDir + "/peak_loci");

        /*

        redirectOutput(bedtoolsPath + " intersect -a " + ctcfPath + " -b " + outputDir+"/peak_loci.txt -c | awk '{if ($NF>0) {print $1 \"\\t\" $2 \"\\t\" $3 \"\\t\" NR}}' " ,
                outputDir+"/CTCF/CTCF_in_peak_loci.txt");
        redirectOutput("awk -v f=\"" + outputDir+"/CTCF/REN_fimo_out/motifs_best.txt\" 'BEGIN{while(getline<f>0){motif[$2 \" \" $3 \" \" $4]=$1}}{if (motif[$1 \" \" $2 \" \" $3]!=0) {print $0 \"\\t\" motif[$1 \" \" $2 \" \" $3]} else {print $0 \"\\t\" \"NA\"}}' " + outputDir+"/CTCF/CTCF_in_peak_loci.txt " ,
                outputDir+"/CTCF/tmp1");
        redirectOutput("awk -v f=\"" + outputDir+"/CTCF/M1_fimo_out/motifs_best.txt\" 'BEGIN{while(getline<f>0){motif[$2 \" \" $3 \" \" $4]=$1}}{if (motif[$1 \" \" $2 \" \" $3]!=0) {print $0 \"\\t\" motif[$1 \" \" $2 \" \" $3]} else {print $0 \"\\t\" \"NA\"}}' " + outputDir+"/CTCF/tmp1 " ,
                outputDir+"/CTCF/tmp2");
        executeCommand("rm " + outputDir+"/CTCF/tmp1");
        redirectOutput("awk -v f=\"" + outputDir + "/CTCF/REN_fimo_out/motifs_unique.txt\" 'BEGIN{while(getline<f>0){motif[$2 \" \" $3 \" \" $4]=$1}}{if (motif[$1 \" \" $2 \" \" $3]!=0) {print $0 \"\\t\" motif[$1 \" \" $2 \" \" $3]} else {print $0 \"\\t\" \"NA\"}}' " + outputDir + "/CTCF/tmp2 ",
                outputDir+"/CTCF/tmp3");
        executeCommand("rm " + outputDir+"/CTCF/tmp2");
        redirectOutput("awk -v f=\"" + outputDir + "/CTCF/M1_fimo_out/motifs_unique.txt\" 'BEGIN{while(getline<f>0){motif[$2 \" \" $3 \" \" $4]=$1}}{if (motif[$1 \" \" $2 \" \" $3]!=0) {print $0 \"\\t\" motif[$1 \" \" $2 \" \" $3]} else {print $0 \"\\t\" \"NA\"}}' " + outputDir + "/CTCF/tmp3 ",
                outputDir+"/CTCF/tmp4");
        executeCommand("rm " + outputDir+"/CTCF/tmp3");
        redirectOutput("awk -v f1=\"" + outputDir + "/CTCF/REN_fimo_out/motifs.txt\" -v f2=\"" + outputDir + "/CTCF/M1_fimo_out/motifs.txt\" 'BEGIN{while(getline<f1>0){motif1[$1]=$5 \"\\t\" $6 \"\\t\" $7 \"\\t\" $8 \"\\t\" $9 \"\\t\" $10 \"\\t\" $11} while(getline<f2>0){motif2[$1]=$5 \"\\t\" $6 \"\\t\" $7 \"\\t\" $8 \"\\t\" $9 \"\\t\" $10 \"\\t\" $11}}{line=$0; if ($5!=\"NA\") {line=line \"\\t\" motif1[$5]} else {line=line \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\"} if ($6!=\"NA\") {line = line \"\\t\" motif2[$6]} else {line=line \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\"} if ($7!=\"NA\") {line = line \"\\t\" motif1[$7]} else {line=line \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\"} if ($8!=\"NA\") {line = line \"\\t\" motif2[$8]} else {line=line \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\"} print line}' " + outputDir + "/CTCF/tmp4",
                outputDir + "/CTCF/CTCF_in_peak_loci.txt");
        executeCommand("rm " + outputDir+"/CTCF/tmp4");
        redirectOutput(bedtoolsPath + " intersect -a " + ctcfPath + " -b " + outputDir + "/peak_loci.txt -c | awk '{if ($NF==0) {print $1 \"\\t\" $2 \"\\t\" $3 \"\\t\" NR}}' ",
                outputDir + "/CTCF/CTCF_not_in_peak_loci.txt");
        redirectOutput("awk -v f=\"" + outputDir+"/CTCF/REN_fimo_out/motifs_best.txt\" 'BEGIN{while(getline<f>0){motif[$2 \" \" $3 \" \" $4]=$1}}{if (motif[$1 \" \" $2 \" \" $3]!=0) {print $0 \"\\t\" motif[$1 \" \" $2 \" \" $3]} else {print $0 \"\\t\" \"NA\"}}' " + outputDir+"/CTCF/CTCF_not_in_peak_loci.txt " ,
                outputDir+"/CTCF/tmp1");
        redirectOutput("awk -v f=\"" + outputDir+"/CTCF/M1_fimo_out/motifs_best.txt\" 'BEGIN{while(getline<f>0){motif[$2 \" \" $3 \" \" $4]=$1}}{if (motif[$1 \" \" $2 \" \" $3]!=0) {print $0 \"\\t\" motif[$1 \" \" $2 \" \" $3]} else {print $0 \"\\t\" \"NA\"}}' " + outputDir+"/CTCF/tmp1 " ,
                outputDir+"/CTCF/tmp2");
        executeCommand("rm " + outputDir+"/CTCF/tmp1");
        redirectOutput("awk -v f=\"" + outputDir + "/CTCF/REN_fimo_out/motifs_unique.txt\" 'BEGIN{while(getline<f>0){motif[$2 \" \" $3 \" \" $4]=$1}}{if (motif[$1 \" \" $2 \" \" $3]!=0) {print $0 \"\\t\" motif[$1 \" \" $2 \" \" $3]} else {print $0 \"\\t\" \"NA\"}}' " + outputDir + "/CTCF/tmp2 ",
                outputDir+"/CTCF/tmp3");
        executeCommand("rm " + outputDir+"/CTCF/tmp2");
        redirectOutput("awk -v f=\"" + outputDir + "/CTCF/M1_fimo_out/motifs_unique.txt\" 'BEGIN{while(getline<f>0){motif[$2 \" \" $3 \" \" $4]=$1}}{if (motif[$1 \" \" $2 \" \" $3]!=0) {print $0 \"\\t\" motif[$1 \" \" $2 \" \" $3]} else {print $0 \"\\t\" \"NA\"}}' " + outputDir + "/CTCF/tmp3 ",
                outputDir+"/CTCF/tmp4");
        executeCommand("rm " + outputDir+"/CTCF/tmp3");
        redirectOutput("awk -v f1=\"" + outputDir+"/CTCF/REN_fimo_out/motifs.txt\" -v f2=\"" + outputDir+"/CTCF/M1_fimo_out/motifs.txt\" 'BEGIN{while(getline<f1>0){motif1[$1]=$5 \"\\t\" $6 \"\\t\" $7 \"\\t\" $8 \"\\t\" $9 \"\\t\" $10 \"\\t\" $11} while(getline<f2>0){motif2[$1]=$5 \"\\t\" $6 \"\\t\" $7 \"\\t\" $8 \"\\t\" $9 \"\\t\" $10 \"\\t\" $11}}{line=$0; if ($5!=\"NA\") {line=line \"\\t\" motif1[$5]} else {line=line \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\"} if ($6!=\"NA\") {line = line \"\\t\" motif2[$6]} else {line=line \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\"} if ($7!=\"NA\") {line = line \"\\t\" motif1[$7]} else {line=line \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\"} if ($8!=\"NA\") {line = line \"\\t\" motif2[$8]} else {line=line \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\"} print line}' " + outputDir+"/CTCF/tmp4" ,
                outputDir+"/CTCF/CTCF_not_in_peak_loci.txt");
        executeCommand("rm " + outputDir+"/CTCF/tmp4");
        redirectOutput(bedtoolsPath + " intersect -a " + outputDir + "/peak_loci.txt -b " + outputDir + "/CTCF/CTCF_in_peak_loci.txt -c | awk '{if ($NF>0) {print $1 \"\\t\" $2 \"\\t\" $3}}' ",
                outputDir + "/peak_loci/peak_loci_CTCF.txt");
        redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/peak_loci.txt -b " + outputDir+"/CTCF/CTCF_in_peak_loci.txt -c | awk '{if ($NF==1) {print $1 \"\\t\" $2 \"\\t\" $3}}' " ,
                outputDir+"/peak_loci/peak_loci_1CTCFvCTCF.txt");
        redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/CTCF/CTCF_in_peak_loci.txt -b " + outputDir+"/peak_loci/peak_loci_1CTCFvCTCF.txt -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev " ,
                outputDir+"/CTCF/unique_CTCFvCTCF_in_peak_loci.txt");

        if (ctcfCollapsedPath.length() > 0) {
            executeCommand("mkdir -p " + outputDir+"/CTCF_collapsed");
            redirectOutput(bedtoolsPath + " intersect -a " + outputDir + "/CTCF/CTCF_in_peak_loci.txt -b " + ctcfCollapsedPath + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev ",
                    outputDir + "/CTCF_collapsed/cCTCF_in_peak_loci.txt");
            redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/CTCF/CTCF_not_in_peak_loci.txt -b " + ctcfCollapsedPath + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev " ,
                    outputDir+"/CTCF_collapsed/cCTCF_not_in_peak_loci.txt");
            redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/CTCF/unique_CTCFvCTCF_in_peak_loci.txt -b " + ctcfCollapsedPath + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev " ,
                    outputDir+"/CTCF_collapsed/unique_cCTCFvCTCF_in_peak_loci.txt");
            redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/peak_loci.txt -b " + outputDir+"/CTCF_collapsed/cCTCF_in_peak_loci.txt -c | awk '{if ($NF>0) {print $1 \"\\t\" $2 \"\\t\" $3}}' " ,
                    outputDir+"/peak_loci/peak_loci_cCTCF.txt");
            redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/peak_loci.txt -b " + outputDir+"/CTCF_collapsed/cCTCF_in_peak_loci.txt -c | awk '{if ($NF==1) {print $1 \"\\t\" $2 \"\\t\" $3}}' " ,
                    outputDir+"/peak_loci/peak_loci_1cCTCFvcCTCF.txt");
            redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/CTCF_collapsed/cCTCF_in_peak_loci.txt -b " + outputDir+"/peak_loci/peak_loci_1cCTCFvcCTCF.txt -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev " ,
                    outputDir+"/CTCF_collapsed/unique_cCTCFvcCTCF_in_peak_loci.txt");
        }

        if (rad21Path.length() > 0) {
            redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/CTCF/CTCF_in_peak_loci.txt -b " + rad21Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev " ,
                    outputDir+"/CTCF/CTCF_RAD21_in_peak_loci.txt");
            redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/CTCF/CTCF_not_in_peak_loci.txt -b " + rad21Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev " ,
                    outputDir+"/CTCF/CTCF_RAD21_not_in_peak_loci.txt");
            redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/CTCF/unique_CTCFvCTCF_in_peak_loci.txt -b " + rad21Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev " ,
                    outputDir+"/CTCF/unique_CTCFvCTCF_RAD21_in_peak_loci.txt");
            redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/peak_loci.txt -b " + outputDir+"/CTCF/CTCF_RAD21_in_peak_loci.txt -c | awk '{if ($NF>0) {print $1 \"\\t\" $2 \"\\t\" $3}}' " ,
                    outputDir+"/peak_loci/peak_loci_CTCF_RAD21.txt");
            if (ctcfCollapsedPath.length() > 0) {
                redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/CTCF_collapsed/cCTCF_in_peak_loci.txt -b " + rad21Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev " ,
                        outputDir+"/CTCF_collapsed/cCTCF_RAD21_in_peak_loci.txt");
                redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/CTCF_collapsed/cCTCF_not_in_peak_loci.txt -b " + rad21Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev " ,
                        outputDir+"/CTCF_collapsed/cCTCF_RAD21_not_in_peak_loci.txt");
                redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/CTCF_collapsed/unique_cCTCFvCTCF_in_peak_loci.txt -b " + rad21Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev " ,
                        outputDir+"/CTCF_collapsed/unique_cCTCFvCTCF_RAD21_in_peak_loci.txt");
                redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/CTCF_collapsed/unique_cCTCFvcCTCF_in_peak_loci.txt -b " + rad21Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev " ,
                        outputDir+"/CTCF_collapsed/unique_cCTCFvcCTCF_RAD21_in_peak_loci.txt");
                redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/peak_loci.txt -b " + outputDir+"/CTCF_collapsed/cCTCF_RAD21_in_peak_loci.txt -c | awk '{if ($NF>0) {print $1 \"\\t\" $2 \"\\t\" $3}}' " ,
                        outputDir+"/peak_loci/peak_loci_cCTCF_RAD21.txt");
            }
        }

        if (smc3Path.length() > 0) {
            redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/CTCF/CTCF_in_peak_loci.txt -b " + smc3Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev " ,
                    outputDir+"/CTCF/CTCF_SMC3_in_peak_loci.txt");
            redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/CTCF/CTCF_not_in_peak_loci.txt -b " + smc3Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev " ,
                    outputDir+"/CTCF/CTCF_SMC3_not_in_peak_loci.txt");
            redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/CTCF/unique_CTCFvCTCF_in_peak_loci.txt -b " + smc3Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev " ,
                    outputDir+"/CTCF/unique_CTCFvCTCF_SMC3_in_peak_loci.txt");
            redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/peak_loci.txt -b " + outputDir+"/CTCF/CTCF_SMC3_in_peak_loci.txt -c | awk '{if ($NF>0) {print $1 \"\\t\" $2 \"\\t\" $3}}' " ,
                    outputDir+"/peak_loci/peak_loci_CTCF_SMC3.txt");
            if (ctcfCollapsedPath.length() > 0) {
                redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/CTCF_collapsed/cCTCF_in_peak_loci.txt -b " + smc3Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev " ,
                        outputDir+"/CTCF_collapsed/cCTCF_SMC3_in_peak_loci.txt");
                redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/CTCF_collapsed/cCTCF_not_in_peak_loci.txt -b " + smc3Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev " ,
                        outputDir+"/CTCF_collapsed/cCTCF_SMC3_not_in_peak_loci.txt");
                redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/CTCF_collapsed/unique_cCTCFvCTCF_in_peak_loci.txt -b " + smc3Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev " ,
                        outputDir+"/CTCF_collapsed/unique_cCTCFvCTCF_SMC3_in_peak_loci.txt");
                redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/CTCF_collapsed/unique_cCTCFvcCTCF_in_peak_loci.txt -b " + smc3Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev " ,
                        outputDir+"/CTCF_collapsed/unique_cCTCFvcCTCF_SMC3_in_peak_loci.txt");
                redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/peak_loci.txt -b " + outputDir+"/CTCF_collapsed/cCTCF_SMC3_in_peak_loci.txt -c | awk '{if ($NF>0) {print $1 \"\\t\" $2 \"\\t\" $3}}' " ,
                        outputDir+"/peak_loci/peak_loci_cCTCF_SMC3.txt");
            }
            if (rad21Path.length() > 0) {
                redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/CTCF/CTCF_RAD21_in_peak_loci.txt -b " + smc3Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev " ,
                        outputDir+"/CTCF/CTCF_RAD21_SMC3_in_peak_loci.txt");
                redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/CTCF/CTCF_RAD21_not_in_peak_loci.txt -b " + smc3Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev " ,
                        outputDir+"/CTCF/CTCF_RAD21_SMC3_not_in_peak_loci.txt");
                redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/CTCF/unique_CTCFvCTCF_RAD21_in_peak_loci.txt -b " + smc3Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev " ,
                        outputDir+"/CTCF/unique_CTCFvCTCF_RAD21_SMC3_in_peak_loci.txt");
                redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/peak_loci.txt -b " + outputDir+"/CTCF/CTCF_RAD21_SMC3_in_peak_loci.txt -c | awk '{if ($NF>0) {print $1 \"\\t\" $2 \"\\t\" $3}}' " ,
                        outputDir+"/peak_loci/peak_loci_CTCF_RAD21_SMC3.txt");
                if (ctcfCollapsedPath.length() > 0) {
                    redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/CTCF_collapsed/cCTCF_RAD21_in_peak_loci.txt -b " + smc3Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev " ,
                            outputDir+"/CTCF_collapsed/cCTCF_RAD21_SMC3_in_peak_loci.txt");
                    redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/CTCF_collapsed/cCTCF_RAD21_not_in_peak_loci.txt -b " + smc3Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev " ,
                            outputDir+"/CTCF_collapsed/cCTCF_RAD21_SMC3_not_in_peak_loci.txt");
                    redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/CTCF_collapsed/unique_cCTCFvCTCF_RAD21_in_peak_loci.txt -b " + smc3Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev " ,
                            outputDir+"/CTCF_collapsed/unique_cCTCFvCTCF_RAD21_SMC3_in_peak_loci.txt");
                    redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/CTCF_collapsed/unique_cCTCFvcCTCF_RAD21_in_peak_loci.txt -b " + smc3Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev " ,
                            outputDir+"/CTCF_collapsed/unique_cCTCFvcCTCF_RAD21_SMC3_in_peak_loci.txt");
                    redirectOutput(bedtoolsPath + " intersect -a " + outputDir+"/peak_loci.txt -b " + outputDir+"/CTCF_collapsed/cCTCF_RAD21_SMC3_in_peak_loci.txt -c | awk '{if ($NF>0) {print $1 \"\\t\" $2 \"\\t\" $3}}' " ,
                            outputDir+"/peak_loci/peak_loci_cCTCF_RAD21_SMC3.txt");
                }
            }
        }
        */


    }
}
