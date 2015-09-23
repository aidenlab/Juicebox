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
import juicebox.tools.clt.CommandLineParserForJuicer;
import juicebox.tools.clt.JuicerCLT;

/**
 * Created by muhammadsaadshamim on 9/4/15.
 */
public class MotifFinder extends JuicerCLT {

    private boolean noFilesSpecified = true;

    private String ctcfPath = "";
    private String ctcfCollapsedPath = "";
    private String rad21Path = "";
    private String smc3Path = "";
    private String genomeID;
    private String outputDir;
    private String loopListPath;

    public MotifFinder() {
        super("motifs [-a CTCF_collapsed_input_file] [-r RAD21_input_file] [-s SMC3_input_file]" +
                " <CTCF_input_file> <genomeID> <looplist> <output directory>");
    }


    @Override
    public void readArguments(String[] args, CmdLineParser parser) {

        ctcfPath = args[1];
        genomeID = args[2];
        loopListPath = args[3];
        outputDir = args[4];

        CommandLineParserForJuicer juicerParser = (CommandLineParserForJuicer) parser;
        String ctcfCollapsed = juicerParser.getCTCFCollapsedOption();
        String rad21 = juicerParser.getRAD21Option();
        String smc3 = juicerParser.getSMC3Option();

        if (ctcfCollapsed != null) {
            noFilesSpecified = false;
            ctcfCollapsedPath = ctcfCollapsed;
        }
        if (rad21 != null) {
            noFilesSpecified = false;
            rad21Path = rad21;
        }
        if (smc3 != null) {
            noFilesSpecified = false;
            smc3Path = smc3;
        }

        if (noFilesSpecified) {
            printUsage();
        }
    }


    @Override
    public void run() {

        executeCommand("awk -v g=" + genomeID + " '{if ((g==\"mm9\"||g==\"galGal4\")&&NR>1) {print $1 \"\\t\" $2 \"\\t\" $3; print $4 \"\\t\" $5 \"\\t\" $6} else if (g==\"hg19\"&&NR>1) {print \"chr\"$1 \"\\t\" $2 \"\\t\" $3; print \"chr\"$4 \"\\t\" $5 \"\\t\" $6}}' \"" + loopListPath + "\" | sort -k1,1 -k2n,2 | uniq > \"" + outputDir + "\"\"/temp_anchors_origsize.txt\"");
        executeCommand("bedtools merge -i \"" + outputDir + "\"\"/temp_anchors_origsize.txt\" | awk '{if ($3-$2<15000){d=15000-($3-$2); print $1 \"\\t\" $2-int(d/2) \"\\t\" $3+int(d/2)} else {print $0}}' > \"" + outputDir + "\"\"/peak_loci.txt\"");
        executeCommand("rm \"" + outputDir + "\"\"/temp_anchors_origsize.txt\"");
        executeCommand("awk -v f=" + outputDir + "\"/peak_loci.txt\" 'BEGIN{while(getline<f>0) {locus[$1 \" \" $2 \" \" $3]++;}}{end1=0;end2=0; if (NR>1) {for (i in locus) {split(i,a,\" \"); if (a[1]==\"chr\"$1&&$2>=a[2]&&$3<=a[3]) {end1=i} if (a[1]==\"chr\"$4&&$5>=a[2]&&$6<=a[3]) {end2=i}}} if (NR>1) {print end1 \"\\t\" end2;}}' \"" + loopListPath + "\" > \"" + outputDir + "\"\"/combined_peakloci_looplist.txt\"");

        executeCommand("mkdir -p \"" + outputDir + "\"\"/CTCF\"");
        executeCommand("mkdir -p \"" + outputDir + "\"\"/peak_loci\"");

        if (genomeID.equals("hg19")) {
            executeCommand("awk '{print $0 \"\\t\" NR}' " + ctcfPath + " > \"" + outputDir + "\"\"/tmp1\"");
            // TODO remove hardcoded files (and find them)
            executeCommand("/broad/aidenlab/suhas/scripts/twoBitToFa /broad/aidenlab/suhas/scripts/hg19.2bit " + outputDir + "\"/CTCF/CTCF.fa\" -noMask -bed=" + outputDir + "\"/tmp1\" -bedPos");
            executeCommand("rm " + outputDir + "\"/tmp1\"");
            // TODO mapping of .mat files?
            executeCommand("fimo --thresh .001 -max-stored-scores 500000 --o " + outputDir + "\"/CTCF/REN_fimo_out\" CTCF_FIMO.mat " + outputDir + "\"/CTCF/CTCF.fa\"");
            executeCommand("fimo --thresh .001 -max-stored-scores 500000 --o " + outputDir + "\"/CTCF/M1_fimo_out\" M1_FIMO_flipped.mat " + outputDir + "\"/CTCF/CTCF.fa\"");
        }

        executeCommand("awk '{if (NR>1) {split($2,a,\":\");split(a[2],b,\"-\"); print NR-1 \"\\t\" a[1] \"\\t\" b[1] \"\\t\" b[2] \"\\t\" b[1]+$3 \"\\t\" b[1]+$4 \"\\t\" $5 \"\\t\" $6 \"\\t\" $7 \"\\t\" $8 \"\\t\" $9}}' \"" + outputDir + "\"\"/CTCF/REN_fimo_out/fimo.txt\" > \"" + outputDir + "\"\"/CTCF/REN_fimo_out/motifs.txt\"");
        executeCommand("awk '{if (counter[$2 \" \" $3 \" \" $4]==0) {counter[$2 \" \" $3 \" \" $4]++;motif[$2 \" \" $3 \" \" $4]=$0;score[$2 \" \" $3 \" \" $4]=$8} else if (score[$2 \" \" $3 \" \" $4]<$8) {motif[$2 \" \" $3 \" \" $4]=$0;score[$2 \" \" $3 \" \" $4]=$8}}END{for (i in motif) {print motif[i]}}' \"" + outputDir + "\"\"/CTCF/REN_fimo_out/motifs.txt\" > \"" + outputDir + "\"\"/CTCF/REN_fimo_out/motifs_best.txt\"");
        executeCommand("awk '{counter[$2 \" \" $3 \" \" $4]++;motif[$2 \" \" $3 \" \" $4]=$0}END{for (i in motif) {if (counter[i]==1) {print motif[i]}}}' \"" + outputDir + "\"\"/CTCF/REN_fimo_out/motifs.txt\" > \"" + outputDir + "\"\"/CTCF/REN_fimo_out/motifs_unique.txt\"");
        executeCommand("awk '{if (NR>1) {split($2,a,\":\");split(a[2],b,\"-\"); print NR-1 \"\\t\" a[1] \"\\t\" b[1] \"\\t\" b[2] \"\\t\" b[1]+$3 \"\\t\" b[1]+$4 \"\\t\" $5 \"\\t\" $6 \"\\t\" $7 \"\\t\" $8 \"\\t\" $9}}' \"" + outputDir + "\"\"/CTCF/M1_fimo_out/fimo.txt\" > \"" + outputDir + "\"\"/CTCF/M1_fimo_out/motifs.txt\"");
        executeCommand("awk '{if (counter[$2 \" \" $3 \" \" $4]==0) {counter[$2 \" \" $3 \" \" $4]++;motif[$2 \" \" $3 \" \" $4]=$0;score[$2 \" \" $3 \" \" $4]=$8} else if (score[$2 \" \" $3 \" \" $4]<$8) {motif[$2 \" \" $3 \" \" $4]=$0;score[$2 \" \" $3 \" \" $4]=$8}}END{for (i in motif) {print motif[i]}}' \"" + outputDir + "\"\"/CTCF/M1_fimo_out/motifs.txt\" > \"" + outputDir + "\"\"/CTCF/M1_fimo_out/motifs_best.txt\"");
        executeCommand("awk '{counter[$2 \" \" $3 \" \" $4]++;motif[$2 \" \" $3 \" \" $4]=$0}END{for (i in motif) {if (counter[i]==1) {print motif[i]}}}' \"" + outputDir + "\"\"/CTCF/M1_fimo_out/motifs.txt\" > \"" + outputDir + "\"\"/CTCF/M1_fimo_out/motifs_unique.txt\"");

        executeCommand("bedtools intersect -a " + ctcfPath + " -b " + outputDir + "\"/peak_loci.txt\" -c | awk '{if ($NF>0) {print $1 \"\\t\" $2 \"\\t\" $3 \"\\t\" NR}}' > " + outputDir + "\"/CTCF/CTCF_in_peak_loci.txt\"");
        executeCommand("awk -v f=\"" + outputDir + "\"\"/CTCF/REN_fimo_out/motifs_best.txt\" 'BEGIN{while(getline<f>0){motif[$2 \" \" $3 \" \" $4]=$1}}{if (motif[$1 \" \" $2 \" \" $3]!=0) {print $0 \"\\t\" motif[$1 \" \" $2 \" \" $3]} else {print $0 \"\\t\" \"NA\"}}' " + outputDir + "\"/CTCF/CTCF_in_peak_loci.txt\" > " + outputDir + "\"/CTCF/tmp1\"");
        executeCommand("awk -v f=\"" + outputDir + "\"\"/CTCF/M1_fimo_out/motifs_best.txt\" 'BEGIN{while(getline<f>0){motif[$2 \" \" $3 \" \" $4]=$1}}{if (motif[$1 \" \" $2 \" \" $3]!=0) {print $0 \"\\t\" motif[$1 \" \" $2 \" \" $3]} else {print $0 \"\\t\" \"NA\"}}' " + outputDir + "\"/CTCF/tmp1\" > " + outputDir + "\"/CTCF/tmp2\"");
        executeCommand("rm " + outputDir + "\"/CTCF/tmp1\"");
        executeCommand("awk -v f=\"" + outputDir + "\"\"/CTCF/REN_fimo_out/motifs_unique.txt\" 'BEGIN{while(getline<f>0){motif[$2 \" \" $3 \" \" $4]=$1}}{if (motif[$1 \" \" $2 \" \" $3]!=0) {print $0 \"\\t\" motif[$1 \" \" $2 \" \" $3]} else {print $0 \"\\t\" \"NA\"}}' " + outputDir + "\"/CTCF/tmp2\" > " + outputDir + "\"/CTCF/tmp3\"");
        executeCommand("rm " + outputDir + "\"/CTCF/tmp2\"");
        executeCommand("awk -v f=\"" + outputDir + "\"\"/CTCF/M1_fimo_out/motifs_unique.txt\" 'BEGIN{while(getline<f>0){motif[$2 \" \" $3 \" \" $4]=$1}}{if (motif[$1 \" \" $2 \" \" $3]!=0) {print $0 \"\\t\" motif[$1 \" \" $2 \" \" $3]} else {print $0 \"\\t\" \"NA\"}}' " + outputDir + "\"/CTCF/tmp3\" > " + outputDir + "\"/CTCF/tmp4\"");
        executeCommand("rm " + outputDir + "\"/CTCF/tmp3\"");
        executeCommand("awk -v f1=\"" + outputDir + "\"\"/CTCF/REN_fimo_out/motifs.txt\" -v f2=\"" + outputDir + "\"\"/CTCF/M1_fimo_out/motifs.txt\" 'BEGIN{while(getline<f1>0){motif1[$1]=$5 \"\\t\" $6 \"\\t\" $7 \"\\t\" $8 \"\\t\" $9 \"\\t\" $10 \"\\t\" $11} while(getline<f2>0){motif2[$1]=$5 \"\\t\" $6 \"\\t\" $7 \"\\t\" $8 \"\\t\" $9 \"\\t\" $10 \"\\t\" $11}}{line=$0; if ($5!=\"NA\") {line=line \"\\t\" motif1[$5]} else {line=line \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\"} if ($6!=\"NA\") {line = line \"\\t\" motif2[$6]} else {line=line \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\"} if ($7!=\"NA\") {line = line \"\\t\" motif1[$7]} else {line=line \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\"} if ($8!=\"NA\") {line = line \"\\t\" motif2[$8]} else {line=line \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\"} print line}' " + outputDir + "\"/CTCF/tmp4\" > " + outputDir + "\"/CTCF/CTCF_in_peak_loci.txt\"");
        executeCommand("rm " + outputDir + "\"/CTCF/tmp4\"");
        executeCommand("bedtools intersect -a " + ctcfPath + " -b " + outputDir + "\"/peak_loci.txt\" -c | awk '{if ($NF==0) {print $1 \"\\t\" $2 \"\\t\" $3 \"\\t\" NR}}' > " + outputDir + "\"/CTCF/CTCF_not_in_peak_loci.txt\"");
        executeCommand("awk -v f=\"" + outputDir + "\"\"/CTCF/REN_fimo_out/motifs_best.txt\" 'BEGIN{while(getline<f>0){motif[$2 \" \" $3 \" \" $4]=$1}}{if (motif[$1 \" \" $2 \" \" $3]!=0) {print $0 \"\\t\" motif[$1 \" \" $2 \" \" $3]} else {print $0 \"\\t\" \"NA\"}}' " + outputDir + "\"/CTCF/CTCF_not_in_peak_loci.txt\" > " + outputDir + "\"/CTCF/tmp1\"");
        executeCommand("awk -v f=\"" + outputDir + "\"\"/CTCF/M1_fimo_out/motifs_best.txt\" 'BEGIN{while(getline<f>0){motif[$2 \" \" $3 \" \" $4]=$1}}{if (motif[$1 \" \" $2 \" \" $3]!=0) {print $0 \"\\t\" motif[$1 \" \" $2 \" \" $3]} else {print $0 \"\\t\" \"NA\"}}' " + outputDir + "\"/CTCF/tmp1\" > " + outputDir + "\"/CTCF/tmp2\"");
        executeCommand("rm " + outputDir + "\"/CTCF/tmp1\"");
        executeCommand("awk -v f=\"" + outputDir + "\"\"/CTCF/REN_fimo_out/motifs_unique.txt\" 'BEGIN{while(getline<f>0){motif[$2 \" \" $3 \" \" $4]=$1}}{if (motif[$1 \" \" $2 \" \" $3]!=0) {print $0 \"\\t\" motif[$1 \" \" $2 \" \" $3]} else {print $0 \"\\t\" \"NA\"}}' " + outputDir + "\"/CTCF/tmp2\" > " + outputDir + "\"/CTCF/tmp3\"");
        executeCommand("rm " + outputDir + "\"/CTCF/tmp2\"");
        executeCommand("awk -v f=\"" + outputDir + "\"\"/CTCF/M1_fimo_out/motifs_unique.txt\" 'BEGIN{while(getline<f>0){motif[$2 \" \" $3 \" \" $4]=$1}}{if (motif[$1 \" \" $2 \" \" $3]!=0) {print $0 \"\\t\" motif[$1 \" \" $2 \" \" $3]} else {print $0 \"\\t\" \"NA\"}}' " + outputDir + "\"/CTCF/tmp3\" > " + outputDir + "\"/CTCF/tmp4\"");
        executeCommand("rm " + outputDir + "\"/CTCF/tmp3\"");
        executeCommand("awk -v f1=\"" + outputDir + "\"\"/CTCF/REN_fimo_out/motifs.txt\" -v f2=\"" + outputDir + "\"\"/CTCF/M1_fimo_out/motifs.txt\" 'BEGIN{while(getline<f1>0){motif1[$1]=$5 \"\\t\" $6 \"\\t\" $7 \"\\t\" $8 \"\\t\" $9 \"\\t\" $10 \"\\t\" $11} while(getline<f2>0){motif2[$1]=$5 \"\\t\" $6 \"\\t\" $7 \"\\t\" $8 \"\\t\" $9 \"\\t\" $10 \"\\t\" $11}}{line=$0; if ($5!=\"NA\") {line=line \"\\t\" motif1[$5]} else {line=line \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\"} if ($6!=\"NA\") {line = line \"\\t\" motif2[$6]} else {line=line \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\"} if ($7!=\"NA\") {line = line \"\\t\" motif1[$7]} else {line=line \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\"} if ($8!=\"NA\") {line = line \"\\t\" motif2[$8]} else {line=line \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\" \"\\t\" \"NA\"} print line}' " + outputDir + "\"/CTCF/tmp4\" > " + outputDir + "\"/CTCF/CTCF_not_in_peak_loci.txt\"");
        executeCommand("rm " + outputDir + "\"/CTCF/tmp4\"");
        executeCommand("bedtools intersect -a " + outputDir + "\"/peak_loci.txt\" -b " + outputDir + "\"/CTCF/CTCF_in_peak_loci.txt\" -c | awk '{if ($NF>0) {print $1 \"\\t\" $2 \"\\t\" $3}}' > " + outputDir + "\"/peak_loci/peak_loci_CTCF.txt\"");
        executeCommand("bedtools intersect -a " + outputDir + "\"/peak_loci.txt\" -b " + outputDir + "\"/CTCF/CTCF_in_peak_loci.txt\" -c | awk '{if ($NF==1) {print $1 \"\\t\" $2 \"\\t\" $3}}' > " + outputDir + "\"/peak_loci/peak_loci_1CTCFvCTCF.txt\"");
        executeCommand("bedtools intersect -a " + outputDir + "\"/CTCF/CTCF_in_peak_loci.txt\" -b " + outputDir + "\"/peak_loci/peak_loci_1CTCFvCTCF.txt\" -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev > " + outputDir + "\"/CTCF/unique_CTCFvCTCF_in_peak_loci.txt\"");

        if (ctcfCollapsedPath.length() > 0) {
            executeCommand("mkdir -p \"" + outputDir + "\"\"/CTCF_collapsed\"");
            executeCommand("bedtools intersect -a " + outputDir + "\"/CTCF/CTCF_in_peak_loci.txt\" -b " + ctcfCollapsedPath + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev > " + outputDir + "\"/CTCF_collapsed/cCTCF_in_peak_loci.txt\"");
            executeCommand("bedtools intersect -a " + outputDir + "\"/CTCF/CTCF_not_in_peak_loci.txt\" -b " + ctcfCollapsedPath + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev > " + outputDir + "\"/CTCF_collapsed/cCTCF_not_in_peak_loci.txt\"");
            executeCommand("bedtools intersect -a " + outputDir + "\"/CTCF/unique_CTCFvCTCF_in_peak_loci.txt\" -b " + ctcfCollapsedPath + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev > " + outputDir + "\"/CTCF_collapsed/unique_cCTCFvCTCF_in_peak_loci.txt\"");
            executeCommand("bedtools intersect -a " + outputDir + "\"/peak_loci.txt\" -b " + outputDir + "\"/CTCF_collapsed/cCTCF_in_peak_loci.txt\" -c | awk '{if ($NF>0) {print $1 \"\\t\" $2 \"\\t\" $3}}' > " + outputDir + "\"/peak_loci/peak_loci_cCTCF.txt\"");
            executeCommand("bedtools intersect -a " + outputDir + "\"/peak_loci.txt\" -b " + outputDir + "\"/CTCF_collapsed/cCTCF_in_peak_loci.txt\" -c | awk '{if ($NF==1) {print $1 \"\\t\" $2 \"\\t\" $3}}' > " + outputDir + "\"/peak_loci/peak_loci_1cCTCFvcCTCF.txt\"");
            executeCommand("bedtools intersect -a " + outputDir + "\"/CTCF_collapsed/cCTCF_in_peak_loci.txt\" -b " + outputDir + "\"/peak_loci/peak_loci_1cCTCFvcCTCF.txt\" -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev > " + outputDir + "\"/CTCF_collapsed/unique_cCTCFvcCTCF_in_peak_loci.txt\"");
        }

        if (rad21Path.length() > 0) {
            executeCommand("bedtools intersect -a " + outputDir + "\"/CTCF/CTCF_in_peak_loci.txt\" -b " + rad21Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev > " + outputDir + "\"/CTCF/CTCF_RAD21_in_peak_loci.txt\"");
            executeCommand("bedtools intersect -a " + outputDir + "\"/CTCF/CTCF_not_in_peak_loci.txt\" -b " + rad21Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev > " + outputDir + "\"/CTCF/CTCF_RAD21_not_in_peak_loci.txt\"");
            executeCommand("bedtools intersect -a " + outputDir + "\"/CTCF/unique_CTCFvCTCF_in_peak_loci.txt\" -b " + rad21Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev > " + outputDir + "\"/CTCF/unique_CTCFvCTCF_RAD21_in_peak_loci.txt\"");
            executeCommand("bedtools intersect -a " + outputDir + "\"/peak_loci.txt\" -b " + outputDir + "\"/CTCF/CTCF_RAD21_in_peak_loci.txt\" -c | awk '{if ($NF>0) {print $1 \"\\t\" $2 \"\\t\" $3}}' > " + outputDir + "\"/peak_loci/peak_loci_CTCF_RAD21.txt\"");
            if (ctcfCollapsedPath.length() > 0) {
                executeCommand("bedtools intersect -a " + outputDir + "\"/CTCF_collapsed/cCTCF_in_peak_loci.txt\" -b " + rad21Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev > " + outputDir + "\"/CTCF_collapsed/cCTCF_RAD21_in_peak_loci.txt\"");
                executeCommand("bedtools intersect -a " + outputDir + "\"/CTCF_collapsed/cCTCF_not_in_peak_loci.txt\" -b " + rad21Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev > " + outputDir + "\"/CTCF_collapsed/cCTCF_RAD21_not_in_peak_loci.txt\"");
                executeCommand("bedtools intersect -a " + outputDir + "\"/CTCF_collapsed/unique_cCTCFvCTCF_in_peak_loci.txt\" -b " + rad21Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev > " + outputDir + "\"/CTCF_collapsed/unique_cCTCFvCTCF_RAD21_in_peak_loci.txt\"");
                executeCommand("bedtools intersect -a " + outputDir + "\"/CTCF_collapsed/unique_cCTCFvcCTCF_in_peak_loci.txt\" -b " + rad21Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev > " + outputDir + "\"/CTCF_collapsed/unique_cCTCFvcCTCF_RAD21_in_peak_loci.txt\"");
                executeCommand("bedtools intersect -a " + outputDir + "\"/peak_loci.txt\" -b " + outputDir + "\"/CTCF_collapsed/cCTCF_RAD21_in_peak_loci.txt\" -c | awk '{if ($NF>0) {print $1 \"\\t\" $2 \"\\t\" $3}}' > " + outputDir + "\"/peak_loci/peak_loci_cCTCF_RAD21.txt\"");
            }
        }

        if (smc3Path.length() > 0) {
            executeCommand("bedtools intersect -a " + outputDir + "\"/CTCF/CTCF_in_peak_loci.txt\" -b " + smc3Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev > " + outputDir + "\"/CTCF/CTCF_SMC3_in_peak_loci.txt\"");
            executeCommand("bedtools intersect -a " + outputDir + "\"/CTCF/CTCF_not_in_peak_loci.txt\" -b " + smc3Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev > " + outputDir + "\"/CTCF/CTCF_SMC3_not_in_peak_loci.txt\"");
            executeCommand("bedtools intersect -a " + outputDir + "\"/CTCF/unique_CTCFvCTCF_in_peak_loci.txt\" -b " + smc3Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev > " + outputDir + "\"/CTCF/unique_CTCFvCTCF_SMC3_in_peak_loci.txt\"");
            executeCommand("bedtools intersect -a " + outputDir + "\"/peak_loci.txt\" -b " + outputDir + "\"/CTCF/CTCF_SMC3_in_peak_loci.txt\" -c | awk '{if ($NF>0) {print $1 \"\\t\" $2 \"\\t\" $3}}' > " + outputDir + "\"/peak_loci/peak_loci_CTCF_SMC3.txt\"");
            if (ctcfCollapsedPath.length() > 0) {
                executeCommand("bedtools intersect -a " + outputDir + "\"/CTCF_collapsed/cCTCF_in_peak_loci.txt\" -b " + smc3Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev > " + outputDir + "\"/CTCF_collapsed/cCTCF_SMC3_in_peak_loci.txt\"");
                executeCommand("bedtools intersect -a " + outputDir + "\"/CTCF_collapsed/cCTCF_not_in_peak_loci.txt\" -b " + smc3Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev > " + outputDir + "\"/CTCF_collapsed/cCTCF_SMC3_not_in_peak_loci.txt\"");
                executeCommand("bedtools intersect -a " + outputDir + "\"/CTCF_collapsed/unique_cCTCFvCTCF_in_peak_loci.txt\" -b " + smc3Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev > " + outputDir + "\"/CTCF_collapsed/unique_cCTCFvCTCF_SMC3_in_peak_loci.txt\"");
                executeCommand("bedtools intersect -a " + outputDir + "\"/CTCF_collapsed/unique_cCTCFvcCTCF_in_peak_loci.txt\" -b " + smc3Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev > " + outputDir + "\"/CTCF_collapsed/unique_cCTCFvcCTCF_SMC3_in_peak_loci.txt\"");
                executeCommand("bedtools intersect -a " + outputDir + "\"/peak_loci.txt\" -b " + outputDir + "\"/CTCF_collapsed/cCTCF_SMC3_in_peak_loci.txt\" -c | awk '{if ($NF>0) {print $1 \"\\t\" $2 \"\\t\" $3}}' > " + outputDir + "\"/peak_loci/peak_loci_cCTCF_SMC3.txt\"");
            }
            if (rad21Path.length() > 0) {
                executeCommand("bedtools intersect -a " + outputDir + "\"/CTCF/CTCF_RAD21_in_peak_loci.txt\" -b " + smc3Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev > " + outputDir + "\"/CTCF/CTCF_RAD21_SMC3_in_peak_loci.txt\"");
                executeCommand("bedtools intersect -a " + outputDir + "\"/CTCF/CTCF_RAD21_not_in_peak_loci.txt\" -b " + smc3Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev > " + outputDir + "\"/CTCF/CTCF_RAD21_SMC3_not_in_peak_loci.txt\"");
                executeCommand("bedtools intersect -a " + outputDir + "\"/CTCF/unique_CTCFvCTCF_RAD21_in_peak_loci.txt\" -b " + smc3Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev > " + outputDir + "\"/CTCF/unique_CTCFvCTCF_RAD21_SMC3_in_peak_loci.txt\"");
                executeCommand("bedtools intersect -a " + outputDir + "\"/peak_loci.txt\" -b " + outputDir + "\"/CTCF/CTCF_RAD21_SMC3_in_peak_loci.txt\" -c | awk '{if ($NF>0) {print $1 \"\\t\" $2 \"\\t\" $3}}' > " + outputDir + "\"/peak_loci/peak_loci_CTCF_RAD21_SMC3.txt\"");
                if (ctcfCollapsedPath.length() > 0) {
                    executeCommand("bedtools intersect -a " + outputDir + "\"/CTCF_collapsed/cCTCF_RAD21_in_peak_loci.txt\" -b " + smc3Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev > " + outputDir + "\"/CTCF_collapsed/cCTCF_RAD21_SMC3_in_peak_loci.txt\"");
                    executeCommand("bedtools intersect -a " + outputDir + "\"/CTCF_collapsed/cCTCF_RAD21_not_in_peak_loci.txt\" -b " + smc3Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev > " + outputDir + "\"/CTCF_collapsed/cCTCF_RAD21_SMC3_not_in_peak_loci.txt\"");
                    executeCommand("bedtools intersect -a " + outputDir + "\"/CTCF_collapsed/unique_cCTCFvCTCF_RAD21_in_peak_loci.txt\" -b " + smc3Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev > " + outputDir + "\"/CTCF_collapsed/unique_cCTCFvCTCF_RAD21_SMC3_in_peak_loci.txt\"");
                    executeCommand("bedtools intersect -a " + outputDir + "\"/CTCF_collapsed/unique_cCTCFvcCTCF_RAD21_in_peak_loci.txt\" -b " + smc3Path + " -c | awk '{if ($NF>0) {print $0}}' | rev | cut -f2- | rev > " + outputDir + "\"/CTCF_collapsed/unique_cCTCFvcCTCF_RAD21_SMC3_in_peak_loci.txt\"");
                    executeCommand("bedtools intersect -a " + outputDir + "\"/peak_loci.txt\" -b " + outputDir + "\"/CTCF_collapsed/cCTCF_RAD21_SMC3_in_peak_loci.txt\" -c | awk '{if ($NF>0) {print $1 \"\\t\" $2 \"\\t\" $3}}' > " + outputDir + "\"/peak_loci/peak_loci_cCTCF_RAD21_SMC3.txt\"");
                }
            }
        }


    }

    private void executeCommand(String command) {
        Process p;
        try {
            p = Runtime.getRuntime().exec(command);
            p.waitFor();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
