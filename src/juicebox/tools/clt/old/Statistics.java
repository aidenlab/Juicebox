/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab
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

package juicebox.tools.clt.old;

import java.io.*;
import java.util.*;
import java.text.NumberFormat;
import java.nio.charset.StandardCharsets;
import juicebox.data.HiCFileTools;
import juicebox.tools.clt.CommandLineParser;
import juicebox.tools.clt.JuiceboxCLT;
import juicebox.data.ChromosomeHandler;
import juicebox.tools.utils.original.AlignmentPair;
import juicebox.tools.utils.original.AlignmentPairLong;
import juicebox.tools.utils.original.AsciiPairIterator;
import juicebox.tools.utils.original.FragmentCalculation;

public class Statistics extends JuiceboxCLT {
    //variables for getting parameters of input file, flags set to default initially
    private String siteFile ;
    private String statsFile;
    private String inFile;
    private String outFile;
    private ChromosomeHandler localHandler;
    private String ligationJunction = "GATCGATC";
    private int mapqThreshold = 1;


    //Variables for calculating statistics
    private FragmentCalculation chromosomes;
    private final Map<Integer,Integer> hindIII = new HashMap<>();
    private final Map<Integer,Integer> mapQ = new HashMap<>();
    private final Map<Integer,Integer> mapQInter = new HashMap<>();
    private final Map<Integer,Integer> mapQIntra = new HashMap<>();
    private final Map<Integer,Integer> innerM = new HashMap<>();
    private final Map<Integer,Integer> outerM = new HashMap<>();
    private final Map<Integer,Integer> rightM = new HashMap<>();
    private final Map<Integer,Integer> leftM = new HashMap<>();

    private int threePrimeEnd = 0;
    private int fivePrimeEnd = 0;
    private int dangling = 0;
    private int ligation = 0;
    private int inner = 0;
    private int outer = 0;
    private int left = 0;
    private int right = 0;
    private int intra = 0;
    private int inter = 0;
    private int small = 0;
    private int large = 0;
    private int verySmall = 0;
    private int verySmallDangling = 0;
    private int smallDangling = 0;
    private int largeDangling = 0;
    private int interDangling = 0;
    private int trueDanglingIntraSmall = 0;
    private int trueDanglingIntraLarge = 0;
    private int trueDanglingInter = 0;
    private int totalCurrent = 0;
    private int underMapQ = 0;
    private int intraFragment = 0;
    private int unique = 0;
    private static final int posDistThreshold = 20000;
    private static final int distThreshold = 2000;
    private static final int mapqValThreshold = 200;
    private static final long[] bins = {10,12,15,19,23,28,35,43,53,66,81,100,123,152,187,231,285,351,433,534,658,811,1000,1233,1520,1874,2310,2848,3511,4329,5337,6579,8111,10000,12328,15199,18738,23101,28480,35112,43288,53367,65793,81113,100000,123285,151991,187382,231013,284804,351119,432876,533670,657933,811131,1000000,1232847,1519911,1873817,2310130,2848036,3511192,4328761,5336699,6579332,8111308,10000000,12328467,15199111,18738174,23101297,28480359,35111917,43287613,53366992,65793322,81113083,100000000,123284674,151991108,187381742,231012970,284803587,351119173,432876128,533669923,657933225,811130831,1000000000,1232846739,1519911083,1873817423,2310129700L,2848035868L,3511191734L,4328761281L,5336699231L,6579332247L,8111308308L,10000000000L};

    public Statistics(){
        //constructor
        super(getUsage());
    }

    public static String getUsage(){
        return  " Usage: statistics [--ligation NNNN] [-q mapq] <site file> <stats file> <infile> <genome ID> [outfile]\n" +
                " --ligation: ligation junction\n" +
                " -q: mapping quality threshold, do not consider reads < threshold\n" +
                " <site file>: list of HindIII restriction sites, one line per chromosome\n" +
                " <stats file>: output file containing total reads, for library complexity\n"+
                " <infile>: file in intermediate format to calculate statistics on, can be stream\n" +
                " <genome ID>: file to create chromosome handler\n" +
                " [outfile]: output, results of fragment search\n";
    }

    public void readSiteFile(){
        //read in restriction site file and store as multidimensional array q
        try {
            chromosomes = FragmentCalculation.readFragments(siteFile, localHandler);
        }
        catch (IOException error){
            error.printStackTrace();
        }
    }
    public void infileStatistics(){
        //read in infile and calculate statistics
        String danglingJunction = ligationJunction.substring(ligationJunction.length()/2);
        try {
            //create index for AsciiIterator
            Map<String, Integer> chromosomeIndexes = new HashMap<>();
            for (int i = 0; i < localHandler.size(); i++) {
                chromosomeIndexes.put(localHandler.getChromosomeFromIndex(i).getName(), i);
            }
            //iterate through input file
            AsciiPairIterator files = new AsciiPairIterator(inFile, chromosomeIndexes, localHandler);
            while (files.hasNext()) {
                unique++;
                AlignmentPairLong pair = (AlignmentPairLong) files.next();
                int chr1,chr2,pos1,pos2,frag1,frag2,mapq1,mapq2;
                boolean str1,str2;
                String seq1,seq2;
                chr1 = pair.getChr1();
                chr2 = pair.getChr2();
                pos1 = pair.getPos1();
                pos2 = pair.getPos2();
                frag1 = pair.getFrag1();
                frag2 = pair.getFrag2();
                mapq1 = pair.getMapq1();
                mapq2 = pair.getMapq2();
                str1 = pair.getStrand1();
                str2 = pair.getStrand2();
                seq1 = pair.getSeq1();
                seq2 = pair.getSeq2();
                //don't count as Hi-C contact if fails mapq or intra fragment test
                boolean countMe = true;

                if ((chr1==chr2) && (frag1==frag2)){
                    intraFragment++;
                    countMe = false;
                }
                else if (mapq1>=0&&mapq2>=0) {
                    int mapqValue = Math.min(mapq1,mapq2);
                    if (mapqValue < mapqThreshold) {
                        underMapQ++;
                        countMe = false;
                    }
                }

                if (countMe) {
                    totalCurrent++;
                    //position distance
                    int posDist = Math.abs(pos1-pos2);
                    int histDist = bSearch(posDist);
                    boolean isDangling = false;
                    //one part of read pair has unligated end
                    if ((seq1!=null&&seq2!=null) && (seq1.contains(ligationJunction) || seq2.contains(ligationJunction))) {
                        dangling++;
                        isDangling = true;
                    }
                    //look at chromosomes
                    if (chr1==chr2) {
                        intra++;
                        //determine right/left/inner/outer ordering of chromosomes/strands
                        if (str1==str2) {
                            if (str1) {
                                if (posDist >= posDistThreshold) {
                                    right++;
                                }
                                rightM.put(histDist, rightM.getOrDefault(histDist,0) + 1);
                            } else {
                                if (posDist >= posDistThreshold) {
                                    left++;
                                }
                                leftM.put(histDist, leftM.getOrDefault(histDist, 0) + 1);
                            }
                        }
                        else {
                            if (str1) {
                                if (pos1<pos2) {
                                    if (posDist >= posDistThreshold) {
                                        inner++;
                                    }
                                    innerM.put(histDist, innerM.getOrDefault(histDist,0) + 1);
                                }
                                else {
                                    if (posDist >= posDistThreshold) {
                                        outer++;
                                    }
                                    outerM.put(histDist, outerM.getOrDefault(histDist,0) + 1);
                                }
                            }
                            else {
                                if (pos1<pos2) {
                                    if (posDist >= posDistThreshold) {
                                        outer++;
                                    }
                                    outerM.put(histDist, outerM.getOrDefault(histDist,0) + 1);
                                }
                                else {
                                    if (posDist >= posDistThreshold) {
                                        inner++;
                                    }
                                    innerM.put(histDist, innerM.getOrDefault(histDist,0) + 1);
                                }
                            }
                        }
                        //intra reads less than 20KB apart
                        if (posDist < 10) {
                            verySmall++;
                            if (isDangling) {
                                verySmallDangling++;
                            }
                        }
                        else if (posDist < posDistThreshold) {
                            small++;
                            if (isDangling) {
                                smallDangling++;
                            }
                        }
                        else {
                            large++;
                            if (isDangling) {
                                largeDangling++;
                            }
                        }
                    }
                    else {
                        inter++;
                        if (isDangling) {
                            interDangling++;
                        }
                    }
                    if ((seq1!=null&&seq2!=null)&&(mapq1>=0&&mapq2>=0)) {
                        int mapqVal = Math.min(mapq1,mapq2);
                        if (mapqVal <= mapqValThreshold) {
                            mapQ.put(mapqVal, mapQ.getOrDefault(mapqVal,0) + 1);
                            if (chr1==chr2) {
                                mapQIntra.put(mapqVal, mapQIntra.getOrDefault(mapqVal,0) + 1);
                            }
                            else {
                                mapQInter.put(mapqVal, mapQInter.getOrDefault(mapqVal,0) + 1);
                            }
                        }
                        //read pair contains ligation junction
                        if (seq1.contains(ligationJunction) || seq2.contains(ligationJunction)) {
                            ligation++;
                        }
                    }
                    //determine distance from nearest HindIII site, add to histogram
                    if (!siteFile.contains("none")) {
                        boolean report = ((chr1!=chr2) || (posDist >= posDistThreshold));
                        int dist = distHindIII(str1,chr1,pos1,frag1,report);
                        if (dist <= distThreshold) {
                            hindIII.put(dist, hindIII.getOrDefault(dist,0) + 1);
                        }
                        dist = distHindIII(str2,chr2,pos2,frag2,report);
                        if (dist <= distThreshold) {
                            hindIII.put(dist, hindIII.getOrDefault(dist,0) + 1);
                        }
                    }
                    if (isDangling) {
                        int dist;
                        if (seq1.contains(danglingJunction)) {
                            dist = distHindIII(str1,chr1,pos1,frag1,true);
                        }
                        else {
                            dist = distHindIII(str2,chr2,pos2,frag2,true);
                        } //$record[13] =~ m/^$danglingJunction/
                        if (dist == 1) {
                            if (chr1==chr2) {
                                if (posDist < posDistThreshold) {
                                    trueDanglingIntraSmall++;
                                }
                                else {
                                    trueDanglingIntraLarge++;
                                }
                            }
                            else {
                                trueDanglingInter++;
                            }
                        }
                    }
                }
            }
        }
        catch (IOException e){
            e.printStackTrace();
        }
    }

    public void outputStatsFile(){
        boolean seq = false;
        int reads = 1;
        File statFile = new File(statsFile);
        if(statFile.exists()){
            try{
                BufferedReader stats = new BufferedReader(new FileReader(statFile));
                String statsData = stats.readLine();
                while(statsData!= null){
                    if(statsData.contains("Sequenced")){
                        seq = true;
                        String[] tokens = statsData.split(":");
                        reads = Integer.parseInt(tokens[1].replaceAll("[, ]",""));
                    }
                    statsData = stats.readLine();
                }
                stats.close();
            }
            catch (IOException error){
                error.printStackTrace();
            }
        }
        try{
            BufferedWriter statsOut = new BufferedWriter(new FileWriter(statFile, true));
            if(unique==0) {
                unique++;
            }
            statsOut.write("Intra-fragment Reads: " + commify(intraFragment) + " (");
            if(seq) {
                statsOut.write(String.format("%.2f", (float)intraFragment*100/reads) + "%");
            }
            else {
                statsOut.write("(");
            }
            statsOut.write(" / " + String.format("%.2f", (float)intraFragment*100/unique) + "%)" + "\n");

            statsOut.write("Below MAPQ Threshold: " + commify(underMapQ) + " (");
            if(seq) {
                statsOut.write(String.format("%.2f", (float)underMapQ*100/reads) + "%");
            }
            else {
                statsOut.write("(");
            }
            statsOut.write(" / " + String.format("%.2f", (float)underMapQ*100/unique) + "%)" + "\n");

            statsOut.write("Hi-C Contacts: " + commify(totalCurrent) + " (");
            if(seq) {
                statsOut.write(String.format("%.2f", (float)totalCurrent*100/reads) + "%");
            }
            else {
                statsOut.write("(");
            }
            statsOut.write(" / " + String.format("%.2f", (float)totalCurrent*100/unique) + "%)" + "\n");

            statsOut.write(" Ligation Motif Present: " + commify(ligation) + " (");
            if(seq) {
                statsOut.write(String.format("%.2f", (float)ligation*100/reads) + "%");
            }
            else {
                statsOut.write("(");
            }
            statsOut.write(" / " + String.format("%.2f", (float)ligation*100/unique) + "%)" + "\n");

            if((fivePrimeEnd + threePrimeEnd)>0) {
                float f1 = (float)threePrimeEnd*100f/(threePrimeEnd+fivePrimeEnd);
                float f2 = (float)fivePrimeEnd*100f/(threePrimeEnd+fivePrimeEnd);
                statsOut.write(" 3' Bias (Long Range): " + (String.format("%.0f", f1)) + "%");
                statsOut.write(" - " + (String.format("%.0f", f2)) + "%" + "\n");
            }
            else {
                statsOut.write(" 3' Bias (Long Range): 0\\% \\- 0\\%\n");
            }
            if(large>0) {
                statsOut.write(" Pair Type %(L-I-O-R): " + (String.format("%.0f", (float)left*100/large)) + "%");
                statsOut.write(" - " + (String.format("%.0f", (float)inner*100/large)) + "%");
                statsOut.write(" - " + (String.format("%.0f", (float)outer*100/large)) + "%");
                statsOut.write(" - " + (String.format("%.0f", (float)right*100/large)) + "%" + "\n");
            }
            else {
                statsOut.write(" Pair Type %(L-I-O-R): 0\\% - 0\\% - 0\\% - 0\\%\n");
            }

            statsOut.write("Inter-chromosomal: " + commify(inter) + " (");
            if(seq) {
                statsOut.write(String.format("%.2f", (float)inter*100/reads) + "%");
            }
            else {
                statsOut.write("(");
            }
            statsOut.write(" / " + String.format("%.2f", (float)inter*100/unique) + "%)" + "\n");

            statsOut.write("Intra-chromosomal: %s " + commify(intra) + " (");
            if(seq) {
                statsOut.write(String.format("%.2f", (float)intra*100/reads) + "%");
            }
            else {
                statsOut.write("(");
            }
            statsOut.write(" / " + String.format("%.2f", (float)intra*100/unique) + "%)" + "\n");

            statsOut.write("Short Range (<20Kb): " + commify(small) + " (");
            if(seq) {
                statsOut.write(String.format("%.2f", (float)small*100/reads) + "%");
            }
            else {
                statsOut.write("(");
            }
            statsOut.write(" / " + String.format("%.2f", (float)small*100/unique) + "%)" + "\n");

            statsOut.write("Long Range (<20Kb): " + commify(large) + " (");
            if(seq) {
                statsOut.write(String.format("%.2f", (float)large*100/reads) + "%");
            }
            else {
                statsOut.write("(");
            }
            statsOut.write(" / " + String.format("%.2f", (float)large*100/unique)  + "%)" + "\n");
            statsOut.close();
        }
        catch (IOException error){
            error.printStackTrace();
        }
    }

    public void writeHistFile(){
        //separate stats file name
        int index = statsFile.lastIndexOf("\\");
        String statsFilePath = statsFile.substring(0,index+1); //directories
        String statsFileName = statsFile.substring(index+1).replaceAll(".txt",""); //filename
        String histsFile = statsFilePath + statsFileName + "_hists.m";

        try{
            BufferedWriter hist = new BufferedWriter(new FileWriter(histsFile, StandardCharsets.UTF_8, false));
            hist.write("A = [\n");
            for(int i=1; i<=2000;i++){
                int tmp = hindIII.getOrDefault(i,0);
                hist.write(tmp + " ");
            }
            hist.write("\n];\n");

            hist.write("B = [\n");
            for(int i=1; i<=200;i++){
                int tmp = mapQ.getOrDefault(i,0);
                int tmp2 = mapQIntra.getOrDefault(i,0);
                int tmp3 = mapQInter.getOrDefault(i,0);
                hist.write(tmp + " " + tmp2 + " " + tmp3 + "\n");
            }
            hist.write("\n];\n");

            hist.write("D = [\n");
            for (int i=0; i < bins.length; i++) {
                int tmp = innerM.getOrDefault(i,0);
                int tmp2 = outerM.getOrDefault(i,0);
                int tmp3 = rightM.getOrDefault(i,0);
                int tmp4 = leftM.getOrDefault(i,0);
                hist.write(tmp + " " + tmp2 + " " + tmp3 + " " + tmp4 + "\n");
            }
            hist.write("\n];");

            hist.write("x = [\n");
            for (long bin : bins) {
                hist.write(bin + " ");
            }
            hist.write("\n];\n");
            hist.close();
        }
        catch (IOException error) {
            error.printStackTrace();
        }
    }

    private void fragmentSearch() {
        try {
            BufferedReader files = new BufferedReader(new FileReader(inFile));
            BufferedWriter fragOut = new BufferedWriter(new FileWriter(outFile, false));
            String file = files.readLine();
            while (file != null) {
                unique++;
                String[] record = file.split("\\s+");
                int indexOne = FragmentCalculation.binarySearch(chromosomes.getSites(localHandler.cleanUpName(record[1])),Integer.parseInt(record[2]));
                int indexTwo = FragmentCalculation.binarySearch(chromosomes.getSites(localHandler.cleanUpName(record[4])),Integer.parseInt(record[5]));
                fragOut.write(record[0] + " " + record[1] + " " + record[2] + " " + indexOne + " ");
                fragOut.write(record[3] + " " + record[4] + " " + record[5] + " " + indexTwo + " ");
                for (int i = 6; i < record.length; i++) {
                    fragOut.write(record[i] + " ");
                }
                fragOut.write("\n");
                file = files.readLine();
            }
            files.close();
            fragOut.close();
        }
        catch (IOException error){
            error.printStackTrace();
        }
    }
    private int distHindIII(boolean strand, int chr, int pos, int frag, boolean rep){
        //Find distance to nearest HindIII restriction site
        //find upper index of position in sites array via binary search
        //get distance to each end of HindIII fragment
        int dist1;
        int dist2;
        if (frag ==0){
            //# first fragment, distance is position
            dist1 = pos;
        }
        else{
            dist1 = Math.abs(pos - chromosomes.getSites(Integer.toString(chr))[frag-1]);}

        dist2 = Math.abs(pos - chromosomes.getSites(Integer.toString(chr))[frag]);
        //get minimum value -- if (dist1 <= dist2), it's dist1, else dist2
        int retVal = Math.min(dist1,dist2);
        //get which end of the fragment this is, 3' or 5' (depends on strand)
        if ((retVal==dist1)&&(rep)){
            if (strand) {
                fivePrimeEnd++;
            }
            else {
                threePrimeEnd++;
            }
        }
        else if ((retVal==dist2)&&(rep)){
            if (!strand) {
                fivePrimeEnd++;
            }
            else {
                threePrimeEnd++;
            }
        }
        return retVal;
    }

    private String commify(int value){
        return NumberFormat.getNumberInstance(Locale.US).format(value);
    }

    private static int bSearch(int distance){
        //search for int distance in array binary
        int lower = 0;
        int upper = bins.length-1;
        int index;
        while(lower<=upper){
            index = (lower+upper)/2;
            if (bins[index]<distance) {
                lower=index+1;
            }
            else if (bins[index]>distance) {
                upper=index-1;
            }
            else {
                return index;
            }
        }
        return lower;
    }

    @Override
    public void readArguments(String[] args, CommandLineParser parser){
        if(args.length!=5){
            printUsageAndExit();
        }
        //set required arguments to variables
        siteFile = args[1];
        statsFile = args[2];
        inFile = args[3];
        localHandler = HiCFileTools.loadChromosomes(args[4]); //genomeID

        //check for flags
        int mapQT =  parser.getMapqThresholdOption();
        if(mapQT>0){
            mapqThreshold = mapQT;
        }

        String ligJunc = parser.getLigationOption();
        if(ligJunc!=null){
            ligationJunction = ligJunc;
        }
    }

    @Override
    public void run(){
        //if restriction enzyme exists, find the RE distance//
        if(!siteFile.contains("none")){
            readSiteFile();
        }
        infileStatistics();
        outputStatsFile();
        writeHistFile();
        //todo fragmentSearch();
    }
}

