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

import juicebox.tools.clt.CommandLineParser;
import juicebox.tools.clt.JuiceboxCLT;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.text.NumberFormat;
import java.util.*;

public class StatisticsScript extends JuiceboxCLT {
    //variables for getting parameters of input file, flags set to default initially
    String outFile;
    String siteFile ;
    String ligationJunction = "GATCGATC";
    String statsFile;
    int mapqThreshold = 1;
    String inFile;

    String danglingJunction;
    String statsFilePath;
    String statsFileName;
    String histsFile;

    //Variables for calculating statistics
    Map<String,long[]> chromosomes = new HashMap<>();
    Map<Integer,Integer> hindIII = new HashMap<>();
    Map<Integer,Integer> mapQ = new HashMap<>();
    Map<Integer,Integer> mapQInter = new HashMap<>();
    Map<Integer,Integer> mapQIntra = new HashMap<>();
    Map<Integer,Integer> innerM = new HashMap<>();
    Map<Integer,Integer> outerM = new HashMap<>();
    Map<Integer,Integer> rightM = new HashMap<>();
    Map<Integer,Integer> leftM = new HashMap<>();

    int threePrimeEnd = 0;
    int fivePrimeEnd = 0;
    int dangling = 0;
    int ligation = 0;
    int inner = 0;
    int outer = 0;
    int left = 0;
    int right = 0;
    int intra = 0;
    int inter = 0;
    int small = 0;
    int large = 0;
    int verySmall = 0;
    int verySmallDangling = 0;
    int smallDangling = 0;
    int largeDangling = 0;
    int interDangling = 0;
    int trueDanglingIntraSmall = 0;
    int trueDanglingIntraLarge = 0;
    int trueDanglingInter = 0;
    int totalCurrent = 0;
    int underMapQ = 0;
    int intraFragment = 0;
    int unique = 0;
    static int posDistThreshold = 20000;
    static int distThreshold = 2000;
    static int mapqValThreshold = 200;
    static long[] bins = {10,12,15,19,23,28,35,43,53,66,81,100,123,152,187,231,285,351,433,534,658,811,1000,1233,1520,1874,2310,2848,3511,4329,5337,6579,8111,10000,12328,15199,18738,23101,28480,35112,43288,53367,65793,81113,100000,123285,151991,187382,231013,284804,351119,432876,533670,657933,811131,1000000,1232847,1519911,1873817,2310130,2848036,3511192,4328761,5336699,6579332,8111308,10000000,12328467,15199111,18738174,23101297,28480359,35111917,43287613,53366992,65793322,81113083,100000000,123284674,151991108,187381742,231012970,284803587,351119173,432876128,533669923,657933225,811130831,1000000000,1232846739,1519911083,1873817423,2310129700L,2848035868L,3511191734L,4328761281L,5336699231L,6579332247L,8111308308L,10000000000L};

    public StatisticsScript(){
        //constructor
        super(getUsage());
    }

    public static String getUsage(){
        return  " Usage: StatisticsScript [--ligation NNNN] [-q mapq] <site file> <stats file> <infile> [outfile]\n" +
                " --ligation: ligation junction\n" +
                " -q: mapping quality threshold, do not consider reads < threshold\n" +
                " <site file>: list of HindIII restriction sites, one line per chromosome\n" +
                " <stats file>: output file containing total reads, for library complexity\n"+
                " <infile>: file in intermediate format to calculate statistics on, can be stream\n" +
                " [outfile]: output, results of fragment search\n";
    }

    public void readSiteFile(){
        //read in restriction site file and store as multidimensional array q
        try{
            BufferedReader restriction_site = new BufferedReader(new FileReader(siteFile));
            String data = restriction_site.readLine();
            while(data!= null){
                String[] locs = data.split("\\s+");
                String locKey = locs[0];
                String[] locsShift = Arrays.copyOfRange(locs, 1, locs.length);
                long [] locShift = new long[locsShift.length];
                int counter = 0;
                for(String loc: locsShift){
                    locShift[counter] = Integer.parseInt(loc);
                    counter++;}
                chromosomes.put(locKey,locShift);
                //adding keys for fragment reads
                if (locKey.equals("14")){
                    chromosomes.put(locKey+"m",locShift);
                    chromosomes.put(locKey+"p",locShift);
                }
                data = restriction_site.readLine();
            }
            restriction_site.close();}
        catch (IOException error){
            error.printStackTrace();
        }
    }

    public void infileStatistics(){
        //read in infile and calculate statistics
        danglingJunction = ligationJunction.substring(ligationJunction.length()/2);
        try {
            BufferedReader files = new BufferedReader(new FileReader(inFile));
            String file = files.readLine();
            while (file != null) {
                unique++;
                String[] record = file.split("\\s+");
                int num_records = record.length;
                //don't count as Hi-C contact if fails mapq or intra fragment test
                boolean countMe = true;

                if ((record[1].equals(record[5])) && (record[3].equals(record[7]))) {
                    intraFragment++;
                    countMe = false;
                }
                else if (num_records > 8) {
                    int mapqValue = Math.min(Integer.parseInt(record[8]), Integer.parseInt(record[11]));
                    if (mapqValue < mapqThreshold) {
                        underMapQ++;
                        countMe = false;
                    }
                }

                if (countMe) {
                    totalCurrent++;
                    //position distance
                    int posDist = Math.abs(Integer.parseInt(record[2]) - Integer.parseInt(record[6]));
                    int histDist = bSearch(posDist, bins);
                    boolean isDangling = false;
                    //one part of read pair has unligated end
                    if (num_records > 8 && (record[10].contains(ligationJunction) || record[13].contains(ligationJunction))) {
                        dangling++;
                        isDangling = true;
                    }
                    //look at chromosomes
                    if (record[1].equals(record[5])) {
                        intra++;
                        //determine right/left/inner/outer ordering of chromosomes/strands
                        if (record[0].equals(record[4])) {
                            if (Integer.parseInt(record[0]) == 0) {
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
                            if (Integer.parseInt(record[0]) == 0) {
                                if (Integer.parseInt(record[2]) < Integer.parseInt(record[6])) {
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
                                if (Integer.parseInt(record[2]) < Integer.parseInt(record[6])) {
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
                    if (num_records > 8) {
                        int mapqVal = Math.min(Integer.parseInt(record[8]), Integer.parseInt(record[11]));
                        if (mapqVal <= mapqValThreshold) {
                            mapQ.put(mapqVal, mapQ.getOrDefault(mapqVal,0) + 1);
                            if (record[1].equals(record[5])) {
                                mapQIntra.put(mapqVal, mapQIntra.getOrDefault(mapqVal,0) + 1);
                            }
                            else {
                                mapQInter.put(mapqVal, mapQInter.getOrDefault(mapqVal,0) + 1);
                            }
                        }
                        //read pair contains ligation junction
                        if (record[10].contains(ligationJunction) || record[13].contains(ligationJunction)) {
                            ligation++;
                        }
                    }
                    //determine distance from nearest HindIII site, add to histogram
                    if (!siteFile.contains("none")) {
                        boolean report = ((!record[1].equals(record[5])) || (posDist >= posDistThreshold));
                        int dist = distHindIII(record[0], record[1], record[2], record[3], report);
                        if (dist <= distThreshold) {
                            hindIII.put(dist, hindIII.getOrDefault(dist,0) + 1);
                        }
                        dist = distHindIII(record[4], record[5], record[6], record[7], report);
                        if (dist <= distThreshold) {
                            hindIII.put(dist, hindIII.getOrDefault(dist,0) + 1);
                        }
                    }
                    if (isDangling) {
                        int dist;
                        if (record[10].contains(danglingJunction)) {
                            dist = distHindIII(record[0], record[1], record[2], record[3], true);
                        }
                        else {
                            dist = distHindIII(record[4], record[5], record[6], record[7], true);
                        } //$record[13] =~ m/^$danglingJunction/
                        if (dist == 1) {
                            if (record[1].equals(record[5])) {
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
                file = files.readLine();
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
                float f1 = (float)threePrimeEnd*100/(threePrimeEnd+fivePrimeEnd);
                float f2 = (float)fivePrimeEnd*100/(threePrimeEnd+fivePrimeEnd);
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
        int index = statsFile.lastIndexOf("\\");
        statsFilePath = statsFile.substring(0,index+1); //directories
        statsFileName = statsFile.substring(index+1).replaceAll(".txt",""); //filename
        histsFile = statsFilePath + statsFileName + "_hists.m";
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
                int indexOne = bSearch(Integer.parseInt(record[2]),chromosomes.get(record[1]));
                int indexTwo = bSearch(Integer.parseInt(record[5]),chromosomes.get(record[4]));
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
    private int distHindIII(String record_0, String record_1, String record_2, String record_3, boolean rep){
        //Find distance to nearest HindIII restriction site
        //find upper index of position in sites array via binary search
        int index = Integer.parseInt(record_3);
        //get distance to each end of HindIII fragment
        int dist1;
        int dist2;
        if (index==0){
            //# first fragment, distance is position
            dist1 = Integer.parseInt(record_2);
        }
        else{
            dist1 = Math.abs(Integer.parseInt(record_2) - (int) chromosomes.get(record_1)[index-1]);
        }
        if(chromosomes.get(record_1).length>index) {
            dist2 = Math.abs(Integer.parseInt(record_2) - (int) chromosomes.get(record_1)[index]);
        }
        else{
            dist2 = 0;
        }
        //get minimum value -- if (dist1 <= dist2), it's dist1, else dist2
        int retVal = Math.min(dist1,dist2);
        //get which end of the fragment this is, 3' or 5' (depends on strand)
        if ((retVal==dist1)&&(rep)){
            if (Integer.parseInt(record_0) == 0) {
                fivePrimeEnd++;
            }
            else {
                threePrimeEnd++;
            }
        }
        else if ((retVal==dist2)&&(rep)){
            if (Integer.parseInt(record_0) == 16) {
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

    private static int bSearch(int distance, long[] bin){
        //search for int distance in array binary
        int lower = 0;
        int upper = bin.length-1;
        int index;
        while(lower<=upper){
            index = (lower+upper)/2;
            if (bin[index]<distance) {
                lower=index+1;
            }
            else if (bin[index]>distance) {
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
        if(args.length!=4){
            printUsageAndExit();
        }

        siteFile = args[1];
        statsFile = args[2];
        inFile = args[3];

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

