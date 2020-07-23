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

import juicebox.data.ChromosomeHandler;
import juicebox.tools.utils.original.*;
import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class StatisticsWorker {
    //variables for getting parameters of input file, flags set to default initially
    private String outFile;
    private long mndIndexStart = -1L;
    private final String siteFile;
    private final String inFile;
    private final String danglingJunction;
    private final String ligationJunction;
    private final List<String> statsFiles;
    private final List<Integer> mapqThresholds;
    private final FragmentCalculation chromosomes;
    private final ChromosomeHandler localHandler;
    private final StatisticsContainer resultsContainer;

    private static final int posDistThreshold = 20000;
    private static final int distThreshold = 2000;
    private static final int mapqValThreshold = 200;
    private static final long[] bins = {10,12,15,19,23,28,35,43,53,66,81,100,123,152,187,231,285,351,433,534,658,811,1000,1233,1520,1874,2310,2848,3511,4329,5337,6579,8111,10000,12328,15199,18738,23101,28480,35112,43288,53367,65793,81113,100000,123285,151991,187382,231013,284804,351119,432876,533670,657933,811131,1000000,1232847,1519911,1873817,2310130,2848036,3511192,4328761,5336699,6579332,8111308,10000000,12328467,15199111,18738174,23101297,28480359,35111917,43287613,53366992,65793322,81113083,100000000,123284674,151991108,187381742,231012970,284803587,351119173,432876128,533669923,657933225,811130831,1000000000,1232846739,1519911083,1873817423,2310129700L,2848035868L,3511191734L,4328761281L,5336699231L,6579332247L,8111308308L,10000000000L};

    public StatisticsWorker(String siteFile, List<String> statsFiles, List<Integer> mapqThresholds, String ligationJunction,
                            String inFile, ChromosomeHandler localHandler, FragmentCalculation chromosomes){
        //default constructor for non-multithreading inputs, no mnd-indexing
        this.inFile = inFile;
        this.siteFile = siteFile;
        this.statsFiles = statsFiles;
        this.mapqThresholds = mapqThresholds;
        this.ligationJunction = ligationJunction;
        this.localHandler = localHandler;
        this.chromosomes = chromosomes;
        this.resultsContainer = new StatisticsContainer();

        this.danglingJunction = ligationJunction.substring(ligationJunction.length()/2);
        resultsContainer.initMaps();
    }

    public StatisticsWorker(String siteFile, List<String> statsFiles, List<Integer> mapqThresholds, String ligationJunction,
                            String inFile, ChromosomeHandler localHandler, Long mndIndexStart, FragmentCalculation chromosomes){
        //constructor for multithreaded, get command line inputs
        this(siteFile, statsFiles, mapqThresholds, ligationJunction, inFile, localHandler, chromosomes);
        this.mndIndexStart = mndIndexStart;
    }

    public StatisticsContainer getResultsContainer(){
        return resultsContainer;
    }

    public void infileStatistics(){
        //read in infile and calculate statistics
        try {
            //create index for AsciiIterator
            Map<String, Integer> chromosomeIndexes = new HashMap<>();
            for (int i = 0; i < localHandler.size(); i++) {
                chromosomeIndexes.put(localHandler.getChromosomeFromIndex(i).getName(), i);
            }
            //iterate through input file
            AsciiPairIterator files;
            if (mndIndexStart<0) {
                        files = new AsciiPairIterator(inFile, chromosomeIndexes, localHandler);
                        while(files.hasNext()){
                            AlignmentPairLong pair = (AlignmentPairLong) files.next();
                            processSingleEntry(pair, "", false);
                        }
            } else {
                files = new AsciiPairIterator(inFile, chromosomeIndexes, mndIndexStart, localHandler);
                if(files.hasNext()) {
                    AlignmentPairLong firstPair = (AlignmentPairLong) files.next();
                    String previousBlock = firstPair.getChr1() + "_" + firstPair.getChr2();
                    processSingleEntry(firstPair, previousBlock, true);
                    while (files.hasNext()) {
                        AlignmentPairLong pair = (AlignmentPairLong) files.next();
                        if (processSingleEntry(pair, previousBlock, true)) {
                            break;
                        }
                    }
                }
            }
        }
        catch (IOException e){
            e.printStackTrace();
        }
    }
    private boolean processSingleEntry(AlignmentPairLong pair, String blockKey, boolean multithread){
        int chr1,chr2,pos1,pos2,frag1,frag2,mapq1,mapq2;
        boolean str1,str2;
        String seq1,seq2;
        chr1 = pair.getChr1();
        chr2 = pair.getChr2();
        String currentBlock = chr1 + "_" + chr2;
        if(!currentBlock.equals(blockKey)&& multithread){
            return true;
        }
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

        resultsContainer.unique++;
        //don't count as Hi-C contact if fails mapq or intra fragment test
        for(int ind=0; ind<statsFiles.size(); ind++) {
            boolean countMe = true;
            //if(null||null) {do nothing}
            if ((chr1 == chr2) && (frag1 == frag2)) {
                resultsContainer.intraFragment[ind]++;
                countMe = false;
            } else if (mapq1 >= 0 && mapq2 >= 0) {
                int mapqValue = Math.min(mapq1, mapq2);
                if (mapqValue < mapqThresholds.get(ind)) {
                    resultsContainer.underMapQ[ind]++;
                    countMe = false;
                }
            }
            //calculate statistics for mapq threshold
            if (countMe && statsFiles.get(ind) != null) {
                resultsContainer.totalCurrent[ind]++;
                //position distance
                int posDist = Math.abs(pos1 - pos2);
                int histDist = bSearch(posDist);
                boolean isDangling = false;
                //one part of read pair has unligated end
                if ((seq1 != null && seq2 != null) && (seq1.startsWith(danglingJunction) || seq2.startsWith(danglingJunction))) {
                    resultsContainer.dangling[ind]++;
                    isDangling = true;
                }
                //look at chromosomes
                if (chr1 == chr2) {
                    resultsContainer.intra[ind]++;
                    //determine right/left/inner/outer ordering of chromosomes/strands
                    if (str1 == str2) {
                        if (str1) {
                            if (posDist >= posDistThreshold) {
                                resultsContainer.right[ind]++;
                            }
                            resultsContainer.rightM[ind].put(histDist, resultsContainer.rightM[ind].getOrDefault(histDist, 0) + 1);
                        } else {
                            if (posDist >= posDistThreshold) {
                                resultsContainer.left[ind]++;
                            }
                            resultsContainer.leftM[ind].put(histDist, resultsContainer.leftM[ind].getOrDefault(histDist, 0) + 1);
                        }
                    } else {
                        if (str1) {
                            if (pos1 < pos2) {
                                if (posDist >= posDistThreshold) {
                                    resultsContainer.inner[ind]++;
                                }
                                resultsContainer.innerM[ind].put(histDist, resultsContainer.innerM[ind].getOrDefault(histDist, 0) + 1);
                            } else {
                                if (posDist >= posDistThreshold) {
                                    resultsContainer.outer[ind]++;
                                }
                                resultsContainer.outerM[ind].put(histDist, resultsContainer.outerM[ind].getOrDefault(histDist, 0) + 1);
                            }
                        } else {
                            if (pos1 < pos2) {
                                if (posDist >= posDistThreshold) {
                                    resultsContainer.outer[ind]++;
                                }
                                resultsContainer.outerM[ind].put(histDist, resultsContainer.outerM[ind].getOrDefault(histDist, 0) + 1);
                            } else {
                                if (posDist >= posDistThreshold) {
                                    resultsContainer.inner[ind]++;
                                }
                                resultsContainer.innerM[ind].put(histDist, resultsContainer.innerM[ind].getOrDefault(histDist, 0) + 1);
                            }
                        }
                    }
                    //intra reads less than 20KB apart
                    if (posDist < 10) {
                        resultsContainer.verySmall[ind]++;
                        if (isDangling) {
                            resultsContainer.verySmallDangling[ind]++;
                        }
                    } else if (posDist < 1000) {
                        resultsContainer.oneKBRes[ind]++;
                        if (isDangling) {
                            resultsContainer.oneKBResDangling[ind]++;
                        }
                    } else if (posDist < 2000) {
                        resultsContainer.twoKBRes[ind]++;
                        if (isDangling) {
                            resultsContainer.twoKBResDangling[ind]++;
                        }
                    } else if (posDist < 5000) {
                        resultsContainer.fiveKBRes[ind]++;
                        if (isDangling) {
                            resultsContainer.fiveKBResDangling[ind]++;
                        }
                    } else if (posDist < posDistThreshold) {
                        resultsContainer.small[ind]++;
                        if (isDangling) {
                            resultsContainer.smallDangling[ind]++;
                        }
                    } else {
                        resultsContainer.large[ind]++;
                        if (isDangling) {
                            resultsContainer.largeDangling[ind]++;
                        }
                    }
                } else {
                    resultsContainer.inter[ind]++;
                    if (isDangling) {
                        resultsContainer.interDangling[ind]++;
                    }
                }
                if ((seq1 != null && seq2 != null) && (mapq1 >= 0 && mapq2 >= 0)) {
                    int mapqVal = Math.min(mapq1, mapq2);
                    if (mapqVal <= mapqValThreshold) {
                        resultsContainer.mapQ[ind].put(mapqVal, resultsContainer.mapQ[ind].getOrDefault(mapqVal, 0) + 1);
                        if (chr1 == chr2) {
                            resultsContainer.mapQIntra[ind].put(mapqVal, resultsContainer.mapQIntra[ind].getOrDefault(mapqVal, 0) + 1);
                        } else {
                            resultsContainer.mapQInter[ind].put(mapqVal, resultsContainer.mapQInter[ind].getOrDefault(mapqVal, 0) + 1);
                        }
                    }
                    //read pair contains ligation junction
                    if (seq1.contains(ligationJunction) || seq2.contains(ligationJunction)) {
                        resultsContainer.ligation[ind]++;
                    }
                }
                //determine distance from nearest HindIII site, add to histogram
                if (!siteFile.contains("none")) {
                    boolean report = ((chr1 != chr2) || (posDist >= posDistThreshold));
                    int dist = distHindIII(str1, chr1, pos1, frag1, report, ind);
                    if (dist <= distThreshold) {
                        resultsContainer.hindIII[ind].put(dist, resultsContainer.hindIII[ind].getOrDefault(dist, 0) + 1);
                    }
                    dist = distHindIII(str2, chr2, pos2, frag2, report, ind);
                    if (dist <= distThreshold) {
                        resultsContainer.hindIII[ind].put(dist, resultsContainer.hindIII[ind].getOrDefault(dist, 0) + 1);
                    }
                }
                if (isDangling) {
                    int dist;
                    if (seq1.startsWith(danglingJunction)) {
                        dist = distHindIII(str1, chr1, pos1, frag1, true, ind);
                    } else {
                        dist = distHindIII(str2, chr2, pos2, frag2, true, ind);
                    } //$record[13] =~ m/^$danglingJunction/
                    if (dist == 1) {
                        if (chr1 == chr2) {
                            if (posDist < posDistThreshold) {
                                resultsContainer.trueDanglingIntraSmall[ind]++;
                            } else {
                                resultsContainer.trueDanglingIntraLarge[ind]++;
                            }
                        } else {
                            resultsContainer.trueDanglingInter[ind]++;
                        }
                    }
                }
            }
        }
        return false;
    }

    private void fragmentSearch() {
        try {
            BufferedReader files = new BufferedReader(new FileReader(inFile));
            BufferedWriter fragOut = new BufferedWriter(new FileWriter(outFile, false));
            String file = files.readLine();
            while (file != null) {
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
    private int distHindIII(boolean strand, int chr, int pos, int frag, boolean rep, int index){
        //Find distance to nearest HindIII restriction site
        //find upper index of position in sites array via binary search
        //get distance to each end of HindIII fragment
        int dist1;
        int dist2;
        int arr = chromosomes.getSites(localHandler.getChromosomeFromIndex(chr).getName()).length;
        if(frag>=arr){
            return 0;
        }
        if (frag ==0){
            //# first fragment, distance is position
            dist1 = pos;
        }
        else{
            dist1 = Math.abs(pos - chromosomes.getSites(localHandler.getChromosomeFromIndex(chr).getName())[frag-1]);}

        dist2 = Math.abs(pos - chromosomes.getSites(localHandler.getChromosomeFromIndex(chr).getName())[frag]);
        //get minimum value -- if (dist1 <= dist2), it's dist1, else dist2
        int retVal = Math.min(dist1,dist2);
        //get which end of the fragment this is, 3' or 5' (depends on strand)
            if ((retVal == dist1) && (rep)) {
                if (strand) {
                    resultsContainer.fivePrimeEnd[index]++;
                } else {
                    resultsContainer.threePrimeEnd[index]++;
                }
            } else if ((retVal == dist2) && (rep)) {
                if (!strand) {
                    resultsContainer.fivePrimeEnd[index]++;
                } else {
                    resultsContainer.threePrimeEnd[index]++;
                }
            }
        return retVal;
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

}
