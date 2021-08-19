/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2021 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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
 *  FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.tools.utils.original.stats;

import juicebox.tools.utils.original.FragmentCalculation;
import juicebox.tools.utils.original.mnditerator.AlignmentPair;
import juicebox.tools.utils.original.mnditerator.AlignmentPairLong;

import java.util.List;
import java.util.Map;

public abstract class StatisticsWorker {
    protected static final int TWENTY_KB = 20000;
    protected static final int FIVE_HUNDRED_BP = 500;
    protected static final int FIVE_KB = 5000;
    protected static final int distThreshold = 2000;
    protected static final int mapqValThreshold = 200;
    protected static final long[] bins = {10, 12, 15, 19, 23, 28, 35, 43, 53, 66, 81, 100, 123, 152, 187, 231, 285, 351, 433, 534, 658, 811, 1000, 1233, 1520, 1874, 2310, 2848, 3511, 4329, 5337, 6579, 8111, 10000, 12328, 15199, 18738, 23101, 28480, 35112, 43288, 53367, 65793, 81113, 100000, 123285, 151991, 187382, 231013, 284804, 351119, 432876, 533670, 657933, 811131, 1000000, 1232847, 1519911, 1873817, 2310130, 2848036, 3511192, 4328761, 5336699, 6579332, 8111308, 10000000, 12328467, 15199111, 18738174, 23101297, 28480359, 35111917, 43287613, 53366992, 65793322, 81113083, 100000000, 123284674, 151991108, 187381742, 231012970, 284803587, 351119173, 432876128, 533669923, 657933225, 811130831, 1000000000, 1232846739, 1519911083, 1873817423, 2310129700L, 2848035868L, 3511191734L, 4328761281L, 5336699231L, 6579332247L, 8111308308L, 10000000000L};
    //variables for getting parameters of input file, flags set to default initially
    protected final String siteFile;
    protected final String inFile;
    protected final String danglingJunction;
    protected final String ligationJunction;
    protected final List<String> statsFiles;
    protected final List<Integer> mapqThresholds;
    protected final FragmentCalculation fragmentCalculation;
    protected final StatisticsContainer resultsContainer;

    public StatisticsWorker(String siteFile, List<String> statsFiles, List<Integer> mapqThresholds,
                            String ligationJunction, String inFile, FragmentCalculation fragmentCalculation) {
        this.inFile = inFile;
        this.siteFile = siteFile;
        this.statsFiles = statsFiles;
        this.mapqThresholds = mapqThresholds;
        this.ligationJunction = ligationJunction;
        this.fragmentCalculation = fragmentCalculation;
        this.resultsContainer = new StatisticsContainer();
        this.danglingJunction = ligationJunction.substring(ligationJunction.length() / 2);
    }

    public StatisticsContainer getResultsContainer() {
        return resultsContainer;
    }

    protected boolean processSingleEntry(AlignmentPair pair, String blockKey, boolean multithread) {
        int chr1, chr2, pos1, pos2, frag1, frag2, mapq1, mapq2;
        boolean str1, str2;
        chr1 = pair.getChr1();
        chr2 = pair.getChr2();
        String currentBlock = chr1 + "_" + chr2;
        if (multithread && !currentBlock.equals(blockKey)) {
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

        resultsContainer.unique++;
        //don't count as Hi-C contact if fails mapq or intra fragment test
        for(int ind=0; ind<statsFiles.size(); ind++) {
            boolean countMe = pair.isValid();
            //if(null||null) {do nothing}
            if ((chr1 == chr2) && (frag1 == frag2)) {
                resultsContainer.intraFragment[ind]++;
                countMe = false;
            } else if (!pair.isShort() && mapq1 >= 0 && mapq2 >= 0) {
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
                if (pair instanceof AlignmentPairLong) {
                    AlignmentPairLong longPair = (AlignmentPairLong) pair;
                    String seq1 = longPair.getSeq1();
                    String seq2 = longPair.getSeq2();
                    if ((seq1 != null && seq2 != null) && (seq1.startsWith(danglingJunction) || seq2.startsWith(danglingJunction))) {
                        resultsContainer.dangling[ind]++;
                        isDangling = true;
                    }
                }
                //look at chromosomes
                if (chr1 == chr2) {
                    resultsContainer.intra[ind]++;
                    //determine right/left/inner/outer ordering of chromosomes/strands
                    boolean distGT20KB = posDist >= TWENTY_KB;
                    if (str1 == str2) {
                        if (str1) {
                            populateLIOR(distGT20KB, resultsContainer.right, resultsContainer.rightM, ind, histDist);
                        } else {
                            populateLIOR(distGT20KB, resultsContainer.left, resultsContainer.leftM, ind, histDist);
                        }
                    } else {
                        if (str1) {
                            if (pos1 < pos2) {
                                populateLIOR(distGT20KB, resultsContainer.inner, resultsContainer.innerM, ind, histDist);
                            } else {
                                populateLIOR(distGT20KB, resultsContainer.outer, resultsContainer.outerM, ind, histDist);
                            }
                        } else {
                            if (pos1 < pos2) {
                                populateLIOR(distGT20KB, resultsContainer.outer, resultsContainer.outerM, ind, histDist);
                            } else {
                                populateLIOR(distGT20KB, resultsContainer.inner, resultsContainer.innerM, ind, histDist);
                            }
                        }
                    }
                    //intra reads less than 20KB apart
                    if (posDist < FIVE_HUNDRED_BP) {
                        populateDist(resultsContainer.fiveHundredBPRes, resultsContainer.fiveHundredBPResDangling, ind, isDangling);
                    } else if (posDist < FIVE_KB) {
                        populateDist(resultsContainer.fiveKBRes, resultsContainer.fiveKBResDangling, ind, isDangling);
                    } else if (posDist < TWENTY_KB) {
                        populateDist(resultsContainer.twentyKBRes, resultsContainer.twentyKBResDangling, ind, isDangling);
                    } else {
                        populateDist(resultsContainer.large, resultsContainer.largeDangling, ind, isDangling);
                    }
                } else {
                    populateDist(resultsContainer.inter, resultsContainer.interDangling, ind, isDangling);
                }
                if (pair instanceof AlignmentPairLong) {
                    AlignmentPairLong longPair = (AlignmentPairLong) pair;
                    String seq1 = longPair.getSeq1();
                    String seq2 = longPair.getSeq2();
                    if ((seq1 != null && seq2 != null) && (mapq1 >= 0 && mapq2 >= 0)) {
                        int mapqVal = Math.min(mapq1, mapq2);
                        if (mapqVal <= mapqValThreshold) {
                            resultsContainer.mapQ.get(ind).put(mapqVal, resultsContainer.mapQ.get(ind).getOrDefault(mapqVal, 0L) + 1);
                            if (chr1 == chr2) {
                                resultsContainer.mapQIntra.get(ind).put(mapqVal, resultsContainer.mapQIntra.get(ind).getOrDefault(mapqVal, 0L) + 1);
                            } else {
                                resultsContainer.mapQInter.get(ind).put(mapqVal, resultsContainer.mapQInter.get(ind).getOrDefault(mapqVal, 0L) + 1);
                            }
                        }
                        //read pair contains ligation junction
                        if (seq1.contains(ligationJunction) || seq2.contains(ligationJunction)) {
                            resultsContainer.ligation[ind]++;
                        }
                    }
                }
                //determine distance from nearest HindIII site, add to histogram
                if (!siteFile.contains("none") && fragmentCalculation != null) {
                    try {
                        boolean report = ((chr1 != chr2) || (posDist >= TWENTY_KB));
                        int dist = distHindIII(str1, chr1, pos1, frag1, report, ind);
                        if (dist <= distThreshold) {
                            resultsContainer.hindIII.get(ind).put(dist, resultsContainer.hindIII.get(ind).getOrDefault(dist, 0L) + 1);
                        }
                        dist = distHindIII(str2, chr2, pos2, frag2, report, ind);
                        if (dist <= distThreshold) {
                            resultsContainer.hindIII.get(ind).put(dist, resultsContainer.hindIII.get(ind).getOrDefault(dist, 0L) + 1);
                        }
                    } catch (Exception e) {
                       // System.err.println(e.getLocalizedMessage());
                        // do nothing, fail gracefully; likely a chromosome issue
                    }
                }
                if (pair instanceof AlignmentPairLong && fragmentCalculation != null) {
                    AlignmentPairLong longPair = (AlignmentPairLong) pair;
                    String seq1 = longPair.getSeq1();
                    String seq2 = longPair.getSeq2();
                    if (isDangling) {
                        try {
                            int dist;
                            if (seq1.startsWith(danglingJunction)) {
                                dist = distHindIII(str1, chr1, pos1, frag1, true, ind);
                            } else {
                                dist = distHindIII(str2, chr2, pos2, frag2, true, ind);
                            } //$record[13] =~ m/^$danglingJunction/
                            if (dist == 1) {
                                if (chr1 == chr2) {
                                    if (posDist < TWENTY_KB) {
                                        resultsContainer.trueDanglingIntraSmall[ind]++;
                                    } else {
                                        resultsContainer.trueDanglingIntraLarge[ind]++;
                                    }
                                } else {
                                    resultsContainer.trueDanglingInter[ind]++;
                                }
                            }
                        } catch (Exception e) {
                            // do nothing, fail gracefully; likely a chromosome issue
                        }
                    }
                }
            }
        }
        return false;
    }

    private void populateDist(long[] array, long[] arrayDangling, int ind, boolean isDangling) {
        array[ind]++;
        if (isDangling) {
            arrayDangling[ind]++;
        }
    }

    private void populateLIOR(boolean distGT20KB, long[] array, List<Map<Integer, Long>> arrayM, int ind, int histDist) {
        if (distGT20KB) {
            array[ind]++;
        }
        arrayM.get(ind).put(histDist, arrayM.get(ind).getOrDefault(histDist, 0L) + 1);
    }

    /*
    private void fragmentSearch() {
        try {
            BufferedReader files = new BufferedReader(new FileReader(inFile));
            BufferedWriter fragOut = new BufferedWriter(new FileWriter(outFile, false));
            String file = files.readLine();
            while (file != null) {
                String[] record = file.split("\\s+");
                int indexOne = FragmentCalculation.binarySearch(fragmentCalculation.getSites(localHandler.cleanUpName(record[1])),Integer.parseInt(record[2]));
                int indexTwo = FragmentCalculation.binarySearch(fragmentCalculation.getSites(localHandler.cleanUpName(record[4])),Integer.parseInt(record[5]));
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
    */

    private int distHindIII(boolean strand, int chrIndex, int pos, int frag, boolean rep, int index) {
        //Find distance to nearest HindIII restriction site
        //find upper index of position in sites array via binary search
        //get distance to each end of HindIII fragment
        int dist1;
        int dist2;
        int[] sites = fragmentCalculation.getSites(getChromosomeNameFromIndex(chrIndex));
        int arr = sites.length;
        if (frag >= arr) {
            return 0;
        }
        if (frag == 0) {
            //# first fragment, distance is position
            dist1 = pos;
        } else {
            dist1 = Math.abs(pos - sites[frag - 1]);
        }

        dist2 = Math.abs(pos - sites[frag]);
        //get minimum value -- if (dist1 <= dist2), it's dist1, else dist2
        int retVal = Math.min(dist1, dist2);
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

    protected abstract String getChromosomeNameFromIndex(int chr);

    private static int bSearch(int distance) {
        //search for int distance in array binary
        int lower = 0;
        int upper = bins.length - 1;
        int index;
        while (lower <= upper) {
            index = (lower + upper) / 2;
            if (bins[index] < distance) {
                lower = index + 1;
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
