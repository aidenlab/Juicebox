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

package juicebox.assembly;

import juicebox.HiCGlobals;
import juicebox.data.Block;
import juicebox.data.ContactRecord;
import juicebox.gui.SuperAdapter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by muhammadsaadshamim on 4/17/17.
 */
public class AssemblyHeatmapHandler {

    private static SuperAdapter superAdapter;
    private static List<Scaffold> listOfOSortedAggregateScaffolds = new ArrayList<>();
//    Does not seem to offer any speedup.
//    private static Scaffold guessScaffold = null;

    public static void setListOfOSortedAggregateScaffolds(List<Scaffold> listOfAggregateScaffolds) {
        AssemblyHeatmapHandler.listOfOSortedAggregateScaffolds = new ArrayList<>(listOfAggregateScaffolds);
        Collections.sort(listOfOSortedAggregateScaffolds, Scaffold.originalStateComparator);
    }

    public static SuperAdapter getSuperAdapter() {
        return AssemblyHeatmapHandler.superAdapter;
    }

    public static void setSuperAdapter(SuperAdapter superAdapter) {
        AssemblyHeatmapHandler.superAdapter = superAdapter;
    }

    public static Block modifyBlock(Block block, String key, int binSize, int chr1Idx, int chr2Idx) {
        //temp fix for AllByAll. TODO: trace this!
        if (chr1Idx == 0 && chr2Idx == 0) {
            binSize = 1000 * binSize; // AllByAll is measured in kb
        }

        List<ContactRecord> alteredContacts = new ArrayList<>();
        for (ContactRecord record : block.getContactRecords()) {

            int alteredAsmBinX = getAlteredAsmBin(record.getBinX(), binSize);
            int alteredAsmBinY = getAlteredAsmBin(record.getBinY(), binSize);

            if (alteredAsmBinX == -1 || alteredAsmBinY == -1) {
                alteredContacts.add(record);
            } else {
                if (alteredAsmBinX > alteredAsmBinY) {
                    alteredContacts.add(new ContactRecord(
                            alteredAsmBinY,
                            alteredAsmBinX, record.getCounts()));
                } else {
                    alteredContacts.add(new ContactRecord(
                            alteredAsmBinX,
                            alteredAsmBinY, record.getCounts()));
                }
            }
        }
        block = new Block(block.getNumber(), alteredContacts, key);
        return block;
    }



    private static int getAlteredAsmBin(int binValue, int binSize) {

        long originalFirstNucleotide = (long) (binValue * HiCGlobals.hicMapScale * binSize + 1);
        long currentFirstNucleotide;
        Scaffold aggregateScaffold = lookUpOriginalAggregateScaffold(originalFirstNucleotide);

        if (aggregateScaffold != null) {
            if (!aggregateScaffold.getInvertedVsInitial()) {
                currentFirstNucleotide = (aggregateScaffold.getCurrentStart() + originalFirstNucleotide - aggregateScaffold.getOriginalStart());
            } else {
                currentFirstNucleotide = (aggregateScaffold.getCurrentEnd() - originalFirstNucleotide + 2 - (long) (HiCGlobals.hicMapScale * binSize) + aggregateScaffold.getOriginalStart());
            }

            return (int) ((currentFirstNucleotide - 1) / (HiCGlobals.hicMapScale * binSize));
        }
        return -1;
    }

    private static Scaffold lookUpOriginalAggregateScaffold(long genomicPos) {
//        Does not seem to offer much advantage
//        if (guessScaffold!=null && guessScaffold.getOriginalStart()<genomicPos && guessScaffold.getOriginalEnd()>=genomicPos){
//            return guessScaffold;
//        }
        Scaffold tmp = new Scaffold("tmp", 1, 1);
        tmp.setOriginalStart(genomicPos);
        int idx = Collections.binarySearch(listOfOSortedAggregateScaffolds, tmp, Scaffold.originalStateComparator);
        if (-idx - 2 >= 0) {
            return listOfOSortedAggregateScaffolds.get(-idx - 2);
        }
        else
            return null;

    }
}