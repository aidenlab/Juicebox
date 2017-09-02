/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2017 Broad Institute, Aiden Lab
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

import juicebox.data.Block;
import juicebox.data.ContactRecord;
import juicebox.gui.SuperAdapter;
import juicebox.track.feature.Contig2D;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

/**
 * Created by muhammadsaadshamim on 4/17/17.
 */
public class AssemblyHeatmapHandler {

    private static SuperAdapter superAdapter;

    /**
     * if neighboring contigs are not inverted, share original continuity
     * and share current continuity, they can essentially be merged
     * this will reduce the number of contigs, and improve speed
     *
     * @param currentContigs
     * @return mergedContigs
     */
    public static List<Contig2D> mergeRedundantContiguousContigs(List<Contig2D> currentContigs) {

        List<Contig2D> mergedContigs = new ArrayList<>();
        Contig2D growingContig = null;

        for (Contig2D contig : currentContigs) {
            if (growingContig == null) {
                growingContig = contig.deepCopy().toContig();
                continue;
            } else {
                Contig2D result = growingContig.mergeContigs(contig);
                if (result == null) {
                    // cannot be merged
                    if (growingContig != null) mergedContigs.add(growingContig);
                    growingContig = contig.deepCopy().toContig();
                    continue;
                } else {
                    growingContig = result;
                }
            }
        }
        if (growingContig != null) mergedContigs.add(growingContig);

        return new ArrayList<>(new HashSet<>(mergedContigs));
    }

    public static SuperAdapter getSuperAdapter() {
        return AssemblyHeatmapHandler.superAdapter;
    }

    public static void setSuperAdapter(SuperAdapter superAdapter) {
        AssemblyHeatmapHandler.superAdapter = superAdapter;
    }

    public static Block modifyBlock(Block block, String key, int binSize, int chr1Idx, int chr2Idx, AssemblyFragmentHandler aFragHandler) {
        //TODO: do some filtering here
        List<ContactRecord> alteredContacts = new ArrayList<>();
        for (ContactRecord record : block.getContactRecords()) {

            int alteredAsmBinX = getAlteredAsmBin(chr1Idx, chr2Idx, record.getBinX(), binSize, aFragHandler);
            int alteredAsmBinY = getAlteredAsmBin(chr1Idx, chr2Idx, record.getBinY(), binSize, aFragHandler);

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

    private static int getAlteredAsmBin(int chr1Idx, int chr2Idx, int binValue, int binSize, AssemblyFragmentHandler aFragHandler) {
        int originalBinCenterCoordinate = binValue * binSize + binSize / 2;
        Contig2D contig2D = aFragHandler.lookupCurrentFragmentForOriginalAsmCoordinate(chr1Idx, chr2Idx, originalBinCenterCoordinate);
        int fragCoordinate = aFragHandler.liftOriginalAsmCoordinateToFragmentCoordinate(contig2D, originalBinCenterCoordinate);
        int currentBinCenterCoordinate = aFragHandler.liftFragmentCoordinateToAsmCoordinate(contig2D, fragCoordinate);
        if (currentBinCenterCoordinate == -1) {
            return -1;
        } else {
            return (currentBinCenterCoordinate - binSize / 2) / binSize;
        }
    }
}