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

import juicebox.HiCGlobals;
import juicebox.data.Block;
import juicebox.data.ContactRecord;
import juicebox.gui.SuperAdapter;
import juicebox.track.feature.Contig2D;
import juicebox.track.feature.Feature2D;

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
//        System.out.println(block.getNumber());
        List<ContactRecord> alteredContacts = new ArrayList<>();
        for (ContactRecord record : block.getContactRecords()) {

//            if (block.getNumber()==3){
//                System.out.println("before: "+record.getBinX()+" "+record.getBinY());
//            }
//            int alteredAsmBinX = getAlteredAsmBin(chr1Idx, chr2Idx, record.getBinX(), binSize, aFragHandler);
//            int alteredAsmBinY = getAlteredAsmBin(chr1Idx, chr2Idx, record.getBinY(), binSize, aFragHandler);

            int alteredAsmBinX = newGetAlteredAsmBin(chr1Idx, chr2Idx, record.getBinX(), binSize, aFragHandler);
            int alteredAsmBinY = newGetAlteredAsmBin(chr1Idx, chr2Idx, record.getBinY(), binSize, aFragHandler);

//            if (block.getNumber()==3){
//                System.out.println("after: "+alteredAsmBinX+" "+alteredAsmBinY);
//            }

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
        long originalBinCenterCoordinate = (long) ((binValue * binSize + binSize / 2) * HiCGlobals.hicMapScale);
        //Contig2D contig2D = aFragHandler.lookupCurrentFragmentForOriginalAsmCoordinate(chr1Idx, chr2Idx, originalBinCenterCoordinate);
        //System.out.println("start lookup fragmentProperty: "+ Calendar.getInstance().getTime());
        FragmentProperty fragmentProperty = aFragHandler.newLookupCurrentFragmentForOriginalAsmCoordinate(chr1Idx, chr2Idx, originalBinCenterCoordinate);
        //System.out.println("finish lookup fragmentProperty: "+fragmentProperty.getName()+" "+ Calendar.getInstance().getTime());

        //int fragCoordinate = aFragHandler.liftOriginalAsmCoordinateToFragmentCoordinate(contig2D, originalBinCenterCoordinate);
        long fragCoordinate = aFragHandler.newLiftOriginalAsmCoordinateToFragmentCoordinate(fragmentProperty, originalBinCenterCoordinate);
        //System.out.println("finished lookup of fragment coordinate "+fragCoordinate+" "+Calendar.getInstance().getTime());

//        int currentBinCenterCoordinate = aFragHandler.liftFragmentCoordinateToAsmCoordinate(contig2D, fragCoordinate);
        long currentBinCenterCoordinate = aFragHandler.newLiftFragmentCoordinateToAsmCoordinate(fragmentProperty, fragCoordinate);

        //System.out.println("finished lookup of fragment coordinate "+fragCoordinate+" "+Calendar.getInstance().getTime());

        if (currentBinCenterCoordinate == -1) {
            return -1;
        } else {
            return (int) ((currentBinCenterCoordinate - binSize / 2) / binSize);
        }
    }

    private static int newGetAlteredAsmBin(int chr1Idx, int chr2Idx, int binValue, int binSize, AssemblyFragmentHandler aFragHandler) {

        int originalBinCenterCoordinate = (binValue * binSize + binSize / 2);
        int currentBinCenterCoordinate;
        net.sf.jsi.Rectangle currentWindow = new net.sf.jsi.Rectangle(
                originalBinCenterCoordinate,
                originalBinCenterCoordinate,
                originalBinCenterCoordinate,
                originalBinCenterCoordinate);

        List<Feature2D> containedFeatures = aFragHandler.getOriginalAggregateFeature2DHandler().getIntersectingFeatures(chr1Idx, chr2Idx, currentWindow, true);

        if (!containedFeatures.isEmpty()) {

            int aggregateScaffoldId = Integer.parseInt(containedFeatures.get(0).getAttribute("Scaffold name")) - 1;
            //System.out.println(aggregateScaffoldId);
            FragmentProperty aggregateFragmentProperty = aFragHandler.getListOfAggregateScaffoldProperties().get(aggregateScaffoldId);
            if (!aggregateFragmentProperty.isInvertedVsInitial()) {
                currentBinCenterCoordinate = (int) (aggregateFragmentProperty.getCurrentStart() / HiCGlobals.hicMapScale) + originalBinCenterCoordinate - (int) (aggregateFragmentProperty.getInitialStart() / HiCGlobals.hicMapScale);
            } else {
                currentBinCenterCoordinate = (int) (aggregateFragmentProperty.getCurrentStart() / HiCGlobals.hicMapScale - originalBinCenterCoordinate + aggregateFragmentProperty.getInitialEnd() / HiCGlobals.hicMapScale);
            }

            return Math.round((currentBinCenterCoordinate - binSize / 2) / binSize);

        } else {
//            System.out.println("I am here");
            return -1;
        }
    }
}