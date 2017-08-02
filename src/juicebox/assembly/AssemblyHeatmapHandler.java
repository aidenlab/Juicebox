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
import juicebox.track.feature.Feature2DList;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.util.Pair;

import javax.swing.*;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Created by muhammadsaadshamim on 4/17/17.
 */
public class AssemblyHeatmapHandler {

    private static SuperAdapter superAdapter;

    public static void makeChanges(String[] encodedInstructions, SuperAdapter superAdapter) {

        AssemblyHeatmapHandler.superAdapter = superAdapter;
        Feature2DList features = superAdapter.getMainLayer().getAnnotationLayer().getFeatureHandler()
                .getAllVisibleLoops();
        makeAssemblyChanges(features, superAdapter.getHiC().getXContext().getChromosome(), encodedInstructions);
        superAdapter.getContigLayer().getAnnotationLayer().getFeatureHandler().remakeRTree();
        HiCGlobals.assemblyModeEnabled = true;
        //superAdapter.getHiC().clearMatrixZoomDataCache();
        superAdapter.refresh();
    }

    private static void makeAssemblyChanges(Feature2DList features, Chromosome chromosome, String[] encodedInstructions) {
        final String key = Feature2DList.getKey(chromosome, chromosome);

        features.convertFeaturesToContigs(key);
        List<Feature2D> contigs = features.get(key);

        for (String instruction : encodedInstructions) {
            if (instruction.startsWith("-")) {
                parseInversionInstruction(contigs, instruction);
            } else if (instruction.contains("->")) {
                parseTranslationInstruction(contigs, instruction);
            } else {
                showInvalidInstructionErrorMessage(instruction);
            }
        }
        recalculateAllAlterations(contigs);
    }

    private static void parseInversionInstruction(List<Feature2D> contigs, String instruction) {
        String reformattedInstruction = instruction;
        if (!(reformattedInstruction.contains(":"))) {
            reformattedInstruction = reformattedInstruction.concat(":")
                    .concat(reformattedInstruction);
        }
        String[] contigIndices = reformattedInstruction.split(":");
        String startIndexString = contigIndices[0];
        String endIndexString = contigIndices[1];
        if (!(isNumeric(startIndexString) && isNumeric(endIndexString))) {
            showInvalidInstructionErrorMessage(instruction);
            return;
        }
        Integer startIndex = Math.abs(Integer.parseInt(startIndexString));
        Integer endIndex = Math.abs(Integer.parseInt(endIndexString));
        invertMultipleContiguousEntriesAt(contigs, startIndex, endIndex);
    }

    private static void parseTranslationInstruction(List<Feature2D> contigs, String instruction) {
        String[] indices = instruction.split("->");
        if (!(isNumeric(indices[0]) && isNumeric(indices[1]))) {
            showInvalidInstructionErrorMessage(instruction);
            return;
        }
        int currentIndex = Integer.parseInt(indices[0]);
        int newIndex = Integer.parseInt(indices[1]);
        moveFeatureToNewIndex(contigs, currentIndex, newIndex);
    }

    private static void invertSingleEntryAt(List<Feature2D> contigs, int index) {
        if (!(index >= 0 && index < contigs.size())) {
            return;
        }
        ((Contig2D) contigs.get(index)).toggleInversion();
    }

    private static void invertMultipleContiguousEntriesAt(List<Feature2D> contigs, int startIndex, int endIndex) {
        // Invert each of the sub-contigs
        for (int currentIndex = startIndex; currentIndex <= endIndex; currentIndex++) {
            invertSingleEntryAt(contigs, currentIndex);
        }
        // Reverse the order of the sub-contigs
        for (int currentIndex = startIndex; currentIndex < (startIndex + endIndex) / 2.0; currentIndex++) {
            moveFeatureToNewIndex(contigs, currentIndex, startIndex + endIndex - currentIndex);
            moveFeatureToNewIndex(contigs, startIndex + endIndex - currentIndex - 1, currentIndex);
        }
    }

    public static void invertMultipleContiguousEntriesAt(List<Feature2D> selectedFeatures, List<Feature2D> contigs, int startIndex, int endIndex) {


        // Invert each of the sub-contigs
        for (int currentIndex = startIndex; currentIndex <= endIndex; currentIndex++) {
            invertSingleEntryAt(contigs, currentIndex);
        }

        // Reverse the order of the sub-contigs
        for (int currentIndex = startIndex; currentIndex < (startIndex + endIndex) / 2.0; currentIndex++) {
            moveFeatureToNewIndex(contigs, currentIndex, startIndex + endIndex - currentIndex);
            moveFeatureToNewIndex(contigs, startIndex + endIndex - currentIndex - 1, currentIndex);
        }

    }


    private static void moveFeatureToNewIndex(List<Feature2D> contigs, int currentIndex, int newIndex) {
        // http://stackoverflow.com/questions/4938626/moving-items-around-in-an-arraylist
        if (!((currentIndex >= 0 && currentIndex < contigs.size()) && (newIndex >= 0 && newIndex < contigs.size()))) {
            return;
        }
        Feature2D item = contigs.remove(currentIndex);
        contigs.add(newIndex, item);
    }

    private static boolean isNumeric(String s) {
        String numericRegularExpression = "[-+]?\\d*\\.?\\d+";
        return s != null && s.matches(numericRegularExpression);
    }

    private static void showInvalidInstructionErrorMessage(String instruction) {
        JOptionPane.showMessageDialog(superAdapter.getMainWindow(), "Invalid command could not be processed: \""
                + instruction + "\"", "Error Message", JOptionPane.ERROR_MESSAGE);
    }

    private static void recalculateAllAlterations(List<Feature2D> contigs) {
        int i = 0;
        for (Feature2D feature2D : contigs) {
            Contig2D contig2D = feature2D.toContig();
            i = contig2D.setNewStart(i);
        }
    }

    /**
     * @param mergedContigs
     * @param blockList
     * @param binSize
     * @return
     */
    public static Set<Block> filterBlockList(Pair<List<Contig2D>, List<Contig2D>> mergedContigs,
                                             Set<Block> blockList, int binSize) {

        List<Contig2D> xMergedContigs = mergedContigs.getFirst();
        List<Contig2D> yMergedContigs = mergedContigs.getSecond();

        Set<Block> alteredBlockList = new HashSet<>();
        if (xMergedContigs.size() < 1 || yMergedContigs.size() < 1) {
            System.err.println("filter limit " + xMergedContigs.size() + " " + yMergedContigs.size());
            return alteredBlockList;
        }

        for (Block block : blockList) {
            List<ContactRecord> alteredContacts = new ArrayList<>();
            for (ContactRecord record : block.getContactRecords()) {

                int alteredX = getAlteredPositionIfContigOverlapExists(record.getBinX(), binSize, xMergedContigs);
                if (alteredX == -1) {
                    //System.err.println("aXY "+alteredX+" "+alteredY+" ");
                    alteredContacts.add(record);
                    continue;
                }

                int alteredY = getAlteredPositionIfContigOverlapExists(record.getBinY(), binSize, yMergedContigs);
                if (alteredY == -1) {
                    //System.err.println("aXY "+alteredX+" "+alteredY+" ");
                    alteredContacts.add(record);
                } else {
                    //System.out.println("altered ax "+includeXRecord+" "+aX+" and ay "+includeYRecord+" "+aY);
                    if (alteredX > alteredY) {
                        alteredContacts.add(new ContactRecord(alteredY, alteredX, record.getCounts()));
                    } else {
                        alteredContacts.add(new ContactRecord(alteredX, alteredY, record.getCounts()));
                    }
                }
            }
            alteredBlockList.add(new Block(block.getNumber(), alteredContacts));
        }
        //System.out.println("num alters "+alteredBlockList.size());
        return alteredBlockList;
    }

    private static int getAlteredPositionIfContigOverlapExists(int binVal, int binSize, List<Contig2D> xMergedContigs) {
        int genomeVal = binVal * binSize;
        for (Contig2D contig : xMergedContigs) {
            if (contig.hasSomeOriginalOverlapWith(genomeVal)) {
                return contig.getAlteredBinIndex(binVal, binSize);
            }
        }
        return -1;
    }


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
            } else {
                Contig2D result = growingContig.mergeContigs(contig);
                if (result == null) {
                    // cannot be merged
                    if (growingContig != null) mergedContigs.add(growingContig);
                    growingContig = contig.deepCopy().toContig();
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

    public static Set<Block> modifyBlockList(Set<Block> blockList, int binSize, int chr1Idx, int chr2Idx) {
        List<Block> alteredBlockList = new ArrayList<>();
        AssemblyFragmentHandler aFragHandler = superAdapter.getAssemblyStateTracker().getAssemblyHandler();
        for (Block block : blockList) {
            alteredBlockList.add(modifyBlock(block, binSize, chr1Idx, chr2Idx, aFragHandler));
        }
        Set<Block> alteredBlockSet = new HashSet<>(alteredBlockList);
        return alteredBlockSet;
    }

//    private static int getAlteredAsmCoordinate(int chr1Idx, int chr2Idx, int binVal, int binSize,
//                                               AssemblyFragmentHandler aFragHandler) {
//        int originalGenomeVal = binVal * binSize + 1;
//        Contig2D contig2D = aFragHandler.lookupCurrentFragmentForOriginalAsmCoordinate(chr1Idx, chr2Idx, originalGenomeVal);
//        int fragCoordinate = aFragHandler.liftOriginalAsmCoordinateToFragmentCoordinate(contig2D, originalGenomeVal);
//        int currentGenomeVal = aFragHandler.liftFragmentCoordinateToAsmCoordinate(contig2D, fragCoordinate);
//        return currentGenomeVal;
//    }

    public static Block modifyBlock(Block block, int binSize, int chr1Idx, int chr2Idx, AssemblyFragmentHandler aFragHandler) {
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
        block = new Block(block.getNumber(), alteredContacts);
        return block;
    }

    private static int getAlteredAsmBin(int chr1Idx, int chr2Idx, int binVal, int binSize, AssemblyFragmentHandler aFragHandler) {
        int originalBinCenterCoordinate = binVal * binSize + binSize / 2;
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
