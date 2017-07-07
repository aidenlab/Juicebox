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

import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.data.Block;
import juicebox.data.ContactRecord;
import juicebox.data.MatrixZoomData;
import juicebox.gui.SuperAdapter;
import juicebox.mapcolorui.Feature2DHandler;
import juicebox.track.feature.*;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import org.broad.igv.feature.Chromosome;

import javax.swing.*;
import java.util.*;

/**
 * Created by muhammadsaadshamim on 4/17/17.
 */
public class AssemblyIntermediateProcessor {

    private static SuperAdapter superAdapter;

    public static void makeChanges(String[] encodedInstructions, SuperAdapter superAdapter) {

        AssemblyIntermediateProcessor.superAdapter = superAdapter;
        Feature2DList features = superAdapter.getContigLayer().getAnnotationLayer().getFeatureHandler()
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

    public static void invertMultipleContiguousEntriesAt(List<Feature2D> contigs, int startIndex, int endIndex) {
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

    public static AssemblyHandler invertMultipleContiguousEntriesAt(List<Feature2D> selectedFeatures, List<Feature2D> contigs, int startIndex, int endIndex) {

        AssemblyHandler assemblyHandler = superAdapter.getAssemblyStateTracker().getNewAssemblyHandler();
        assemblyHandler.invertSelection(selectedFeatures);

        // Invert each of the sub-contigs
        for (int currentIndex = startIndex; currentIndex <= endIndex; currentIndex++) {
            invertSingleEntryAt(contigs, currentIndex);
        }

        // Reverse the order of the sub-contigs
        for (int currentIndex = startIndex; currentIndex < (startIndex + endIndex) / 2.0; currentIndex++) {
            moveFeatureToNewIndex(contigs, currentIndex, startIndex + endIndex - currentIndex);
            moveFeatureToNewIndex(contigs, startIndex + endIndex - currentIndex - 1, currentIndex);
        }

        return assemblyHandler;
    }



    public static void splitContig(Feature2D originalContig, Feature2D debrisContig, SuperAdapter superAdapter, HiC hic) {

        AnnotationLayer contigLayer = superAdapter.getContigLayer().getAnnotationLayer();
        Feature2D firstFragment;
        Feature2D secondFragment;
        Feature2D thirdFragment;
        int chr1Idx = hic.getXContext().getChromosome().getIndex();
        int chr2Idx = hic.getYContext().getChromosome().getIndex();

        if (originalContig.overlapsWith(debrisContig)) {

            contigLayer.getFeatureList().checkAndRemoveFeature(chr1Idx, chr2Idx, originalContig);

            firstFragment = originalContig.deepCopy();
            firstFragment.setEnd1(debrisContig.getStart1());
            firstFragment.setEnd2(debrisContig.getStart2());

            secondFragment = originalContig.deepCopy();
            secondFragment.setStart1(debrisContig.getStart1());
            secondFragment.setStart2(debrisContig.getStart2());
            secondFragment.setEnd1(debrisContig.getEnd1());
            secondFragment.setEnd2(debrisContig.getEnd2());

            thirdFragment = originalContig.deepCopy();
            thirdFragment.setStart1(debrisContig.getEnd1());
            thirdFragment.setStart2(debrisContig.getEnd2());

            contigLayer.add(chr1Idx, chr2Idx, firstFragment);
            contigLayer.add(chr1Idx, chr2Idx, secondFragment);
            contigLayer.add(chr1Idx, chr2Idx, thirdFragment);

            //remake tree or something
            contigLayer.getFeatureHandler().remakeRTree();
            superAdapter.refresh();
        } else {
            System.out.println("error splitting contigs");
        }
    }

    public static void splitGroup(List<Feature2D> contigs) {
        AssemblyHandler assemblyHandler = superAdapter.getAssemblyStateTracker().getNewAssemblyHandler();
        assemblyHandler.splitGroup(contigs);
        superAdapter.getAssemblyStateTracker().assemblyActionPerformed(assemblyHandler);
    }

    public static void mergeGroup(List<Feature2D> contigs) {
        String atributeName = "Scaffold Number";
        AnnotationLayerHandler groupLayer = superAdapter.getActiveLayerHandler(); //todo make check for group layer
        int startingIndex = Integer.parseInt(contigs.get(0).getAttribute(atributeName));
        System.out.println(startingIndex);
        AssemblyHandler assemblyHandler = superAdapter.getAssemblyStateTracker().getNewAssemblyHandler();
        assemblyHandler.mergeGroup(startingIndex, contigs);
        superAdapter.getAssemblyStateTracker().assemblyActionPerformed(assemblyHandler);

    }

    public static void moveFeatureToNewIndex(List<Feature2D> contigs, int currentIndex, int newIndex) {
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

    public static void recalculateAllAlterations(List<Feature2D> contigs) {
        int i = 0;
        for (Feature2D feature2D : contigs) {
            Contig2D contig2D = feature2D.toContig();
            i = contig2D.setNewStart(i);
        }
    }

    public static List<Contig2D> retrieveRelevantBlocks(MatrixZoomData mzd, List<Integer> blocksToLoad,
                                                        List<Block> blockList, Chromosome chr1, Chromosome chr2,
                                                        int binX1, int binY1, int binX2, int binY2, int blockBinCount,
                                                        HiCZoom zoom, NormalizationType no) {
        Feature2DHandler handler = superAdapter.getContigLayer().getAnnotationLayer().getFeatureHandler();
        net.sf.jsi.Rectangle currentWindow = new net.sf.jsi.Rectangle(binX1 * zoom.getBinSize(),
                binY1 * zoom.getBinSize(), binX2 * zoom.getBinSize(), binY2 * zoom.getBinSize());
        handler.getContainedFeatures(chr1.getIndex(), chr2.getIndex(), currentWindow);
        Feature2DList features = handler.getAllVisibleLoops();
        final String keyF = Feature2DList.getKey(chr1, chr2);
        List<Contig2D> contigs = new ArrayList<>();
        for (Feature2D entry : features.get(keyF)) {
            contigs.add(entry.toContig());
        }
        Collections.sort(contigs);

        List<Contig2D> actuallyNeededContigs = new ArrayList<>();
        for (Contig2D contig : contigs) {
            int cStart = contig.getStart1() / zoom.getBinSize();
            int cEnd = contig.getEnd1() / zoom.getBinSize();
            //System.out.println("c "+cStart+" "+cEnd);

            if (cEnd < binX1 && cEnd < binY1) {
                continue;
            }
            if (cStart > binX2 && cStart > binY2) {
                break;
            }

            if ((cStart >= binX1 && cStart <= binX2)
                    || (cEnd >= binX1 && cEnd <= binX2)
                    || (cStart >= binY1 && cStart <= binY2)
                    || (cEnd >= binY1 && cEnd <= binY2)) {
                actuallyNeededContigs.add(contig);
            }
        }

        for (Contig2D contig1 : actuallyNeededContigs) {
            for (Contig2D contig2 : actuallyNeededContigs) {
                int cStart1 = contig1.getStart1() / zoom.getBinSize() / blockBinCount;
                int cEnd1 = contig1.getEnd1() / zoom.getBinSize() / blockBinCount;
                int cStart2 = contig2.getStart1() / zoom.getBinSize() / blockBinCount;
                int cEnd2 = contig2.getEnd1() / zoom.getBinSize() / blockBinCount;

                for (int r = cStart1; r <= cEnd1; r++) {
                    for (int c = cStart2; c <= cEnd2; c++) {
                        mzd.populateBlocksToLoad(r, c, no, blockList, blocksToLoad);
                    }
                }
            }
        }

        return actuallyNeededContigs;
    }

    /**
     *
     * @param preMergeContigs
     * @param blockList
     * @param binSize
     * @return
     */
    public static List<Block> filterBlockList(List<Contig2D> preMergeContigs, Set<Block> blockList, int binSize) {
        List<Contig2D> contigs = mergeRedundantContiguousContigs(preMergeContigs);

        List<Block> alteredBlockList = new ArrayList<>();
        if (contigs.size() < 1) return alteredBlockList;

        for (Block block : blockList) {
            List<ContactRecord> alteredContacts = new ArrayList<>();
            for (ContactRecord record : block.getContactRecords()) {
                boolean includeXRecord = false;
                boolean includeYRecord = false;
                int aX = -1, aY = -1;

                int genomeX = record.getBinX() * binSize;
                int genomeY = record.getBinY() * binSize;

                for (Contig2D contig : contigs) {
                    //System.out.println("contig "+contig);

                    if (contig.hasSomeOriginalOverlapWith(genomeX)) {
                        includeXRecord = true;
                        aX = contig.getAlteredBinIndex(record.getBinX(), binSize);
                        //System.out.println("axed "+record.getBinX()+" "+binSize+" "+aX);
                    }

                    if (contig.hasSomeOriginalOverlapWith(genomeY)) {
                        includeYRecord = true;
                        aY = contig.getAlteredBinIndex(record.getBinY(), binSize);
                        //System.out.println("ayed "+record.getBinY()+" "+binSize+" "+aY);
                    }

                    if (includeXRecord && includeYRecord) {
                        //System.out.println("altered ax and ay ");
                        if (aX > aY) {
                            alteredContacts.add(new ContactRecord(aY, aX, record.getCounts()));
                        } else {
                            alteredContacts.add(new ContactRecord(aX, aY, record.getCounts()));
                        }
                        break;
                    }
                }
            }
            alteredBlockList.add(new Block(block.getNumber(), alteredContacts));
        }
        //System.out.println("num alters "+alteredBlockList.size());
        return alteredBlockList;
    }


    /**
     * if neighboring contigs are not inverted, share original continuity
     * and share current continuity, they can essentially be merged
     * this will reduce the number of contigs, and improve speed
     *
     * @param currentContigs
     * @return mergedContigs
     */
    private static List<Contig2D> mergeRedundantContiguousContigs(List<Contig2D> currentContigs) {

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
        return AssemblyIntermediateProcessor.superAdapter;
    }

    public static void setSuperAdapter(SuperAdapter superAdapter) {
        AssemblyIntermediateProcessor.superAdapter = superAdapter;
    }
}
