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

package juicebox.mapcolorui;

import juicebox.HiCGlobals;
import juicebox.data.Block;
import juicebox.data.ContactRecord;
import juicebox.data.MatrixZoomData;
import juicebox.gui.SuperAdapter;
import juicebox.track.feature.Contig2D;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import org.broad.igv.feature.Chromosome;

import java.util.*;

/**
 * Created by muhammadsaadshamim on 4/17/17.
 */
public class AssemblyIntermediateProcessor {

    private static SuperAdapter superAdapter;

    public static void makeChanges(String[] encodedInstructions, SuperAdapter superAdapter) {

        AssemblyIntermediateProcessor.superAdapter = superAdapter;
        List<Feature2DList> allFeatureLists = superAdapter.getContigLayer().getAnnotationLayer().getFeatureHandler()
                .getAllVisibleLoopLists();
        Feature2DList features = allFeatureLists.get(0);
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
                // TODO future
                // invert selections rather than just one contig
                // this involves inverting each of the sub contigs,
                // but also inverting their order

                invertEntryAt(contigs, Math.abs(Integer.parseInt(instruction)));
            } else {
                String[] indices = instruction.split("->");
                int currentIndex = Integer.parseInt(indices[0]);
                int newIndex = Integer.parseInt(indices[1]);
                moveFeatureToNewIndex(contigs, currentIndex, newIndex);
            }
        }

        recalculateAllAlterations(contigs);
    }

    private static void recalculateAllAlterations(List<Feature2D> contigs) {
        int i = 0;
        for (Feature2D feature2D : contigs) {
            Contig2D contig2D = feature2D.toContig();
            i = contig2D.setNewStart(i);
        }
    }

    private static void moveFeatureToNewIndex(List<Feature2D> contigs, int currentIndex, int newIndex) {
        // http://stackoverflow.com/questions/4938626/moving-items-around-in-an-arraylist
        Feature2D item = contigs.remove(currentIndex);
        contigs.add(newIndex, item);
    }

    private static void invertEntryAt(List<Feature2D> contigs, int index) {
        ((Contig2D) contigs.get(index)).toggleInversion();
    }

    public static List<Contig2D> retrieveRelevantBlocks(MatrixZoomData mzd, List<Integer> blocksToLoad,
                                                        List<Block> blockList, Chromosome chr1, Chromosome chr2,
                                                        int binX1, int binY1, int binX2, int binY2, int blockBinCount,
                                                        HiCZoom zoom, NormalizationType no) {

        //System.out.println("x "+binX1+" "+binX2+" y "+binY1+" "+binY2);

        System.out.println(superAdapter);
        System.out.println(superAdapter.getContigLayer());
        System.out.println(superAdapter.getContigLayer().getAnnotationLayer());
        System.out.println(superAdapter.getContigLayer().getAnnotationLayer().getFeatureHandler());
        Feature2DHandler handler = superAdapter.getContigLayer().getAnnotationLayer().getFeatureHandler();
        List<Feature2DList> allFeatureLists = handler.getAllVisibleLoopLists();
        net.sf.jsi.Rectangle currentWindow = new net.sf.jsi.Rectangle(binX1 * zoom.getBinSize(),
                binY1 * zoom.getBinSize(), binX2 * zoom.getBinSize(), binY2 * zoom.getBinSize());
        handler.getContainedFeatures(chr1.getIndex(), chr2.getIndex(), currentWindow);
        Feature2DList features = allFeatureLists.get(0);
        final String keyF = Feature2DList.getKey(chr1, chr2);
        List<Contig2D> contigs = new ArrayList<>();
        for (Feature2D entry : features.get(keyF)) {
            contigs.add(entry.toContig());
        }
        Collections.sort(contigs);
        //System.out.println("origContigs1 - "+contigs.size());

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
}
