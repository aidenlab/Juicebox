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

package juicebox.data;

import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;

import java.util.*;

public class DynamicMatrixZoomData extends MatrixZoomData {

    private final MatrixZoomData higherResZD;
    private final int scaleFactor;

    /**
     * Constructor, sets the grid axes.  Called when read from file.
     *
     * @param zoom
     */
    public DynamicMatrixZoomData(HiCZoom zoom, MatrixZoomData higherResZD) {
        super(higherResZD.chr1, higherResZD.chr2, zoom, higherResZD.blockBinCount, higherResZD.blockColumnCount, new int[0], new int[0], null);
        this.higherResZD = higherResZD;
        scaleFactor = zoom.getBinSize() / higherResZD.getBinSize();
    }

    @Override
    public List<Block> getNormalizedBlocksOverlapping(long binX1, long binY1, long binX2, long binY2,
                                                      final NormalizationType norm, boolean isImportant, boolean fillUnderDiagonal) {
        // for V8 will be ints
        int higherBinX1 = (int) (binX1 * scaleFactor);
        int higherBinY1 = (int) (binY1 * scaleFactor);
        int higherBinX2 = (int) (binX2 * scaleFactor);
        int higherBinY2 = (int) (binY2 * scaleFactor);
        List<Block> blocksFromHigherRes = higherResZD.getNormalizedBlocksOverlapping(higherBinX1, higherBinY1, higherBinX2, higherBinY2,
                norm, isImportant, fillUnderDiagonal);
        return createBlocksForLowerRes(blocksFromHigherRes, norm);
    }

    private List<Block> createBlocksForLowerRes(List<Block> highResBlocks, NormalizationType norm) {

        Map<Integer, Map<Integer, ContactRecord>> condensedRecords = new HashMap<>();

        for (Block b : highResBlocks) {
            for (ContactRecord record : b.getContactRecords()) {
                int binX = record.getBinX() / scaleFactor;
                int binY = record.getBinY() / scaleFactor;
                float counts = record.getCounts();
                if (!condensedRecords.containsKey(binX)) {
                    condensedRecords.put(binX, new HashMap<>());
                }

                if (condensedRecords.get(binX).containsKey(binY)) {
                    condensedRecords.get(binX).get(binY).incrementCount(counts);
                } else {
                    ContactRecord recordNew = new ContactRecord(binX, binY, counts);
                    condensedRecords.get(binX).put(binY, recordNew);
                }
            }
        }

        Set<Integer> blockNumbers = new HashSet<>();

        for (int bx : condensedRecords.keySet()) {
            int cx = bx / blockBinCount;
            Map<Integer, ContactRecord> yMap = condensedRecords.get(bx);
            for (int by : yMap.keySet()) {
                ContactRecord cr = yMap.get(by);
                int ry = by / blockBinCount;

                int blockNumber = ry * blockColumnCount + cx;
                blockNumbers.add(blockNumber);
                String key = getBlockKey(blockNumber, norm);

                DynamicBlock b;
                if (blockCache.containsKey(key)) {
                    b = (DynamicBlock) blockCache.get(key);
                    b.addContactRecord(cr);
                } else {
                    DynamicBlock block = new DynamicBlock(blockNumber, cr, key);
                    blockCache.put(key, block);
                }
            }
        }

        List<Block> blockList = new ArrayList<>();
        for (Integer num : blockNumbers) {
            blockList.add(blockCache.get(getBlockKey(num, norm)));
        }
        return blockList;
    }

    @Override
    public void printFullDescription() {
        System.out.println("Dynamic Resolution Chromosome: " + chr1.getName() + " - " + chr2.getName());
        System.out.println("unit: " + zoom.getUnit());
        System.out.println("binSize (bp): " + zoom.getBinSize());
    }

}
