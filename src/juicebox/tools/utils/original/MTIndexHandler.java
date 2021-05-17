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

package juicebox.tools.utils.original;

import juicebox.HiCGlobals;
import juicebox.data.ChromosomeHandler;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class MTIndexHandler {
    public static Map<Integer, List<Chunk>> readMndIndex(String mndIndexFile,
                                                         Map<Integer, String> chromosomePairIndexes) {
        FileInputStream is = null;
        Map<String, List<Chunk>> tempIndex = new HashMap<>();
        Map<Integer, List<Chunk>> mndIndex = new ConcurrentHashMap<>();
        try {
            is = new FileInputStream(mndIndexFile);
            BufferedReader reader = new BufferedReader(new InputStreamReader(is), HiCGlobals.bufferSize);
            String nextLine;
            while ((nextLine = reader.readLine()) != null) {
                String[] nextEntry = nextLine.split(",");
                if (nextEntry.length == 4) {
                    // todo should probably just check if tempIndex.contains(nextEntry[0])
                    if (tempIndex.get(nextEntry[0]) == null) {
                        tempIndex.put(nextEntry[0], new ArrayList<>());
                    }
                    Chunk indexEntry = new Chunk(nextEntry[2], nextEntry[3]);
                    tempIndex.get(nextEntry[0]).add(indexEntry);
                } else {
                    System.err.println("Improperly formatted merged nodups index: " + nextLine);
                    System.exit(70);
                }
            }
            if (tempIndex.isEmpty()) {
                System.err.println("Intermediate MNDIndex is empty or could not be read");
                System.exit(44);
            }
        } catch (Exception e) {
            System.err.println("Unable to read merged nodups index");
            System.exit(70);
        }

        for (Map.Entry<Integer, String> entry : chromosomePairIndexes.entrySet()) {
            String reverseName = entry.getValue().split("-")[1] + "-" + entry.getValue().split("-")[0];
            if (tempIndex.containsKey(entry.getValue())) {
                mndIndex.put(entry.getKey(), tempIndex.get(entry.getValue()));
            } else if (tempIndex.containsKey(reverseName)) {
                mndIndex.put(entry.getKey(), tempIndex.get(reverseName));
            } else if (!reverseName.equalsIgnoreCase("all")) {
                System.err.println("Unable to find " + entry.getValue() + "  or  " + reverseName);
            }
        }

        if (mndIndex.isEmpty()) {
            System.err.println("MNDIndex is empty or could not be read");
            System.exit(43);
        }

        return mndIndex;
    }

    public static Map<String, Integer> populateChromosomeIndexes(ChromosomeHandler chromosomeHandler, int numCPUThreads) {
        Map<String, Integer> chromosomeIndexes = new ConcurrentHashMap<>(chromosomeHandler.size(), 0.75f, numCPUThreads);
        for (int i = 0; i < chromosomeHandler.size(); i++) {
            chromosomeIndexes.put(chromosomeHandler.getChromosomeFromIndex(i).getName(), i);
        }
        return chromosomeIndexes;
    }

    public static int populateChromosomePairIndexes(ChromosomeHandler chromosomeHandler,
                                                    Map<Integer, String> chromosomePairIndexes,
                                                    Map<String, Integer> chromosomePairIndexesReverse,
                                                    Map<Integer, Integer> chromosomePairIndex1,
                                                    Map<Integer, Integer> chromosomePairIndex2) {
        int chromosomePairCounter = 0;
        String genomeWideName = chromosomeHandler.getChromosomeFromIndex(0).getName();
        String genomeWidePairName = genomeWideName + "-" + genomeWideName;
        chromosomePairIndexes.put(chromosomePairCounter, genomeWidePairName);
        chromosomePairIndexesReverse.put(genomeWidePairName, chromosomePairCounter);
        chromosomePairIndex1.put(chromosomePairCounter, 0);
        chromosomePairIndex2.put(chromosomePairCounter, 0);
        chromosomePairCounter++;
        for (int i = 1; i < chromosomeHandler.size(); i++) {
            for (int j = i; j < chromosomeHandler.size(); j++) {
                String c1Name = chromosomeHandler.getChromosomeFromIndex(i).getName();
                String c2Name = chromosomeHandler.getChromosomeFromIndex(j).getName();
                String chromosomePairName = c1Name + "-" + c2Name;
                chromosomePairIndexes.put(chromosomePairCounter, chromosomePairName);
                chromosomePairIndexesReverse.put(chromosomePairName, chromosomePairCounter);
                chromosomePairIndex1.put(chromosomePairCounter, i);
                chromosomePairIndex2.put(chromosomePairCounter, j);
                chromosomePairCounter++;
            }
        }
        return chromosomePairCounter;
    }
}
