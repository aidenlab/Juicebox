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

package juicebox.data;

import org.broad.igv.Globals;
import org.broad.igv.feature.Chromosome;

import java.util.*;

/**
 * Created by muhammadsaadshamim on 8/3/16.
 */
public class ChromosomeHandler {
    private final List<Chromosome> chromosomes;
    private Map<String, Chromosome> chromosomeMap = new HashMap<String, Chromosome>();
    private List<String> chrIndices = new ArrayList<String>();
    private int[] chromosomeBoundaries;
    private Chromosome[] chromosomesArray;

    public ChromosomeHandler(List<Chromosome> chromosomes) {

        // set the global chromosome list
        long genomeLength = 0;
        for (Chromosome c : chromosomes) {
            if (c != null)
                genomeLength += c.getLength();
        }
        chromosomes.set(0, new Chromosome(0, Globals.CHR_ALL, (int) (genomeLength / 1000)));

        for (Chromosome c : chromosomes) {
            chromosomeMap.put(c.getName().trim().toLowerCase().replaceAll("chr", ""), c);
            if (c.getName().equalsIgnoreCase("MT")) {
                chromosomeMap.put("m", c); // special case for mitochondria
            }
        }

        for (Chromosome chr : chromosomes) {
            chrIndices.add("" + chr.getIndex());
        }

        // for all-by-all view
        chromosomeBoundaries = new int[chromosomes.size() - 1];
        long bound = 0;
        for (int i = 1; i < chromosomes.size(); i++) {
            Chromosome c = chromosomes.get(i);
            bound += (c.getLength() / 1000);
            chromosomeBoundaries[i - 1] = (int) bound;
        }

        chromosomesArray = chromosomes.toArray(new Chromosome[chromosomes.size()]);
        this.chromosomes = new ArrayList<>(chromosomes);
    }

    private String cleanedChrName(String name) {
        return name.trim().toLowerCase().replaceAll("chr", "");
    }

    public Chromosome getChr(String name) {
        return chromosomeMap.get(cleanedChrName(name));
    }

    public List<String> getChrIndices() {
        return chrIndices;
    }

    public boolean containsChromosome(String name) {
        return chromosomeMap.containsKey(cleanedChrName(name));
    }

    public int size() {
        return chromosomesArray.length;
    }

    public Set<String> getChrNames() {
        return chromosomeMap.keySet();
    }

    public int[] getChromosomeBoundaries() {
        return chromosomeBoundaries;
    }

    public Chromosome[] getChromosomeArray() {
        return chromosomesArray;
    }

    public Chromosome get(int indx) {
        return chromosomesArray[indx];
    }
}
