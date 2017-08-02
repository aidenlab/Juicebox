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
    private final List<Chromosome> cleanedChromosomes;
    private final Map<String, Chromosome> chromosomeMap = new HashMap<>();
    private final List<String> chrIndices = new ArrayList<>();
    private int[] chromosomeBoundaries;
    private Chromosome[] chromosomesArray;
    private Chromosome[] chromosomeArrayWithoutAllByAll;

    public ChromosomeHandler(List<Chromosome> chromosomes) {

        // set the global chromosome list
        long genomeLength = 0;
        for (Chromosome c : chromosomes) {
            if (c != null)
                genomeLength += c.getLength();
        }
        chromosomes.set(0, new Chromosome(0, Globals.CHR_ALL, (int) (genomeLength / 1000)));
        chromosomes.add(new Chromosome(chromosomes.size(), "Custom", (int) (genomeLength / 1000)));

        this.cleanedChromosomes = new ArrayList<>();

        for (Chromosome c : chromosomes) {
            String cleanName = cleanUpName(c.getName());
            Chromosome cleanChromosome = new Chromosome(c.getIndex(), cleanName, c.getLength());
            cleanedChromosomes.add(cleanChromosome);
        }

        initializeInternalVariables();
    }

    public static String cleanUpName(String name) {
        return name.trim().toLowerCase().replaceAll("chr", "").toUpperCase();
    }

    /**
     * Set intersection
     * http://stackoverflow.com/questions/7574311/efficiently-compute-intersection-of-two-sets-in-java
     *
     * @param collection1
     * @param collection2
     * @return intersection of set1 and set2
     */
    private static Set<Chromosome> getSetIntersection(Collection<Chromosome> collection1, Collection<Chromosome> collection2) {
        Set<Chromosome> set1 = new HashSet<>(collection1);
        Set<Chromosome> set2 = new HashSet<>(collection2);

        boolean set1IsLarger = set1.size() > set2.size();
        Set<Chromosome> cloneSet = new HashSet<>(set1IsLarger ? set2 : set1);
        cloneSet.retainAll(set1IsLarger ? set1 : set2);
        return cloneSet;
    }

    public static boolean isAllByAll(Chromosome chromosome) {
        return isAllByAll(chromosome.getName());
    }

    public static boolean isAllByAll(String name) {
        return cleanUpName(name).equalsIgnoreCase(Globals.CHR_ALL);
    }

    public static boolean isCustomChromosome(Chromosome chromosome) {
        return isCustomChromosome(chromosome.getName());
    }

    private static boolean isCustomChromosome(String name) {
        return cleanUpName(name).equalsIgnoreCase("custom");
    }

    private void initializeInternalVariables() {

        for (Chromosome c : cleanedChromosomes) {
            chromosomeMap.put(c.getName(), c);
            if (c.getName().equalsIgnoreCase("MT")) {
                chromosomeMap.put("M", c); // special case for mitochondria
            }
        }

        for (Chromosome chr : cleanedChromosomes) {
            chrIndices.add("" + chr.getIndex());
        }

        // for all-by-all view
        chromosomeBoundaries = new int[cleanedChromosomes.size() - 1];
        long bound = 0;
        for (int i = 1; i < cleanedChromosomes.size(); i++) {
            Chromosome c = cleanedChromosomes.get(i);
            bound += (c.getLength() / 1000);
            chromosomeBoundaries[i - 1] = (int) bound;
        }

        chromosomesArray = cleanedChromosomes.toArray(new Chromosome[cleanedChromosomes.size()]);

        // array without all by all
        chromosomeArrayWithoutAllByAll = new Chromosome[chromosomesArray.length - 1];
        System.arraycopy(chromosomesArray, 1, chromosomeArrayWithoutAllByAll, 0, chromosomesArray.length - 1);
    }

    public Chromosome getChr(String name) {
        return chromosomeMap.get(cleanUpName(name));
    }

    public List<String> getChrIndices() {
        return chrIndices;
    }

    public boolean containsChromosome(String name) {
        return chromosomeMap.containsKey(cleanUpName(name));
    }

    public int size() {
        return chromosomesArray.length;
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

    public ChromosomeHandler getIntersetionWith(ChromosomeHandler handler2) {
        return new ChromosomeHandler(new ArrayList<>(getSetIntersection(this.cleanedChromosomes, handler2.cleanedChromosomes)));
    }

    public Chromosome[] getChromosomeArrayWithoutAllByAll() {
        return chromosomeArrayWithoutAllByAll;
    }
}
