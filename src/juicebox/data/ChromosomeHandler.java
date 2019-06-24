/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2019 Broad Institute, Aiden Lab
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

import juicebox.data.anchor.MotifAnchor;
import juicebox.data.anchor.MotifAnchorParser;
import juicebox.data.anchor.MotifAnchorTools;
import juicebox.data.feature.FeatureFunction;
import juicebox.data.feature.GenomeWideList;
import juicebox.track.feature.Feature2DList;
import org.broad.igv.Globals;
import org.broad.igv.feature.Chromosome;

import java.io.File;
import java.util.*;

/**
 * Created by muhammadsaadshamim on 8/3/16.
 */
public class ChromosomeHandler {
    private final List<Chromosome> cleanedChromosomes = new ArrayList<>();
    private final Map<String, Chromosome> chromosomeMap = new HashMap<>();
    private final Map<Integer, GenomeWideList<MotifAnchor>> customChromosomeRegions = new HashMap<>();
    private int[] chromosomeBoundaries;
    private Chromosome[] chromosomesArray;
    private Chromosome[] chromosomeArrayWithoutAllByAll;
    private Chromosome[] chromosomeArrayAutosomesOnly;

    public ChromosomeHandler(List<Chromosome> chromosomes) {

        // set the global chromosome list
        long genomeLength = getTotalLengthOfAllChromosomes(chromosomes);
        chromosomes.set(0, new Chromosome(0, Globals.CHR_ALL, (int) (genomeLength / 1000)));

        initializeCleanedChromosomesList(chromosomes);
        initializeInternalVariables();
    }

    public static String cleanUpName(String name) {
        if (name.equals("assembly")) {
            return "assembly";
        }
        if (name.equals("pseudoassembly")) {
            return "pseudoassembly";
        }
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
        // TODO: Chromosome defines hashcode based on index + length, but this is incorrect since index can be arbitrary
        cloneSet.retainAll(set1IsLarger ? set1 : set2);
        return cloneSet;
    }

    public static boolean isAllByAll(Chromosome chromosome) {
        return isAllByAll(chromosome.getName());
    }

    public static boolean isAllByAll(String name) {
        return cleanUpName(name).equalsIgnoreCase(Globals.CHR_ALL);
    }

    public Chromosome generateAssemblyChromosome() {
//        long genomeLength = 0;
//        for (Chromosome c : chromosomes) {
//            if (c != null) genomeLength += c.getLength();
//        }
//        return genomeLength;
        //TODO: handle scaling
        int size = (int) getTotalLengthOfAllChromosomes(Arrays.asList(this.chromosomeArrayWithoutAllByAll));

        int newIndex = cleanedChromosomes.size();
        Chromosome newChr = new Chromosome(newIndex, "pseudoassembly", size);
        cleanedChromosomes.add(newChr);
        chromosomeMap.put(newChr.getName(), newChr);
        return newChr;
    }

    public Chromosome generateCustomChromosomeFromBED(File file, int minSize) {
        GenomeWideList<MotifAnchor> regionsInCustomChromosome =
                MotifAnchorParser.loadFromBEDFile(this, file.getAbsolutePath());

        MotifAnchorTools.mergeAndExpandSmallAnchors(regionsInCustomChromosome, minSize);

        String cleanedUpName = cleanUpName(file.getName());

        return addCustomChromosome(regionsInCustomChromosome, cleanedUpName);
    }

    public Chromosome addCustomChromosome(Feature2DList featureList, String chrName) {
        GenomeWideList<MotifAnchor> featureAnchors =
                MotifAnchorTools.extractAllAnchorsFromAllFeatures(featureList, this);
        String cleanedUpName = cleanUpName(chrName);
        return addCustomChromosome(featureAnchors, cleanedUpName);
    }

    private Chromosome addCustomChromosome(GenomeWideList<MotifAnchor> regionsInCustomChromosome, String cleanedUpName) {
        int size = getTotalLengthOfAllRegionsInBedFile(regionsInCustomChromosome);
        int newIndex = cleanedChromosomes.size();
        customChromosomeRegions.put(newIndex, regionsInCustomChromosome);
        Chromosome newChr = new Chromosome(newIndex, cleanedUpName, size);
        cleanedChromosomes.add(newChr);
        chromosomeMap.put(newChr.getName(), newChr);
        return newChr;
    }

    private void initializeCleanedChromosomesList(List<Chromosome> chromosomes) {
        cleanedChromosomes.clear();
        for (Chromosome c : chromosomes) {
            String cleanName = cleanUpName(c.getName());
            Chromosome cleanChromosome = new Chromosome(c.getIndex(), cleanName, c.getLength());
            cleanedChromosomes.add(cleanChromosome);
        }
    }

    private void initializeInternalVariables() {

        for (Chromosome c : cleanedChromosomes) {
            chromosomeMap.put(c.getName(), c);
            if (c.getName().equalsIgnoreCase("MT")) {
                chromosomeMap.put("M", c); // special case for mitochondria
            }
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


        // array without X and Y
        List<Chromosome> autosomes = new ArrayList<>();
        for (Chromosome chr : chromosomeArrayWithoutAllByAll) {
            if (chr.getName().toLowerCase().contains("x") || chr.getName().toLowerCase().contains("y")) continue;
            autosomes.add(chr);
        }

        chromosomeArrayAutosomesOnly = new Chromosome[autosomes.size()];
        for (int i = 0; i < autosomes.size(); i++) {
            chromosomeArrayAutosomesOnly[i] = autosomes.get(i);
        }
    }

    private long getTotalLengthOfAllChromosomes(List<Chromosome> chromosomes) {
        long genomeLength = 0;
        for (Chromosome c : chromosomes) {
            if (c != null) genomeLength += c.getLength();
        }
        return genomeLength;
    }

    private int getTotalLengthOfAllRegionsInBedFile(GenomeWideList<MotifAnchor> regionsInCustomChromosome) {
        final int[] customGenomeLength = new int[]{0};
        regionsInCustomChromosome.processLists(new FeatureFunction<MotifAnchor>() {
            @Override
            public void process(String chr, List<MotifAnchor> featureList) {
                for (MotifAnchor c : featureList) {
                    if (c != null) customGenomeLength[0] += c.getWidth();
                }
            }
        });
        return customGenomeLength[0];
    }

    public boolean isCustomChromosome(Chromosome chromosome) {
        return isCustomChromosome(chromosome.getIndex());
    }

    private boolean isCustomChromosome(int index) {
        return customChromosomeRegions.containsKey(index);
    }

    public Chromosome getChromosomeFromName(String name) {
        return chromosomeMap.get(cleanUpName(name));
    }

    public boolean doesNotContainChromosome(String name) {
        return !chromosomeMap.containsKey(cleanUpName(name));
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

    public Chromosome getChromosomeFromIndex(int indx) {
        return chromosomesArray[indx];
    }

    public ChromosomeHandler getIntersectionWith(ChromosomeHandler handler2) {
        Set<Chromosome> intersection = getSetIntersection(cleanedChromosomes, handler2.cleanedChromosomes);
        if (intersection.isEmpty()) {
            return null;
        }
        else return new ChromosomeHandler(new ArrayList<>(intersection));
    }

    public Chromosome[] getAutosomalChromosomesArray() {
        return chromosomeArrayAutosomesOnly;
    }

    public Chromosome[] getChromosomeArrayWithoutAllByAll() {
        return chromosomeArrayWithoutAllByAll;
    }

    public GenomeWideList<MotifAnchor> getListOfRegionsInCustomChromosome(Integer index) {
        return customChromosomeRegions.get(index);
    }

    public String getGenomeId() {
        List<String> chrom_sizes = Arrays.asList("hg19", "hg38", "b37", "hg18", "mm10", "mm9", "GRCm38","aedAeg1", "anasPlat1", "assembly", "bTaurus3", "calJac3", "canFam3", "capHir1", "dm3", "dMel", "EBV", "equCab2", "felCat8", "galGal4", "hg18",  "loxAfr3", "macMul1", "macMulBaylor", "oryCun2", "oryLat2", "panTro4", "Pf3D7", "ratNor5", "ratNor6", "sacCer3", "sCerS288c", "spretus", "susScr3", "TAIR10");


        for (String id:chrom_sizes)  {
            ChromosomeHandler handler = HiCFileTools.loadChromosomes(id);
            for (Chromosome chr:handler.cleanedChromosomes) {
                for (Chromosome chr2:this.cleanedChromosomes) {
                    if (!chr.getName().equalsIgnoreCase("ALL") &&
                            chr.getName().equals(chr2.getName()) &&
                            chr.getLength() == chr2.getLength()) {
                        return id;
                    }
                }
            }
            // this is more elegant but there's a problem with the Chromosome hashCode
            //ChromosomeHandler handler1 = this.getIntersectionWith(handler);
            //if (handler1 != null && handler1.size() > 1) {
            //    return id;
            //}
        }
        return null;
    }

    public Chromosome[] extractOddOrEvenAutosomes(boolean extractOdd) {
        List<Chromosome> subset = new ArrayList<>();
        for (Chromosome chromosome : chromosomeArrayAutosomesOnly) {
            if (extractOdd && chromosome.getIndex() % 2 == 1) {
                subset.add(chromosome);
            } else if (!extractOdd && chromosome.getIndex() % 2 == 0) {
                subset.add(chromosome);
            }
        }
        Chromosome[] subsetArray = new Chromosome[subset.size()];
        for (int i = 0; i < subset.size(); i++) {
            subsetArray[i] = subset.get(i);
        }
        return subsetArray;
    }
}
