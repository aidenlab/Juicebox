/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab
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
import org.broad.igv.util.Pair;

import java.io.File;
import java.util.*;

/**
 * Created by muhammadsaadshamim on 8/3/16.
 */
public class ChromosomeHandler {
    private static final String GENOMEWIDE_CHR = "GENOMEWIDE";
    public final static int CUSTOM_CHROMOSOME_BUFFER = 5000;
    private final Map<String, Chromosome> chromosomeMap = new HashMap<>();
    private final Map<Integer, GenomeWideList<MotifAnchor>> customChromosomeRegions = new HashMap<>();
    private final List<Chromosome> cleanedChromosomes;
    private final String genomeID;
    private final int[] chromosomeBoundaries;
    private final Chromosome[] chromosomesArray;
    private final Chromosome[] chromosomeArrayWithoutAllByAll;
    private final Chromosome[] chromosomeArrayAutosomesOnly;

    public ChromosomeHandler(List<Chromosome> chromosomes, String genomeID, boolean inferID) {
        this(chromosomes, genomeID, inferID, true);
    }

    public ChromosomeHandler(List<Chromosome> chromosomes, String genomeID, boolean inferID, boolean createAllChr) {

        if (inferID) {
            String inferGenomeId = inferGenomeId();
            // if cannot find matching genomeID, set based on file
            if (inferGenomeId != null) {
                this.genomeID = inferGenomeId;
            } else {
                this.genomeID = genomeID;
            }
        } else {
            this.genomeID = genomeID;
        }


        // set the global chromosome list
        if (createAllChr) {
            long genomeLength = getTotalLengthOfAllChromosomes(chromosomes);
            chromosomes.set(0, new Chromosome(0, cleanUpName(Globals.CHR_ALL), (int) (genomeLength / 1000)));
        }

        cleanedChromosomes = initializeCleanedChromosomesList(chromosomes);
        Pair<int[], List<Chromosome[]>> outputs = initializeInternalVariables();
        chromosomeBoundaries = outputs.getFirst();
        chromosomesArray = outputs.getSecond().get(0);
        chromosomeArrayWithoutAllByAll = outputs.getSecond().get(1);
        chromosomeArrayAutosomesOnly = outputs.getSecond().get(2);
    }

    public static boolean isAllByAll(String name) {
        return name.contains("All") || name.contains("ALL") || name.contains("all");
    }

    public static void sort(List<Chromosome> indices) {
        Collections.sort(indices, new ChromosomeComparator());
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

    public String cleanUpName(String name) {
        if (name.equalsIgnoreCase("assembly")) {
            return "assembly";
        }
        if (name.equalsIgnoreCase("pseudoassembly")) {
            return "pseudoassembly";
        }
        if (genomeID.equalsIgnoreCase("hg19") || genomeID.equalsIgnoreCase("hg38")) {
            return name.trim().toLowerCase().replaceAll("chr", "").toUpperCase();
        }
        return name;
    }

    private GenomeWideList<MotifAnchor> generateChromDotSizesBedFile() {
        GenomeWideList<MotifAnchor> chromDotSizes = new GenomeWideList<>(this);

        for (Chromosome c : getChromosomeArray()) {
            if (isAllByAll(c) || isGenomeWide(c)) continue;
            MotifAnchor chromAnchor = new MotifAnchor(c.getName(), 0, c.getLength(), c.getName());
            List<MotifAnchor> anchors = new ArrayList<>();
            anchors.add(chromAnchor);
            chromDotSizes.setFeatures("" + c.getIndex(), anchors);
        }

        return chromDotSizes;
    }

    private boolean isGenomeWide(Chromosome chromosome) {
        return isGenomeWide(chromosome.getName());
    }

    private boolean isGenomeWide(String name) {
        return cleanUpName(name).equalsIgnoreCase(GENOMEWIDE_CHR);
    }

    public Chromosome addGenomeWideChromosome() {
        GenomeWideList<MotifAnchor> chromDotSizes = generateChromDotSizesBedFile();
        return addCustomChromosome(chromDotSizes, cleanUpName(GENOMEWIDE_CHR));
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

    private int getTotalLengthOfAllRegionsInBedFile(GenomeWideList<MotifAnchor> regionsInCustomChromosome) {
        final int[] customGenomeLength = new int[]{0};
        regionsInCustomChromosome.processLists(new FeatureFunction<MotifAnchor>() {
            @Override
            public void process(String chr, List<MotifAnchor> featureList) {
                for (MotifAnchor c : featureList) {
                    if (c != null) customGenomeLength[0] += c.getWidth() + CUSTOM_CHROMOSOME_BUFFER;
                }
            }
        });
        return customGenomeLength[0];
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

    private List<Chromosome> initializeCleanedChromosomesList(List<Chromosome> chromosomes) {
        List<Chromosome> cleanedChromosomes = new ArrayList<>();
        for (Chromosome c : chromosomes) {
            String cleanName = cleanUpName(c.getName());
            Chromosome cleanChromosome = new Chromosome(c.getIndex(), cleanName, c.getLength());
            cleanedChromosomes.add(cleanChromosome);
        }
        return cleanedChromosomes;
    }

    private Pair<int[], List<Chromosome[]>> initializeInternalVariables() {

        for (Chromosome c : cleanedChromosomes) {
            chromosomeMap.put(c.getName(), c);
            if (c.getName().equalsIgnoreCase("MT")) {
                chromosomeMap.put("M", c); // special case for mitochondria
            }
        }

        // for all-by-all view
        int[] chromosomeBoundaries = new int[cleanedChromosomes.size() - 1];
        long bound = 0;
        for (int i = 1; i < cleanedChromosomes.size(); i++) {
            Chromosome c = cleanedChromosomes.get(i);
            bound += (c.getLength() / 1000);
            chromosomeBoundaries[i - 1] = (int) bound;
        }

        Chromosome[] chromosomesArray = cleanedChromosomes.toArray(new Chromosome[cleanedChromosomes.size()]);

        // array without all by all
        Chromosome[] chromosomeArrayWithoutAllByAll = new Chromosome[chromosomesArray.length - 1];
        System.arraycopy(chromosomesArray, 1, chromosomeArrayWithoutAllByAll, 0, chromosomesArray.length - 1);


        // array without X and Y
        List<Chromosome> autosomes = new ArrayList<>();
        for (Chromosome chr : chromosomeArrayWithoutAllByAll) {
            if (chr.getName().toLowerCase().contains("x") || chr.getName().toLowerCase().contains("y") || chr.getName().toLowerCase().contains("m"))
                continue;
            autosomes.add(chr);
        }

        Chromosome[] chromosomeArrayAutosomesOnly = new Chromosome[autosomes.size()];
        for (int i = 0; i < autosomes.size(); i++) {
            chromosomeArrayAutosomesOnly[i] = autosomes.get(i);
        }

        List<Chromosome[]> outputs = new ArrayList<>();
        outputs.add(chromosomesArray);
        outputs.add(chromosomeArrayWithoutAllByAll);
        outputs.add(chromosomeArrayAutosomesOnly);


        return new Pair<>(chromosomeBoundaries, outputs);
    }

    private long getTotalLengthOfAllChromosomes(List<Chromosome> chromosomes) {
        long genomeLength = 0;
        for (Chromosome c : chromosomes) {
            if (c != null) genomeLength += c.getLength();
        }
        return genomeLength;
    }

    public String getGenomeID() {
        return genomeID;
    }

    static class ChromosomeComparator implements Comparator<Chromosome> {
        @Override
        public int compare(Chromosome a, Chromosome b) {
            Integer aIndx = a.getIndex();
            Integer bIndx = b.getIndex();
            return aIndx.compareTo(bIndx);
        }
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

        List<Chromosome> newSetOfChrs = new ArrayList<>();
        long genomeLength = getTotalLengthOfAllChromosomes(cleanedChromosomes);
        newSetOfChrs.add(new Chromosome(0, Globals.CHR_ALL, (int) (genomeLength / 1000)));
        for (Chromosome chromosome : cleanedChromosomes) {
            if (!isAllByAll(chromosome) && intersection.contains(chromosome)) {
                newSetOfChrs.add(chromosome);
            }
        }

        return new ChromosomeHandler(newSetOfChrs, genomeID, false);
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

    public String inferGenomeId() {
        List<String> chrom_sizes = Arrays.asList("hg19", "hg38", "b37", "hg18", "mm10", "mm9", "GRCm38", "aedAeg1",
                "anasPlat1", "assembly", "bTaurus3", "calJac3", "canFam3", "capHir1", "dm3", "dMel", "EBV", "equCab2",
                "felCat8", "galGal4", "hg18", "loxAfr3", "macMul1", "macMulBaylor", "oryCun2", "oryLat2", "panTro4",
                "Pf3D7", "ratNor5", "ratNor6", "sacCer3", "sCerS288c", "spretus", "susScr3", "TAIR10");

        for (String id : chrom_sizes) {
            ChromosomeHandler handler = HiCFileTools.loadChromosomes(id);
            for (Chromosome chr : handler.cleanedChromosomes) {
                for (Chromosome chr2 : this.cleanedChromosomes) {
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

    public Pair<Chromosome[], Chromosome[]> splitAutosomesIntoHalves() {

        int n = chromosomeArrayAutosomesOnly.length;
        int autosomesLength = 0;
        for (Chromosome chrom : chromosomeArrayAutosomesOnly) {
            autosomesLength += chrom.getLength();
        }
        int halfLength = autosomesLength / 2;

        // default assume chromosomes ordered with biggest first
        // so for human, assuming first 8 chroms
        int firstBatchUpToChr = n / 3 + 1;
        int prevLength = 0;

        for (int i = 0; i < n / 2; i++) {
            int newLength = prevLength + chromosomeArrayAutosomesOnly[i].getLength();
            if (prevLength <= halfLength && newLength >= halfLength) {
                // midpoint found
                if (Math.abs(prevLength - halfLength) < Math.abs(newLength - halfLength)) {
                    firstBatchUpToChr = i - 1;
                } else {
                    firstBatchUpToChr = i;
                }
                break;
            }
            prevLength = newLength;
        }
        System.out.println("Splitting chromosomes; " +
                chromosomeArrayAutosomesOnly[0].getName() + " to " + chromosomeArrayAutosomesOnly[firstBatchUpToChr].getName() + " and " +
                chromosomeArrayAutosomesOnly[firstBatchUpToChr + 1].getName() + " to " + chromosomeArrayAutosomesOnly[n - 1].getName());

        Chromosome[] rowsChromosomes = new Chromosome[firstBatchUpToChr];
        Chromosome[] colsChromosomes = new Chromosome[n - firstBatchUpToChr];
        for (int i = 0; i < n; i++) {
            if (i < firstBatchUpToChr) {
                rowsChromosomes[i] = chromosomeArrayAutosomesOnly[i];
            } else {
                colsChromosomes[i - firstBatchUpToChr] = chromosomeArrayAutosomesOnly[i];
            }
        }
        return new Pair<>(rowsChromosomes, colsChromosomes);
    }


    public Pair<Chromosome[], Chromosome[]> splitAutosomesAndSkipByTwos() {
        int n = chromosomeArrayAutosomesOnly.length;

        List<Chromosome> part1 = new ArrayList<>();
        List<Chromosome> part2 = new ArrayList<>();

        part1.add(chromosomeArrayAutosomesOnly[0]);
        int i = 1;
        int counterOffset = 0;
        boolean addToFirstOne = false;

        while (i < n) {

            if (addToFirstOne) {
                part1.add(chromosomeArrayAutosomesOnly[i]);
            } else {
                part2.add(chromosomeArrayAutosomesOnly[i]);
            }

            counterOffset++;
            i++;

            if (counterOffset == 2) {
                addToFirstOne = !addToFirstOne;
                counterOffset = 0;
            }
        }

        return new Pair<>(chromosomeListToArray(part1), chromosomeListToArray(part2));
    }

    private Chromosome[] chromosomeListToArray(List<Chromosome> chromosomes) {
        Chromosome[] array = new Chromosome[chromosomes.size()];
        for (int i = 0; i < chromosomes.size(); i++) {
            array[i] = chromosomes.get(i);
        }
        return array;
    }

}
