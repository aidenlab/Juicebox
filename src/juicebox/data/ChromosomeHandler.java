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

package juicebox.data;

import juicebox.Context;
import juicebox.data.anchor.*;
import juicebox.data.basics.Chromosome;
import juicebox.data.feature.FeatureFunction;
import juicebox.data.feature.GenomeWideList;
import juicebox.track.feature.Feature2DList;
import org.broad.igv.util.Pair;

import java.io.File;
import java.util.*;

/**
 * Created by muhammadsaadshamim on 8/3/16.
 */
public class ChromosomeHandler {
    private static final String GENOMEWIDE_CHR = "GENOMEWIDE";
    public final static int CUSTOM_CHROMOSOME_BUFFER = 10; // todo make to smallest value of resolution
    private static final String CHR_ALL = "All";
    private final Map<String, Chromosome> chromosomeMap = new HashMap<>();
    private final Map<Integer, GenomeWideList<GenericLocus>> customChromosomeRegions = new HashMap<>();
    private final List<Chromosome> cleanedChromosomes;
    private final String genomeID;
    private final boolean preserveExactNames;
    private final long[] chromosomeBoundaries;
    private javastraw.reader.v10.V10AllOverview allOverview;
    private final Chromosome[] chromosomesArray;
    private final Chromosome[] chromosomeArrayWithoutAllByAll;
    private final Chromosome[] chromosomeArrayAutosomesOnly;

    public ChromosomeHandler(List<Chromosome> chromosomes, String genomeID, boolean inferID) {
        this(chromosomes, genomeID, inferID, true);
    }

    public ChromosomeHandler(List<Chromosome> chromosomes, String genomeID, boolean inferID, boolean createAllChr) {
        this(chromosomes, genomeID, inferID, createAllChr, false);
    }

    public ChromosomeHandler(List<Chromosome> chromosomes, String genomeID, boolean inferID, boolean createAllChr, boolean preserveExactNames) {
        this.preserveExactNames = preserveExactNames;

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
            chromosomes.set(0, new Chromosome(0, cleanUpName(CHR_ALL), (int) (genomeLength / 1000)));
        }

        cleanedChromosomes = initializeCleanedChromosomesList(chromosomes);
        Pair<long[], List<Chromosome[]>> outputs = initializeInternalVariables();
        chromosomeBoundaries = outputs.getFirst();
        chromosomesArray = outputs.getSecond().get(0);
        chromosomeArrayWithoutAllByAll = outputs.getSecond().get(1);
        chromosomeArrayAutosomesOnly = outputs.getSecond().get(2);
    }

    public static boolean isWholeGenomeView(Context xContext, Context yContext) {
        return isAllByAll(xContext.getChromosome()) && isAllByAll(yContext.getChromosome());
    }


    public String cleanUpName(String name) {
        if (preserveExactNames) return name;
        if (genomeID.equalsIgnoreCase("hg19")) {
            return name.trim().toLowerCase().replaceAll("chr", "").toUpperCase();
        }
        return name;
    }

    public static boolean isAllByAll(String name) {
        return name != null && name.equalsIgnoreCase("all");
    }

    public static void sort(List<Chromosome> indices) {
        indices.sort(new ChromosomeComparator());
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
        return chromosome != null && isAllByAll(chromosome.getName());
    }


    private GenomeWideList<GenericLocus> generateChromDotSizesBedFile() {
        GenomeWideList<GenericLocus> chromDotSizes = new GenomeWideList<>(this);

        for (Chromosome c : getChromosomeArray()) {
            if (isAllByAll(c) || isGenomeWide(c)) continue;
            MotifAnchor chromAnchor = new MotifAnchor(c.getName(), 0, (int) c.getLength(), c.getName()); // not implemented or called
            List<GenericLocus> anchors = new ArrayList<>();
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
        GenomeWideList<GenericLocus> chromDotSizes = generateChromDotSizesBedFile();
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
        GenomeWideList<GenericLocus> regionsInCustomChromosome =
                GenericLocusParser.loadFromBEDFile(this, file.getAbsolutePath());

        MotifAnchorTools.mergeAndExpandSmallAnchors(regionsInCustomChromosome, minSize);

        String cleanedUpName = cleanUpName(file.getName());

        return addCustomChromosome(regionsInCustomChromosome, cleanedUpName);
    }

    public Chromosome addCustomChromosome(Feature2DList featureList, String chrName) {
        GenomeWideList<GenericLocus> featureAnchors =
                MotifAnchorTools.extractAllAnchorsFromAllFeatures(featureList, this);
        String cleanedUpName = cleanUpName(chrName);
        return addCustomChromosome(featureAnchors, cleanedUpName);
    }

    private int getTotalLengthOfAllRegionsInBedFile(GenomeWideList<GenericLocus> regionsInCustomChromosome) {
        final int[] customGenomeLength = new int[]{0};
        regionsInCustomChromosome.processLists(new FeatureFunction<GenericLocus>() {
            @Override
            public void process(String chr, List<GenericLocus> featureList) {
                for (GenericLocus c : featureList) {
                    if (c != null) customGenomeLength[0] += c.getWidth() + CUSTOM_CHROMOSOME_BUFFER;
                }
            }
        });
        return customGenomeLength[0];
    }

    private Chromosome addCustomChromosome(GenomeWideList<GenericLocus> regionsInCustomChromosome, String cleanedUpName) {
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

    private Pair<long[], List<Chromosome[]>> initializeInternalVariables() {

        for (Chromosome c : cleanedChromosomes) {
            chromosomeMap.put(c.getName(), c);
            String alias = c.getName().startsWith("chr") ? c.getName().substring(3) : "chr" + c.getName();
            if (!chromosomeMap.containsKey(alias)) chromosomeMap.put(alias, c);
            if (c.getName().equalsIgnoreCase("MT")) {
                chromosomeMap.put("M", c); // special case for mitochondria
            }
        }

        List<Chromosome> realChromosomes = new ArrayList<>();
        for (Chromosome chromosome : cleanedChromosomes) {
            if (!isAllByAll(chromosome)) realChromosomes.add(chromosome);
        }
        long[] chromosomeBoundaries = new long[realChromosomes.size()];
        long bound = 0;
        for (int i = 0; i < realChromosomes.size(); i++) {
            bound += realChromosomes.get(i).getLength() / 1000;
            chromosomeBoundaries[i] = bound;
        }
        Chromosome[] chromosomesArray = cleanedChromosomes.toArray(new Chromosome[0]);
        Chromosome[] chromosomeArrayWithoutAllByAll = realChromosomes.toArray(new Chromosome[0]);

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

    public boolean isCustomChromosome(Chromosome chromosome) {
        return isCustomChromosome(chromosome.getIndex());
    }

    private boolean isCustomChromosome(int index) {
        return customChromosomeRegions.containsKey(index);
    }

    public Chromosome getChromosomeFromName(String name) {
        if (name == null) return null;
        Chromosome exact = chromosomeMap.get(name);
        if (exact != null) return exact;
        if (isAllByAll(name)) for (Chromosome c : chromosomesArray)
            if (isAllByAll(c)) return c;
        return chromosomeMap.get(cleanUpName(name));
    }

    public boolean doesNotContainChromosome(String name) {
        return getChromosomeFromName(name) == null;
    }

    public int size() {
        return chromosomesArray.length;
    }

    public void setAllOverview(javastraw.reader.v10.V10AllOverview overview) { allOverview = overview; }
    public javastraw.reader.v10.V10AllOverview getAllOverview() { return allOverview; }
    public Chromosome getOverviewChromosome(long coordinate) { return getOverviewChromosome((double) coordinate); }
    public Chromosome getOverviewChromosome(double coordinate) {
        if (allOverview != null) return getChromosomeFromIndex(allOverview.chromosomeForCoordinate(coordinate));
        long[] boundaries = getChromosomeBoundaries();
        for (int i = 0; i < boundaries.length; i++)
            if (coordinate < boundaries[i]) return chromosomeArrayWithoutAllByAll[i];
        return chromosomeArrayWithoutAllByAll[chromosomeArrayWithoutAllByAll.length - 1];
    }
    public long getOverviewPosition(Chromosome chromosome, long coordinate) {
        return (long) getOverviewPositionBP(chromosome, coordinate);
    }
    public double getOverviewPositionBP(Chromosome chromosome, double coordinate) {
        if (allOverview != null) {
            for (int i = 0; i < allOverview.chromosomeIndices.length; i++) {
                if (allOverview.chromosomeIndices[i] == chromosome.getIndex()) {
                    double bp = allOverview.version == 2
                            ? coordinate * allOverview.coordinateScale - allOverview.sourceOffsets[i]
                            : (coordinate * allOverview.groupFactor / allOverview.finestResolution - allOverview.sourceOffsets[i]) * allOverview.sourceResolution;
                    return Math.max(0, Math.min(chromosome.getLength() - 1, bp));
                }
            }
        } else {
            long start = 0;
            for (Chromosome c : chromosomeArrayWithoutAllByAll) {
                if (c.getIndex() == chromosome.getIndex()) return Math.max(0, (coordinate - start) * 1000);
                start += c.getLength() / 1000;
            }
        }
        throw new IllegalArgumentException("Unknown overview chromosome");
    }
    /** Base pairs represented by one coordinate unit on this file's All axis. */
    public double getOverviewCoordinateScale() {
        if (allOverview == null) return 1000;
        return allOverview.version == 2 ? allOverview.coordinateScale :
                allOverview.sourceResolution * (double) allOverview.groupFactor / allOverview.finestResolution;
    }

    /** Inverse of getOverviewPosition, using the receiving file's own order and padding. */
    public double getOverviewCoordinate(Chromosome chromosome, double position) {
        if (chromosome == null || isAllByAll(chromosome)) throw new IllegalArgumentException("Unknown overview chromosome");
        double bp = Math.max(0, Math.min(chromosome.getLength(), position));
        if (allOverview != null) {
            for (int i = 0; i < allOverview.chromosomeIndices.length; i++) {
                if (allOverview.chromosomeIndices[i] == chromosome.getIndex()) {
                    return allOverview.version == 2 ? (allOverview.sourceOffsets[i] + bp) / allOverview.coordinateScale :
                        (allOverview.sourceOffsets[i] + bp / allOverview.sourceResolution) * allOverview.finestResolution / allOverview.groupFactor;
                }
            }
        } else {
            double offset = 0;
            for (Chromosome c : chromosomeArrayWithoutAllByAll) {
                if (c.getIndex() == chromosome.getIndex()) return offset + bp / 1000;
                offset += c.getLength() / 1000;
            }
        }
        throw new IllegalArgumentException("Unknown overview chromosome");
    }

    public boolean hasSameOverviewLayout(ChromosomeHandler other) {
        javastraw.reader.v10.V10AllOverview a = allOverview, b = other.allOverview;
        if (a == null || b == null) {
            if (a != b || chromosomeArrayWithoutAllByAll.length != other.chromosomeArrayWithoutAllByAll.length) return false;
            for (int i = 0; i < chromosomeArrayWithoutAllByAll.length; i++) {
                Chromosome x = chromosomeArrayWithoutAllByAll[i], y = other.chromosomeArrayWithoutAllByAll[i];
                if (!cleanUpName(x.getName()).equals(cleanUpName(y.getName())) || x.getLength() != y.getLength()) return false;
            }
            return true;
        }
        if (a.chromosomeIndices.length != b.chromosomeIndices.length) return false;
        if (a.version != b.version || a.effectiveResolution != b.effectiveResolution || a.binCount != b.binCount ||
                (a.version == 2 ? !java.util.Arrays.equals(a.chromosomeLengths, b.chromosomeLengths) :
                a.sourceResolution != b.sourceResolution || a.groupFactor != b.groupFactor ||
                !java.util.Arrays.equals(a.chromosomeBins, b.chromosomeBins))) return false;
        for (int i = 0; i < a.chromosomeIndices.length; i++)
            if (!getChromosomeFromIndex(a.chromosomeIndices[i]).getName().equals(
                other.getChromosomeFromIndex(b.chromosomeIndices[i]).getName())) return false;
        return true;
    }
    public long[] getChromosomeBoundaries() {
        if (allOverview == null) return chromosomeBoundaries;
        long[] boundaries = new long[allOverview.chromosomeIndices.length];
        for (int i = 0; i < boundaries.length; i++)
            boundaries[i] = Math.toIntExact((long) Math.ceil(allOverview.syntheticBoundary(i)));
        boundaries[boundaries.length - 1] = allOverview.syntheticLength;
        return boundaries;
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
        newSetOfChrs.add(new Chromosome(0, CHR_ALL, (int) (genomeLength / 1000)));
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

    public GenomeWideList<GenericLocus> getListOfRegionsInCustomChromosome(Integer index) {
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

    static class ChromosomeComparator implements Comparator<Chromosome> {
        @Override
        public int compare(Chromosome a, Chromosome b) {
            Integer aIndx = a.getIndex();
            Integer bIndx = b.getIndex();
            return aIndx.compareTo(bIndx);
        }
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
        long prevLength = 0;

        for (int i = 0; i < n / 2; i++) {
            long newLength = prevLength + chromosomeArrayAutosomesOnly[i].getLength();
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
