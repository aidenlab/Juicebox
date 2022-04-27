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

import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.data.anchor.GenericLocus;
import juicebox.data.anchor.MotifAnchor;
import juicebox.data.basics.Chromosome;
import juicebox.data.feature.FeatureFunction;
import juicebox.data.feature.GenomeWideList;
import juicebox.windowui.HiCZoom;
import org.broad.igv.util.Pair;

import java.util.*;

/**
 * @author jrobinso
 * @since Aug 12, 2010
 */
public class Matrix {

    private final int chr1;
    private final int chr2;
    private final static Set<Pair<Integer, Integer>> dynamicZoomResolutions = new HashSet<>();
    protected List<MatrixZoomData> bpZoomData = new ArrayList<>();
    protected List<MatrixZoomData> fragZoomData = new ArrayList<>();
    protected List<MatrixZoomData> dynamicBPZoomData = new ArrayList<>();
    protected List<Integer> bpBinSizes = new ArrayList<>();
    protected List<Integer> fragBinSizes = new ArrayList<>();
    private final Comparator<MatrixZoomData> comparator = new Comparator<MatrixZoomData>() {
        @Override
        public int compare(MatrixZoomData o1, MatrixZoomData o2) {
            return o2.getBinSize() - o1.getBinSize();
        }
    };

    /**
     * Constructor for creating a matrix from precomputed data.
     *
     * @param chr1
     * @param chr2
     * @param zoomDataList
     */
    public Matrix(int chr1, int chr2, List<MatrixZoomData> zoomDataList) {
        this.chr1 = chr1;
        this.chr2 = chr2;
        initZoomDataMap(zoomDataList);
    }

    public static Matrix createCustomChromosomeMatrix(Chromosome chr1, Chromosome chr2, ChromosomeHandler handler,
                                                      final Map<String, Matrix> matrices, DatasetReader reader) {
        // TODO some weird null error when X chr in bed file?
        List<Chromosome> indicesForChr1 = getIndicesFromSubChromosomes(handler, chr1);
        List<Chromosome> indicesForChr2;
        if (chr1.getIndex() == chr2.getIndex()) {
            indicesForChr2 = new ArrayList<>(indicesForChr1);
        } else {
            indicesForChr2 = getIndicesFromSubChromosomes(handler, chr2);
        }

        if (HiCGlobals.printVerboseComments) {
            System.out.println("Indices_1 " + indicesForChr1);
            System.out.println("Indices_2 " + indicesForChr2);
        }

        // TODO need to sort first!!
        Chromosome newChr1 = chr1, newChr2 = chr2;
        if (chr1.getIndex() != chr2.getIndex() && indicesForChr1.get(0).getIndex() > indicesForChr2.get(0).getIndex()) {
            newChr1 = chr2;
            newChr2 = chr1;
        }

        Map<HiCZoom, CustomMatrixZoomData> customZDs = new HashMap<>();

        // ensure all regions loaded
        for (Chromosome i : indicesForChr1) {
            for (Chromosome j : indicesForChr2) {

                //System.out.println("from mtrx");
                String key = Matrix.generateKey(i, j);
                try {
                    Matrix m = matrices.get(key);
                    if (m == null) {
                        // TODO sometimes this fails once or twice, but later succeeds -
                        // TODO high priority, needs to be fixed
                        int numAttempts = 0;
                        while (m == null && numAttempts < 3) {
                            numAttempts++;
                            try {
                                m = reader.readMatrix(key);
                            } catch (Exception ignored) {
                            }
                        }
                        if (m == null) {
                            if (HiCGlobals.printVerboseComments) {
                                System.out.println("nothing found for cc4 " + i.getName() + " - " + j.getName());
                            }
                            continue;
                        }
                        matrices.put(key, m);
                    }
                    for (MatrixZoomData zd : m.bpZoomData) {
                        updateCustomZoomDataRegions(newChr1, newChr2, handler, key, zd, customZDs, reader);
                    }
                    for (MatrixZoomData zd : m.dynamicBPZoomData) {
                        updateCustomZoomDataRegions(newChr1, newChr2, handler, key, zd, customZDs, reader);
                    }
                    for (MatrixZoomData zd : m.fragZoomData) {
                        updateCustomZoomDataRegions(newChr1, newChr2, handler, key, zd, customZDs, reader);
                    }
                } catch (Exception ee) {
                    System.err.println("Custom Chr Region Missing " + key);
                    //ee.printStackTrace();
                }
                if (HiCGlobals.printVerboseComments)
                    System.out.println("completed cc4 " + i.getName() + " - " + j.getName());
            }
        }
        return new Matrix(chr1.getIndex(), chr2.getIndex(), new ArrayList<>(customZDs.values()));
    }

    private void initZoomDataMap(List<MatrixZoomData> zoomDataList) {
        for (MatrixZoomData zd : zoomDataList) {
            if (zd.getZoom().getUnit() == HiC.Unit.BP) {
                bpZoomData.add(zd);
            } else {
                fragZoomData.add(zd);
            }
    
            // Zooms should be sorted, but in case they are not...
    
            bpZoomData.sort(comparator);
            fragZoomData.sort(comparator);
        }

        for (Pair<Integer, Integer> resPair : dynamicZoomResolutions) {
            try {
                createDynamicResolutionMZD(resPair, false);
            } catch (Exception e) {
                System.err.println("Dynamic resolution could not be made");
            }
        }
        dynamicBPZoomData.sort(comparator);

    }

    public static String generateKey(int chr1, int chr2) {
        if (chr2 < chr1) return "" + chr2 + "_" + chr1;
        return "" + chr1 + "_" + chr2;
    }

    public static Matrix createAssemblyChromosomeMatrix(ChromosomeHandler handler,
                                                        final Map<String, Matrix> matrices, DatasetReader reader) {
        Map<HiCZoom, MatrixZoomData> assemblyZDs = new HashMap<>();

        Matrix matrix = null;
        int numAttempts = 0;
        while (matrix == null && numAttempts < 3) {
            try {
                matrix = reader.readMatrix("1_1");
            } catch (Exception ignored) {
                numAttempts++;
            }
        }

        long length = handler.getChromosomeFromName("pseudoassembly").getLength(); // TODO: scaling; also maybe chromosome ends need to shift to start with new bin at every zoom?
        for (MatrixZoomData zd : matrix.bpZoomData) {
            // todo @dudcha is this done for resolutions where conversion will be lossy?
            assemblyZDs.put(zd.getZoom(), new MatrixZoomData(handler.getChromosomeFromName("pseudoassembly"), handler.getChromosomeFromName("pseudoassembly"), zd.getZoom(),
                    (int) (length / zd.getBinSize()), (int) (length / zd.getBinSize()), null, null, reader));
        }


        //TODO: assumption that we are doing this before modifying the handler

//        for (Chromosome i : handler.getChromosomeArrayWithoutAllByAll()) {
//            for (Chromosome j : handler.getChromosomeArrayWithoutAllByAll()) {
//
//                //System.out.println("from mtrx");
//                String key = Matrix.generateKey(i, j);
//                try {
//                    Matrix m = matrices.get(key);
//                    if (m == null) {
//                        // TODO sometimes this fails once or twice, but later succeeds -
//                        // TODO high priority, needs to be fixed??????
//                        int numAttempts = 0;
//                        while (m == null && numAttempts < 3) {
//                            try {
//                                m = reader.readMatrix(key);
//                            } catch (Exception ignored) {
//                                numAttempts++;
//                            }
//                        }
//
//                        for(MatrixZoomData tempMatrixZoomData : m.bpZoomData){
//                            tempMatrixZoomData.
//                        }
//
//
//                        // modify m for each zoom
////                        matrices.put(key, m); //perhaps move it to the end
//                    }
//                    for (MatrixZoomData zd : m.bpZoomData) {
//                        updateCustomZoomDataRegions(newChr1, newChr2, handler, key, zd, assemblyZDs, reader);
//                    }
////                    for (MatrixZoomData zd : m.fragZoomData) {
////                        updateCustomZoomDataRegions(newChr1, newChr2, handler, key, zd, customZDs, reader);
////                    }
//                } catch (Exception ee) {
//                    System.err.println("Everything failed in creatingAssemblyChromosomeMatrix " + key);
//                    ee.printStackTrace();
//                }
//            }
//        }

        Matrix m = new Matrix(handler.size(), handler.size(), new ArrayList<>(assemblyZDs.values()));
        matrices.put(generateKey(handler.size(), handler.size()), m);
        return m;
    }

    public void createDynamicResolutionMZD(Pair<Integer, Integer> resPair, boolean addToSet) {
        int newRes = resPair.getFirst();
        int highRes = resPair.getSecond();

        MatrixZoomData highMZD = getZoomData(new HiCZoom(HiC.Unit.BP, highRes));
        MatrixZoomData newMZD = new DynamicMatrixZoomData(new HiCZoom(HiC.Unit.BP, newRes), highMZD);
        if (addToSet) {
            dynamicZoomResolutions.add(resPair);
        }
        dynamicBPZoomData.add(newMZD);
    }

    private static List<Chromosome> getIndicesFromSubChromosomes(final ChromosomeHandler handler, Chromosome chromosome) {
        final List<Chromosome> indices = new ArrayList<>();
        if (handler.isCustomChromosome(chromosome)) {
            GenomeWideList<GenericLocus> regions = handler.getListOfRegionsInCustomChromosome(chromosome.getIndex());
            regions.processLists(new FeatureFunction<GenericLocus>() {
                @Override
                public void process(String chr, List<GenericLocus> featureList) {
                    if (featureList.size() > 0) {
                        Chromosome chromosomeN = handler.getChromosomeFromName(chr);
                        if (chromosomeN != null) {
                            indices.add(chromosomeN);
                        }
                    }
                }
            });
        } else {
            indices.add(chromosome);
        }

        ChromosomeHandler.sort(indices);
        return indices;
    }

    private static void updateCustomZoomDataRegions(Chromosome chr1, Chromosome chr2, ChromosomeHandler handler,
                                                    String regionKey, MatrixZoomData zd,
                                                    Map<HiCZoom, CustomMatrixZoomData> customZDs, DatasetReader reader) {
        if (!customZDs.containsKey(zd.getZoom())) {
            customZDs.put(zd.getZoom(), new CustomMatrixZoomData(chr1, chr2, handler, zd.getZoom(), reader));
        }

        customZDs.get(zd.getZoom()).expandAvailableZoomDatas(regionKey, zd);
    }

    public static String generateKey(Chromosome chr1, Chromosome chr2) {
        int t1 = Math.min(chr1.getIndex(), chr2.getIndex());
        int t2 = Math.max(chr1.getIndex(), chr2.getIndex());
        return generateKey(t1, t2);
    }

    public String getKey() {
        return generateKey(chr1, chr2);
    }

    public MatrixZoomData getFirstZoomData() {
        if (bpZoomData != null && bpZoomData.size() > 0) {
            return getFirstZoomData(HiC.Unit.BP);
       } else {
            return getFirstZoomData(HiC.Unit.FRAG);
        }
    }

    public MatrixZoomData getFirstZoomData(HiC.Unit unit) {
        if (unit == HiC.Unit.BP) {
            return bpZoomData != null && bpZoomData.size() > 0 ? bpZoomData.get(0) : null;
        } else {
            return fragZoomData != null && fragZoomData.size() > 0 ? fragZoomData.get(0) : null;
        }
    }

    public MatrixZoomData getFirstPearsonZoomData(HiC.Unit unit) {
        if (unit == HiC.Unit.BP) {
            return bpZoomData != null ? bpZoomData.get(2) : null;
        } else {
            return fragZoomData != null ? fragZoomData.get(2) : null;
        }

    }

    public MatrixZoomData getZoomData(HiCZoom zoom) {
        int targetZoom = zoom.getBinSize();
        List<MatrixZoomData> zdList = (zoom.getUnit() == HiC.Unit.BP) ? bpZoomData : fragZoomData;
        //linear search for bin size, the lists are not large
        for (MatrixZoomData zd : zdList) {
            if (zd.getBinSize() == targetZoom) {
                return zd;
            }
        }

        for (MatrixZoomData zd : dynamicBPZoomData) {
            if (zd.getBinSize() == targetZoom) {
                return zd;
            }
        }

        // special exception for all by all
        if (chr1 == 0 && chr2 == 0) {

            MatrixZoomData closestValue = zdList.get(0);
            int distance = Math.abs(closestValue.getBinSize() - targetZoom);
            for (MatrixZoomData zd : zdList) {
                int cdistance = Math.abs(zd.getBinSize() - targetZoom);
                if (cdistance < distance) {
                    closestValue = zd;
                    distance = cdistance;
                }
            }

            return closestValue;
        }

        return null;
    }

    public MatrixZoomData getZoomData(int index) {
        if (index < bpBinSizes.size()) {
            HiCZoom zoom = new HiCZoom(HiC.Unit.BP, bpBinSizes.get(index));
            return getZoomData(zoom);
        } else if (index >= bpBinSizes.size() && index < (bpBinSizes.size()+fragBinSizes.size())) {
            HiCZoom zoom = new HiCZoom(HiC.Unit.FRAG, fragBinSizes.get(index));
            return getZoomData(zoom);
        } else {
            return null;
        }
    }

    public int getNumBPResolutions() {
        return bpBinSizes.size();
    }

    public int getNumFragResolutions() {
        return fragBinSizes.size();
    }

    public boolean isNotIntra() {
        return chr1 != chr2;
    }

    public int getChr1Idx() {
        return chr1;
    }

    public int getChr2Idx() {
        return chr2;
    }

    public void setBpBinSizes(ArrayList<Integer> bpBinSizes) {
        this.bpBinSizes = bpBinSizes;
    }

    public void setFragBinSizes(ArrayList<Integer> fragBinSizes) {
        this.fragBinSizes = fragBinSizes;
    }

}
