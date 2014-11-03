package juicebox.data;

import org.apache.log4j.Logger;
import org.broad.igv.feature.Chromosome;
import juicebox.HiC;
import juicebox.HiCZoom;
import juicebox.MainWindow;
import juicebox.NormalizationType;
import org.broad.igv.util.FileUtils;
import org.broad.igv.util.ResourceLocator;
import org.broad.igv.util.collections.LRUCache;

import java.io.*;
import java.util.*;

/**
 * @author jrobinso
 * @since Aug 12, 2010
 */
public class Dataset {

    private static Logger log = Logger.getLogger(Dataset.class);

   // private boolean caching = true;

    //Chromosome lookup table
    public List<Chromosome> chromosomes;

    Map<String, Matrix> matrices = new HashMap<String, Matrix>(25 * 25);

    private DatasetReader reader;

    Map<String, ExpectedValueFunction> expectedValueFunctionMap;

    String genomeId;
    String restrictionEnzyme = null;

    List<HiCZoom> bpZooms;
    List<HiCZoom> fragZooms;
    private Map<String, String> attributes;
    private Map<String, Integer> fragmentCounts;

    LRUCache<String, double[]> eigenvectorCache;
    LRUCache<String, NormalizationVector> normalizationVectorCache;
    Map<String, NormalizationVector> loadedNormalizationVectors;
    private List<NormalizationType> normalizationTypes;

    public Dataset(DatasetReader reader) {
        this.reader = reader;
        eigenvectorCache = new LRUCache<String, double[]>(20);
        normalizationVectorCache = new LRUCache<String, NormalizationVector>(20);
        normalizationTypes = new ArrayList<NormalizationType>();
    }


    public Matrix getMatrix(Chromosome chr1, Chromosome chr2) {

        // order is arbitrary, convention is lower # chr first
        int t1 = Math.min(chr1.getIndex(), chr2.getIndex());
        int t2 = Math.max(chr1.getIndex(), chr2.getIndex());

        String key = Matrix.generateKey(t1, t2);
        Matrix m = matrices.get(key);

        if (m == null && reader != null) {
            try {
                m = reader.readMatrix(key);
                matrices.put(key, m);
            } catch (IOException e) {
                log.error("Error fetching matrix for: " + chr1.getName() + "-" + chr2.getName(), e);
            }
        }

        return m;

    }

    public ResourceLocator getPeaks() {
        String path = reader.getPath().substring(0, reader.getPath().lastIndexOf('.'));
        if (path.lastIndexOf("_30") > -1) {
            path = path.substring(0, path.lastIndexOf("_30"));
        }

        String location = path + "_peaks.txt";

        if (FileUtils.resourceExists(location)) {
            return new ResourceLocator(location);
        }
        else {
            return null;
        }
    }

    public ResourceLocator getBlocks() {
        String path = reader.getPath().substring(0, reader.getPath().lastIndexOf('.'));
        if (path.lastIndexOf("_30") > -1) {
            path = path.substring(0, path.lastIndexOf("_30"));
        }

        String location = path + "_blocks.txt";

        if (FileUtils.resourceExists(location)) {
            return new ResourceLocator(location);
        }
        else {
            return null;
        }
    }

    public void setAttributes(Map<String, String> map) {
        this.attributes = map;
    }


    public List<NormalizationType> getNormalizationTypes() {
        return normalizationTypes;
    }


    public void addNormalizationType(NormalizationType type) {
        if (!normalizationTypes.contains(type)) normalizationTypes.add(type);
    }

    public void setNormalizationTypes(List<NormalizationType> normalizationTypes) {
        this.normalizationTypes = normalizationTypes;
    }

    public int getNumberZooms(HiC.Unit unit) {
        return unit == HiC.Unit.BP ? bpZooms.size() : fragZooms.size();
    }


    public HiCZoom getZoom(HiC.Unit unit, int index) {
        return unit == HiC.Unit.BP ? bpZooms.get(index) : fragZooms.get(index);
    }


    public ExpectedValueFunction getExpectedValues(HiCZoom zoom, NormalizationType type) {
        if (expectedValueFunctionMap == null) return null;
        String key = zoom.getKey() + "_" + type.toString(); // getUnit() + "_" + zoom.getBinSize();
        return expectedValueFunctionMap.get(key);
    }


    public void setExpectedValueFunctionMap(Map<String, ExpectedValueFunction> df) {
        this.expectedValueFunctionMap = df;
    }


    public Map<String, ExpectedValueFunction> getExpectedValueFunctionMap() {
        return expectedValueFunctionMap;
    }


    public List<Chromosome> getChromosomes() {
        return chromosomes;
    }


    public void setChromosomes(List<Chromosome> chromosomes) {
        this.chromosomes = chromosomes;
    }


    public int getVersion() {
        return reader.getVersion();
    }


    public void setGenomeId(String genomeId) {
        this.genomeId = genomeId;
    }


    public String getGenomeId() {
        return genomeId;
    }

    public void setRestrictionEnzyme(int nSites) {
        restrictionEnzyme = findRestrictionEnzyme(nSites);
    }

    public String getRestrictionEnzyme() {
        return restrictionEnzyme;
    }


    public void setBpZooms(int[] bpBinSizes) {

        // Limit resolution in restricted mode
        int n = bpBinSizes.length;
//        if (MainWindow.isRestricted()) {
//            for (int i = 0; i < bpBinSizes.length; i++) {
//                if (bpBinSizes[i] < 25000) {
//                    n = i;
//                    break;
//                }
//            }
//        }

        this.bpZooms = new ArrayList<HiCZoom>(n);
        for (int i = 0; i < n; i++) {
            bpZooms.add(new HiCZoom(HiC.Unit.BP, bpBinSizes[i]));
        }

    }


    public void setFragZooms(int[] fragBinSizes) {

        // Don't show fragments in restricted mode
//        if (MainWindow.isRestricted()) return;

        this.fragZooms = new ArrayList<HiCZoom>(fragBinSizes.length);
        for (int i = 0; i < fragBinSizes.length; i++) {
            fragZooms.add(new HiCZoom(HiC.Unit.FRAG, fragBinSizes[i]));
        }
    }


    public String getStatistics() {
        if (attributes == null) return null;
        else return attributes.get("statistics");
    }


    public String getGraphs() {
        if (attributes == null) return null;
        return attributes.get("graphs");
    }

    public List<HiCZoom> getBpZooms() {
        return bpZooms;
    }


    public List<HiCZoom> getFragZooms() {
        return fragZooms;
    }

    public boolean hasFrags() {
        return fragZooms != null && fragZooms.size() > 0;
    }

    public void setFragmentCounts(Map<String, Integer> map) {
        fragmentCounts = map;
    }

    public Map<String, Integer> getFragmentCounts() {
        return fragmentCounts;
    }
    /**
     * Return the "next" zoom level, relative to the current one, in the direction indicated
     *
     * @param zoom - current zoom level
     * @param b    -- direction, true == increasing resolution, false decreasing
     * @return Next zoom level
     */

    public HiCZoom getNextZoom(HiCZoom zoom, boolean b) {
        final HiC.Unit currentUnit = zoom.getUnit();
        List<HiCZoom> zoomList = currentUnit == HiC.Unit.BP ? bpZooms : fragZooms;

        if (b) {
            for (int i = 0; i < zoomList.size() - 1; i++) {
                if (zoom.equals(zoomList.get(i))) return zoomList.get(i + 1);
            }
            return zoomList.get(zoomList.size() - 1);

        } else {
            // Decreasing
            for (int i = zoomList.size() - 1; i > 0; i--) {
                if (zoom.equals(zoomList.get(i))) {
                    return zoomList.get(i - 1);
                }
            }
            return zoomList.get(0);
        }
    }


    public double[] getEigenvector(Chromosome chr, HiCZoom zoom, int number, NormalizationType type) {

        String key = chr.getName() + "_" + zoom.getKey() + "_" + number + "_" + type;
        if (!eigenvectorCache.containsKey(key)) {

            double[] eigenvector;
            eigenvector = reader.readEigenvector(chr.getName(), zoom, number, type.toString());

            if (eigenvector == null) {
                ExpectedValueFunction df = getExpectedValues(zoom, type);
                Matrix m = getMatrix(chr, chr);
                MatrixZoomData mzd = m.getZoomData(zoom);
                if (df != null && mzd.getPearsons(df) != null) {
                    eigenvector = mzd.computeEigenvector(df, number);
                } else {
                    eigenvector = new double[0];
                }
            }
            eigenvectorCache.put(key, eigenvector);
        }

        return eigenvectorCache.get(key);

    }

    public NormalizationVector getNormalizationVector(int chrIdx, HiCZoom zoom, NormalizationType type) {

        String key = NormalizationVector.getKey(type, chrIdx, zoom.getUnit().toString(), zoom.getBinSize());
        if (type == NormalizationType.NONE) {
            return null;
        } else if (type == NormalizationType.LOADED) {
            return loadedNormalizationVectors == null ? null : loadedNormalizationVectors.get(key);

        } else if (!normalizationVectorCache.containsKey(key)) {

            try {
                NormalizationVector nv = reader.readNormalizationVector(type, chrIdx, zoom.getUnit(), zoom.getBinSize());
                normalizationVectorCache.put(key, nv);
            } catch (IOException e) {
                normalizationVectorCache.put(key, null);
                // TODO -- warn user
                e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
            }
        }

        return normalizationVectorCache.get(key);

    }


    public void putLoadedNormalizationVector(int chrIdx, int resolution, double[] data, double[] exp) {
        NormalizationVector normalizationVector = new NormalizationVector(NormalizationType.LOADED, chrIdx, HiC.Unit.BP, resolution, data);
        if (loadedNormalizationVectors == null) {
            loadedNormalizationVectors = new HashMap<String, NormalizationVector>();

        }
        loadedNormalizationVectors.put(normalizationVector.getKey(), normalizationVector);
        HiCZoom zoom = new HiCZoom(HiC.Unit.BP, resolution);
        String key = zoom.getKey() + "_LOADED";
        ExpectedValueFunctionImpl function = (ExpectedValueFunctionImpl) getExpectedValues(zoom, NormalizationType.KR);

        ExpectedValueFunctionImpl df = new ExpectedValueFunctionImpl(NormalizationType.LOADED, "BP", resolution, exp, function.getNormFactors());
        expectedValueFunctionMap.put(key, df);
    }



    private String findRestrictionEnzyme(int sites) {
        if (genomeId == null) return null;

        if (genomeId.equals("anasPlat1")) {
            if (sites == 13393) return "DpnII/MboI";
        } else if (genomeId.equals("canFam3")) {
            if (sites == 345776) return "DpnII/MboI";
        } else if (genomeId.equals("dMel")) {
            // arm_2L
            if (sites == 60924) return "DpnII/MboI";
            if (sites == 6742) return "HindIII";
            if (sites == 185217) return "MseI";
        } else if (genomeId.equals("galGal4")) {
            if (sites == 465673) return "DpnII/MboI";
        } else if (genomeId.equals("hg18")) {
            if (sites == 575605) return "DpnII/MboI";
            if (sites == 64338) return "HindIII";
        } else if (genomeId.equals("hg19") || genomeId.equals("hg19_contig")) {
         // chromosome 1
            if (sites == 22706) return "Acc65I";
            if (sites == 4217) return "AgeI";
            if (sites == 158473) return "BseYI";
            if (sites == 74263) return "BspHI";
            if (sites == 60834) return "BstUI2";
            if (sites == 2284472) return "CpG";
            if (sites == 576357) return "DpnII/MboI";
            if (sites == 139125) return "HinP1I";
            if (sites == 64395) return "HindIII";
            if (sites == 160930) return "HpyCH4IV2";
            if (sites == 1632) return "MluI";
            if (sites == 1428208) return "MseI";
            if (sites == 194423) return "MspI";
            if (sites == 59852) return "NcoI";
            if (sites == 22347) return "NheI";
            if (sites == 1072254) return "NlaIII";
            if (sites == 1128) return "NruI";
            if (sites == 2344) return "SaII";
            if (sites == 1006921) return "StyD4I";
            if (sites == 256163) return "StyI";
            if (sites == 119506) return "TaqI2";
            if (sites == 9958) return "XhoI";
            if (sites == 31942) return "XmaI";
        } else if (genomeId.equals("mm9")) {
            // chr1
            if (sites == 479082) return "DpnII/MboI";
            if (sites == 62882) return "HindIII";
            if (sites == 1157974) return "MseI";
            if (sites == 60953) return "NcoI";
            if (sites == 933321) return "NlaIII";
        } else if (genomeId.equals("Pf3D7")) {
            if (sites == 13) return "DpnII/MboI";
        } else if (genomeId.equals("sacCer3")) {
            // No restriction site file for this so unknown
        } else if (genomeId.equals("sCerS288c")) {
            // chrI
            if (sites == 65) return "HindIII";
        } else if (genomeId.equals("susScr3")) {
            if (sites == 801622) return "DpnII/MboI";
        }
        return null;
    }

    public void setAttributes(String key, String value) {
        if (this.attributes == null) {
            attributes = new HashMap<String, String>();
        }
        attributes.put(key, value);
    }

}
