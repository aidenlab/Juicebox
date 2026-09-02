/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2023 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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

import com.google.common.primitives.Ints;
import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.data.basics.Chromosome;
import juicebox.tools.dev.Private;
import juicebox.tools.utils.original.Preprocessor;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationHandler;
import juicebox.windowui.NormalizationType;
import org.broad.igv.util.FileUtils;
import org.broad.igv.util.Pair;
import org.broad.igv.util.ResourceLocator;
import org.broad.igv.util.collections.LRUCache;

import javax.swing.*;
import java.awt.event.ActionListener;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.text.ParseException;
import java.util.*;

/**
 * @author jrobinso
 * @since Aug 12, 2010
 */
public class Dataset {

    private final Map<String, Matrix> matrices = new HashMap<>(625);
    private final DatasetReader reader;
    private final LRUCache<String, double[]> eigenvectorCache;
    private final LRUCache<String, NormalizationVector> normalizationVectorCache;
    private final Map<String, NormalizationVector> normalizationsVectorsOnlySavedInRAMCache;
    Map<String, ExpectedValueFunction> expectedValueFunctionMap;
    String genomeId;
    String restrictionEnzyme = null;
    List<HiCZoom> bpZooms, dynamicZooms, fragZooms;
    private int v9DepthBase = 0;
    private List<Integer> bpZoomResolutions;
    private Map<String, String> attributes;
    private Map<String, Integer> fragmentCounts;
    protected NormalizationHandler normalizationHandler = new NormalizationHandler();
    private List<NormalizationType> normalizationTypes;
    private ChromosomeHandler chromosomeHandler;

    public Dataset(DatasetReader reader) {
        this.reader = reader;
        eigenvectorCache = new LRUCache<>(25);
        normalizationVectorCache = new LRUCache<>(25);
        normalizationsVectorsOnlySavedInRAMCache = new HashMap<>();
        normalizationTypes = new ArrayList<>();
    }

    public Matrix getMatrix(Chromosome chr1, Chromosome chr2) {

        // order is arbitrary, convention is lower # chr first
        if (chr1 == null || chr2 == null) return null;

        //System.out.println("from dataset");
        String key = Matrix.generateKey(chr1, chr2);
        Matrix m = matrices.get(key);

        if (m == null && reader != null) {
            try {
                // custom chromosome is handled as separate case
                //if (chromosomeHandler.isCustomAPAChromosome(chr1) || chromosomeHandler.isCustomAPAChromosome(chr2)) {
                //    System.err.println("APA Index key is " + key);
                //    m = Matrix.createCustomChromosomeMatrix(chr1, chr2, chromosomeHandler, matrices, reader);
                //} else
                if (chromosomeHandler.isCustomChromosome(chr1) || chromosomeHandler.isCustomChromosome(chr2)) {
                    if (HiCGlobals.printVerboseComments) System.out.println("Custom Chromosome Index key is " + key);
                    m = Matrix.createCustomChromosomeMatrix(chr1, chr2, chromosomeHandler, matrices, reader);
                } else if (HiCGlobals.isAssemblyMatCheck) {
                    m = Matrix.createAssemblyChromosomeMatrix(chromosomeHandler, matrices, reader);
                } else {
                    m = reader.readMatrix(key);
                }

                matrices.put(key, m);

            } catch (Exception e) {
                System.err.println("Error fetching matrix for: " + chr1.getName() + "-" + chr2.getName() +
                        " in " + reader.getPath());
                e.printStackTrace();
            }
        }

        return m;
    }

    public void addDynamicResolution(int newRes) {

        int highRes = -1;
        for (int potentialRes : bpZoomResolutions) {
            if (potentialRes < newRes && potentialRes > highRes && newRes % potentialRes == 0) {
                highRes = potentialRes;
            }
        }
        if (highRes < 0) {
            System.err.println("No suitable higher resolution found");
            return;
        }

        for (Matrix matrix : matrices.values()) {
            matrix.createDynamicResolutionMZD(new Pair<>(newRes, highRes), true);
        }
        dynamicZooms.add(new HiCZoom(HiC.Unit.BP, newRes));
    }


    public ResourceLocator getSubcompartments() {
        ResourceLocator locator = null;

        String path = reader.getPath();
        //Special case for combined maps:
        if (path == null) {
            return null;
        }

        if (path.contains("gm12878/in-situ/combined")) {
            path = path.substring(0, path.lastIndexOf('.'));
            if (path.lastIndexOf("_30") > -1) {
                path = path.substring(0, path.lastIndexOf("_30"));
            }

            String location = path + "_subcompartments.bed";
            locator = new ResourceLocator(location);

            locator.setName("Subcompartments");
        }
        return locator;
    }

    public ResourceLocator getSuperLoops() {
        ResourceLocator locator = null;

        String path = reader.getPath();
        //Special case for combined maps:
        if (path == null) {
            return null;
        }

        if (path.contains("gm12878/in-situ/combined")) {
            path = path.substring(0, path.lastIndexOf('.'));

            if (path.lastIndexOf("_30") > -1) {
                path = path.substring(0, path.lastIndexOf("_30"));
            }

            String location = path + "_chrX_superloop_list.txt";
            locator = new ResourceLocator(location);

            locator.setName("ChrX super loops");
        }
        return locator;
    }

    public ResourceLocator getPeaks() {

        String path = reader.getPath();

        //Special case for combined maps:
        if (path == null) {
            return null;
        }

        path = path.substring(0, path.lastIndexOf('.'));


        if (path.lastIndexOf("_30") > -1) {
            path = path.substring(0, path.lastIndexOf("_30"));
        }

        String location = path + "_peaks.txt";

        if (FileUtils.resourceExists(location)) {
            return new ResourceLocator(location);
        } else {
            location = path + "_loops.txt";
            if (FileUtils.resourceExists(location)) {
                return new ResourceLocator(location);
            } else {
                return null;
            }
        }

    }

    public ResourceLocator getBlocks() {

        String path = reader.getPath();

        //Special case for combined maps:
        if (path == null) {
            return null;
        }

        path = path.substring(0, path.lastIndexOf('.'));

        if (path.lastIndexOf("_30") > -1) {
            path = path.substring(0, path.lastIndexOf("_30"));
        }

        String location = path + "_blocks.txt";

        if (FileUtils.resourceExists(location)) {
            return new ResourceLocator(location);
        } else {
            location = path + "_domains.txt";
            if (FileUtils.resourceExists(location)) {
                return new ResourceLocator(location);
            } else {
                return null;
            }

        }

    }

    public void setAttributes(Map<String, String> map) {
        this.attributes = map;
        try {
            v9DepthBase = Integer.parseInt(attributes.get(Preprocessor.V9_DEPTH_BASE));
        } catch (Exception e) {
            v9DepthBase = 0;
        }
    }


    public List<NormalizationType> getNormalizationTypes() {
        return normalizationTypes;
    }

    public void setNormalizationTypes(List<NormalizationType> normalizationTypes) {
        this.normalizationTypes = normalizationTypes;
    }

    public void addNormalizationType(NormalizationType type) {
        if (!normalizationTypes.contains(type)) normalizationTypes.add(type);
    }

    public int getNumberZooms(HiC.Unit unit) {
        return unit == HiC.Unit.BP ? bpZooms.size() + dynamicZooms.size() : fragZooms.size();
    }

    // todo deprecate
    public HiCZoom getZoom(HiC.Unit unit, int index) {
        return unit == HiC.Unit.BP ? bpZooms.get(index) : fragZooms.get(index);
    }

    public HiCZoom getZoomForBPResolution(Integer resolution) {
        for (HiCZoom zoom : bpZooms) {
            if (zoom.getBinSize() == resolution) {
                return zoom;
            }
        }
        for (HiCZoom zoom : dynamicZooms) {
            if (zoom.getBinSize() == resolution) {
                return zoom;
            }
        }
        return null;
    }

    public ExpectedValueFunction getExpectedValues(HiCZoom zoom, NormalizationType type) {
        if (expectedValueFunctionMap == null || zoom == null || type == null) return null;
        String key = ExpectedValueFunctionImpl.getKey(zoom, type);
        return expectedValueFunctionMap.get(key);
    }

    public ExpectedValueFunction getExpectedValuesOrExit(HiCZoom zoom, NormalizationType type, Chromosome chromosome, boolean isIntra) {
        ExpectedValueFunction df = getExpectedValues(zoom, type);
        if (isIntra && df == null) {
            System.err.println("O/E data not available at " + chromosome.getName() + " " + zoom + " " + type);
            System.exit(14);
        }
        return df;
    }

    public Map<String, ExpectedValueFunction> getExpectedValueFunctionMap() {
        return expectedValueFunctionMap;
    }

    public void setExpectedValueFunctionMap(Map<String, ExpectedValueFunction> df) {
        this.expectedValueFunctionMap = df;
    }

    public ChromosomeHandler getChromosomeHandler() {
        return chromosomeHandler;
    }

    public void setChromosomeHandler(ChromosomeHandler chromosomeHandler) {
        this.chromosomeHandler = chromosomeHandler;
    }

    public int getVersion() {
        return reader.getVersion();
    }

    public String getGenomeId() {
        return genomeId;
    }

    public void setGenomeId(String genomeId) {
        if (genomeId.equals("GRCm38"))
            genomeId = "mm10";
        this.genomeId = genomeId;
    }

    public String getRestrictionEnzyme() {
        return restrictionEnzyme;
    }

    void setRestrictionEnzyme(int nSites) {
        restrictionEnzyme = findRestrictionEnzyme(nSites);
    }

    private String getSoftware() {
        if (attributes != null) return attributes.get(Preprocessor.SOFTWARE);
        else return null;
    }

    public String getHiCFileScalingFactor() {
        if (attributes != null) return attributes.get(Preprocessor.HIC_FILE_SCALING);
        else return null;
    }

    public String getStatistics() {
        String stats = null;
        if (attributes != null) stats = attributes.get(Preprocessor.STATISTICS);
        if (stats != null && (!stats.contains("<table>")) && HiCGlobals.guiIsCurrentlyActive) {
            return convertStats(stats);
        }
        return stats;
    }

    private String convertStats(String oldStats) {
        HashMap<String, String> statsMap = new HashMap<>();
        StringTokenizer lines = new StringTokenizer(oldStats, "\n");
        DecimalFormat decimalFormat = new DecimalFormat("0.00%");
        NumberFormat numberFormat = NumberFormat.getNumberInstance(Locale.US);
        while (lines.hasMoreTokens()) {
            String current = lines.nextToken();
            StringTokenizer colon = new StringTokenizer(current, ":");
            if (colon.countTokens() != 2) {
                System.err.println("Incorrect form in original statistics attribute. Offending line:");
                System.err.println(current);
            } else { // Appears to be correct format, convert files as appropriate
                String label = colon.nextToken();
                String value = colon.nextToken();
                statsMap.put(label, value);
            }
        }
        String newStats = "";
        int sequenced = -1;
        int unique = -1;

        newStats += "<table><tr><th colspan=2>Experiment Information</th></tr>\n" +
                "        <tr> <td> Experiment #:</td> <td>";
        String filename = reader.getPath();
        boolean mapq30 = filename.lastIndexOf("_30") > 0;
        String[] parts = filename.split("/");
        newStats += parts[parts.length - 2];
        newStats += "</td></tr>";
        newStats += "<tr> <td> Restriction Enzyme:</td><td>";
        newStats += getRestrictionEnzyme() + "</td></tr>";
        if (statsMap.containsKey("Experiment description")) {
            String value = statsMap.get("Experiment description").trim();
            if (!value.isEmpty())
                newStats += "<tr><td>Experiment Description:</td><td>" + value + "</td></tr>";
        }
        if (getSoftware() != null)  {
            newStats += "<tr> <td> Software: </td><td>" + getSoftware() + "</td></tr>";
        }
        if (getHiCFileScalingFactor() != null) {
            newStats += "<tr> <td> File Scaling: </td><td>" + getHiCFileScalingFactor() + "</td></tr>";
        }

        newStats += "<tr><th colspan=2>Alignment Information</th></tr>\n" +
                "        <tr> <td> Reference Genome:</td>";
        newStats += "<td>" + genomeId + "</td></tr>";
        newStats += "<tr> <td> MAPQ Threshold: </td><td>";
        if (mapq30) newStats += "30";
        else newStats += "1";
        newStats += "</td></tr>";



      /*  <table>
        <tr>
        <th colspan=2>Experiment Information</th></tr>
        <tr> <td> Experiment #:</td> <td>HIC034</td></tr>
        <tr> <td> Cell Type: </td><td>GM12878</td></tr>
        <tr> <td> Protocol: </td><td>dilution</td></tr>
        <tr> <td> Restriction Enzyme:</td><td>HindIII</td></tr>
        <tr> <td> Crosslinking: </td><td>1% FA, 10min, RT</td></tr>
        <tr> <td> Biotin Base: </td><td>bio-dCTP</td></tr>
        <tr> <td> Ligation Volume: </td><td>8ml</td></tr>
        <tr></tr>
        <tr><th colspan=2>Alignment Information</th></tr>
        <tr> <td> Reference Genome:</td><td>hg19</td></tr>
        <tr> <td> MAPQ Threshold: </td><td>1</td></tr>
        <tr></tr>
        <tr><th colspan=2>Sequencing Information</th></tr>
        <tr> <td> Instrument:  </td> <td>HiSeq 2000</td></tr>
        <tr> <td> Read 1 Length:  </td> <td>101</td></tr>
        <tr> <td> Read 2 Length:  </td> <td>101</td></tr>
        </table>
         */

        newStats += "</table><table>";
        if (statsMap.containsKey("Total") || statsMap.containsKey("Sequenced Read Pairs")) {
            newStats += "<tr><th colspan=2>Sequencing</th></tr>";
            newStats += "<tr><td>Sequenced Reads:</td>";
            String value = "";
            try {
                if (statsMap.containsKey("Total")) value = statsMap.get("Total").trim();
                else value = statsMap.get("Sequenced Read Pairs").trim();
                sequenced = numberFormat.parse(value).intValue();
            } catch (ParseException error) {
                sequenced = -1;
            }
            newStats += "<td>" + value + "</td></tr>";
            // TODO: add in Total Bases
        }
        if (statsMap.containsKey(" Regular") || statsMap.containsKey(" Normal Paired")) {
            newStats += "<tr></tr>";
            newStats += "<tr><th colspan=2>Alignment (% Sequenced Reads)</th></tr>";
            newStats += "<tr><td>Normal Paired:</td>";
            newStats += "<td>";
            if (statsMap.containsKey(" Regular")) newStats += statsMap.get(" Regular");
            else newStats += statsMap.get(" Normal Paired");
            newStats += "</td></tr>";
        }
        if (statsMap.containsKey(" Normal chimeric") || statsMap.containsKey(" Chimeric Paired")) {
            newStats += "<tr><td>Chimeric Paired:</td>";
            newStats += "<td>";
            if (statsMap.containsKey(" Normal chimeric")) newStats += statsMap.get(" Normal chimeric");
            else newStats += statsMap.get(" Chimeric Paired");
            newStats += "</td></tr>";
        }
        if (statsMap.containsKey(" Abnormal chimeric") || statsMap.containsKey(" Chimeric Ambiguous")) {
            newStats += "<tr><td>Chimeric Ambiguous:</td>";
            newStats += "<td>";
            if (statsMap.containsKey(" Abnormal chimeric")) newStats += statsMap.get(" Abnormal chimeric");
            else newStats += statsMap.get(" Chimeric Ambiguous");
            newStats += "</td></tr>";
        }
        if (statsMap.containsKey(" Unmapped")) {
            newStats += "<tr><td>Unmapped:</td>";
            newStats += "<td>" + statsMap.get(" Unmapped") + "</td></tr>";
        }
        newStats += "<tr></tr>";
        newStats += "<tr><th colspan=2>Duplication and Complexity (% Sequenced Reads)</td></tr>";
        if (statsMap.containsKey(" Total alignable reads") || statsMap.containsKey("Alignable (Normal+Chimeric Paired)")) {
            newStats += "<tr><td>Alignable (Normal+Chimeric Paired):</td>";
            newStats += "<td>";
            if (statsMap.containsKey(" Total alignable reads")) newStats += statsMap.get(" Total alignable reads");
            else newStats += statsMap.get("Alignable (Normal+Chimeric Paired)");
            newStats += "</td></tr>";
        }
        if (statsMap.containsKey("Total reads after duplication removal")) {
            newStats += "<tr><td>Unique Reads:</td>";
            String value = statsMap.get("Total reads after duplication removal");
            try {
                unique = numberFormat.parse(value.trim()).intValue();
            } catch (ParseException error) {
                unique = -1;
            }

            newStats += "<td>" + value;

            if (sequenced != -1) {
                newStats += " (" + decimalFormat.format(unique / (float) sequenced) + ")";
            }
            newStats += "</td></tr>";
        } else if (statsMap.containsKey("Unique Reads")) {
            newStats += "<tr><td>Unique Reads:</td>";
            String value = statsMap.get("Unique Reads");
            newStats += "<td>" + value + "</td></tr>";
            if (value.indexOf('(') >= 0) {
                value = value.substring(0, value.indexOf('('));
            }

            try {
                unique = numberFormat.parse(value.trim()).intValue();
            } catch (ParseException error) {
                unique = -1;
            }
        }
        if (statsMap.containsKey("Duplicate reads")) {
            newStats += "<tr><td>PCR Duplicates:</td>";
            String value = statsMap.get("Duplicate reads");
            newStats += "<td>" + value;
            int num;
            try {
                num = numberFormat.parse(value.trim()).intValue();
            } catch (ParseException error) {
                num = -1;
            }
            if (sequenced != -1) {
                newStats += " (" + decimalFormat.format(num / (float) sequenced) + ")";
            }
            newStats += "</td></tr>";
        } else if (statsMap.containsKey("PCR Duplicates")) {
            newStats += "<tr><td>PCR Duplicates:</td>";
            newStats += "<td>" + statsMap.get("PCR Duplicates") + "</td></tr>";
        }
        if (statsMap.containsKey("Optical duplicates")) {
            newStats += "<tr><td>Optical Duplicates:</td>";
            String value = statsMap.get("Optical duplicates");
            int num;
            try {
                num = numberFormat.parse(value.trim()).intValue();
            } catch (ParseException error) {
                num = -1;
            }
            if (sequenced != -1 && num != -1) {
                newStats += " (" + decimalFormat.format(num / (float) sequenced) + ")";
            }
            newStats += "</td></tr>";
        } else if (statsMap.containsKey("Optical Duplicates")) {
            newStats += "<tr><td>Optical Duplicates:</td>";
            newStats += "<td>" + statsMap.get("Optical Duplicates") + "</td></tr>";
        }
        if (statsMap.containsKey("Library complexity (new)") || statsMap.containsKey("Library Complexity Estimate")) {
            newStats += "<tr><td><b>Library Complexity Estimate:</b></td>";
            newStats += "<td><b>";
            if (statsMap.containsKey("Library complexity (new)")) newStats += statsMap.get("Library complexity (new)");
            else newStats += statsMap.get("Library Complexity Estimate");
            newStats += "</b></td></tr>";
        }
        newStats += "<tr></tr>";
        newStats += "<tr><th colspan=2>Analysis of Unique Reads (% Sequenced Reads / % Unique Reads)</td></tr>";
        if (statsMap.containsKey("Intra-fragment Reads")) {
            newStats += "<tr><td>Intra-fragment Reads:</td>";
            String value = statsMap.get("Intra-fragment Reads");
            if (value.indexOf('(') > 0) value = value.substring(0, value.indexOf('('));
            newStats += "<td>" + value;
            int num;
            try {
                num = numberFormat.parse(value.trim()).intValue();
            } catch (ParseException error) {
                num = -1;
            }
            if (sequenced != -1 && num != -1 && unique != -1) {
                newStats += " (" + decimalFormat.format(num / (float) sequenced) +
                        " / " + decimalFormat.format(num / (float) unique) + ")";
            }
            newStats += "</td></tr>";
        }
        if (statsMap.containsKey("Non-uniquely Aligning Reads")) {
            newStats += "<tr><td>Below MAPQ Threshold:</td>";
            String value = statsMap.get("Non-uniquely Aligning Reads");
            newStats += "<td>" + value.trim();
            int num;
            try {
                num = numberFormat.parse(value).intValue();
            } catch (ParseException error) {
                num = -1;
            }
            if (sequenced != -1 && num != -1 && unique != -1) {
                newStats += " (" + decimalFormat.format(num / (float) sequenced) +
                        " / " + decimalFormat.format(num / (float) unique) + ")";
            }
            newStats += "</td></tr>";
        } else if (statsMap.containsKey("Below MAPQ Threshold")) {
            newStats += "<tr><td>Below MAPQ Threshold:</td>";
            newStats += "<td>" + statsMap.get("Below MAPQ Threshold") + "</td></tr>";
        }
        if (statsMap.containsKey("Total reads in current file")) {
            newStats += "<tr><td><b>Hi-C Contacts:</b></td>";
            String value = statsMap.get("Total reads in current file");
            newStats += "<td><b>" + value.trim();
            int num;
            try {
                num = numberFormat.parse(value).intValue();
            } catch (ParseException error) {
                num = -1;
            }
            if (sequenced != -1 && num != -1 && unique != -1) {
                newStats += " (" + decimalFormat.format(num / (float) sequenced) +
                        " / " + decimalFormat.format(num / (float) unique) + ")";
            }
            newStats += "</b></td></tr>";
            // Error checking
            if (statsMap.containsKey("HiC Contacts")) {
                int num2;
                try {
                    num2 = numberFormat.parse(statsMap.get("HiC Contacts").trim()).intValue();
                } catch (ParseException error) {
                    num2 = -1;
                }
                if (num != num2) {
                    System.err.println("Check files -- \"HiC Contacts\" should be the same as \"Total reads in current file\"");
                }
            }
        } else if (statsMap.containsKey("Hi-C Contacts")) {
            newStats += "<tr><td><b>Hi-C Contacts:</b></td>";
            newStats += "<td><b>" + statsMap.get("Hi-C Contacts") + "</b></td></tr>";

        }
        if (statsMap.containsKey("Ligations") || statsMap.containsKey(" Ligation Motif Present")) {
            newStats += "<tr><td>&nbsp;&nbsp;Ligation Motif Present:</td>";
            String value = statsMap.containsKey("Ligations") ? statsMap.get("Ligations") : statsMap.get(" Ligation Motif Present");
            newStats += "<td>" + value.substring(0, value.indexOf('('));
            int num;
            try {
                num = numberFormat.parse(value.trim()).intValue();
            } catch (ParseException error) {
                num = -1;
            }
            if (sequenced != -1 && num != -1 && unique != -1) {
                newStats += " (" + decimalFormat.format(num / (float) sequenced) +
                        " / " + decimalFormat.format(num / (float) unique) + ")";
            }
            newStats += "</td></tr>";
        }
        if (statsMap.containsKey("Five prime") && statsMap.containsKey("Three prime")) {
            newStats += "<tr><td>&nbsp;&nbsp;3' Bias (Long Range):</td>";
            String value = statsMap.get("Five prime");
            value = value.substring(value.indexOf('(') + 1);
            value = value.substring(0, value.indexOf('%'));
            int num1 = Math.round(Float.parseFloat(value));

            value = statsMap.get("Three prime");
            value = value.substring(value.indexOf('(') + 1);
            value = value.substring(0, value.indexOf('%'));
            int num2 = Math.round(Float.parseFloat(value));

            newStats += "<td>" + num2 + "% - " + num1 + "%</td></tr>";
        } else if (statsMap.containsKey(" 3' Bias (Long Range)")) {
            newStats += "<tr><td>&nbsp;&nbsp;3' Bias (Long Range):</td>";
            newStats += "<td>" + statsMap.get(" 3' Bias (Long Range)") + "</td></tr>";
        }
        if (statsMap.containsKey("Inner") && statsMap.containsKey("Outer") &&
                statsMap.containsKey("Left") && statsMap.containsKey("Right")) {
            newStats += "<tr><td>&nbsp;&nbsp;Pair Type % (L-I-O-R):</td>";
            String value = statsMap.get("Left");
            value = value.substring(value.indexOf('(') + 1);
            value = value.substring(0, value.indexOf('%'));
            int num1 = Math.round(Float.parseFloat(value));

            value = statsMap.get("Inner");
            value = value.substring(value.indexOf('(') + 1);
            value = value.substring(0, value.indexOf('%'));
            int num2 = Math.round(Float.parseFloat(value));

            value = statsMap.get("Outer");
            value = value.substring(value.indexOf('(') + 1);
            value = value.substring(0, value.indexOf('%'));
            int num3 = Math.round(Float.parseFloat(value));

            value = statsMap.get("Right");
            value = value.substring(value.indexOf('(') + 1);
            value = value.substring(0, value.indexOf('%'));
            int num4 = Math.round(Float.parseFloat(value));
            newStats += "<td>" + num1 + "% - " + num2 + "% - " + num3 + "% - " + num4 + "%</td></tr>";
        } else if (statsMap.containsKey(" Pair Type %(L-I-O-R)")) {
            newStats += "<tr><td>&nbsp;&nbsp;Pair Type % (L-I-O-R):</td>";
            newStats += "<td>" + statsMap.get(" Pair Type %(L-I-O-R)") + "</td></tr>";
        }
        newStats += "<tr></tr>";
        newStats += "<tr><th colspan=2>Analysis of Hi-C Contacts (% Sequenced Reads / % Unique Reads)</th></tr>";
        if (statsMap.containsKey("Inter")) {
            newStats += "<tr><td>Inter-chromosomal:</td>";
            String value = statsMap.get("Inter");
            newStats += "<td>" + value.substring(0, value.indexOf('('));
            int num;
            try {
                num = numberFormat.parse(value.trim()).intValue();
            } catch (ParseException error) {
                num = -1;
            }
            if (sequenced != -1 && num != -1 && unique != -1) {
                newStats += " (" + decimalFormat.format(num / (float) sequenced) +
                        " / " + decimalFormat.format(num / (float) unique) + ")";
            }
            newStats += "</td></tr>";
        } else if (statsMap.containsKey("Inter-chromosomal")) {
            newStats += "<tr><td>Inter-chromosomal:</td>";
            newStats += "<td>" + statsMap.get("Inter-chromosomal") + "</td></tr>";
        }
        if (statsMap.containsKey("Intra")) {
            newStats += "<tr><td>Intra-chromosomal:</td>";
            String value = statsMap.get("Intra");
            newStats += "<td>" + value.substring(0, value.indexOf('('));
            int num;
            try {
                num = numberFormat.parse(value.trim()).intValue();
            } catch (ParseException error) {
                num = -1;
            }
            if (sequenced != -1 && num != -1 && unique != -1) {
                newStats += " (" + decimalFormat.format(num / (float) sequenced) +
                        " / " + decimalFormat.format(num / (float) unique) + ")";
            }
            newStats += "</td></tr>";
        } else if (statsMap.containsKey("Intra-chromosomal")) {
            newStats += "<tr><td>Intra-chromosomal:</td>";
            newStats += "<td>" + statsMap.get("Intra-chromosomal") + "</td></tr>";
        }
        if (statsMap.containsKey("Small")) {
            newStats += "<tr><td>&nbsp;&nbsp;Short Range (&lt;20Kb):</td>";
            String value = statsMap.get("Small");
            newStats += "<td>" + value.substring(0, value.indexOf('('));
            int num;
            try {
                num = numberFormat.parse(value.trim()).intValue();
            } catch (ParseException error) {
                num = -1;
            }
            if (sequenced != -1 && num != -1 && unique != -1) {
                newStats += " (" + decimalFormat.format(num / (float) sequenced) +
                        " / " + decimalFormat.format(num / (float) unique) + ")";
            }
            newStats += "</td></tr>";
        } else if (statsMap.containsKey("Short Range (<20Kb)")) {
            newStats += "<tr><td>&nbsp;&nbsp;Short Range (&lt;20Kb):</td>";
            newStats += "<td>" + statsMap.get("Short Range (<20Kb)") + "</td></tr>";
        }
        if (statsMap.containsKey("Large")) {
            newStats += "<tr><td><b>&nbsp;&nbsp;Long Range (&gt;20Kb):</b></td>";
            String value = statsMap.get("Large");
            newStats += "<td><b>" + value.substring(0, value.indexOf('('));
            int num;
            try {
                num = numberFormat.parse(value.trim()).intValue();
            } catch (ParseException error) {
                num = -1;
            }
            if (sequenced != -1 && num != -1 && unique != -1) {
                newStats += " (" + decimalFormat.format(num / (float) sequenced) +
                        " / " + decimalFormat.format(num / (float) unique) + ")";
            }
            newStats += "</b></td></tr>";
        } else if (statsMap.containsKey("Long Range (>20Kb)")) {
            newStats += "<tr><td><b>&nbsp;&nbsp;Long Range (&gt;20Kb):</b></td>";
            newStats += "<td><b>" + statsMap.get("Long Range (>20Kb)") + "</b></td></tr>";
        }
        // Error checking
        if (statsMap.containsKey("Unique Reads")) {
            int num;
            try {
                num = numberFormat.parse(statsMap.get("Unique Reads").trim()).intValue();
            } catch (ParseException error) {
                num = -1;
            }
            if (num != unique) {
                System.err.println("Check files -- \"Unique Reads\" should be the same as \"Total reads after duplication removal\"");
            }
        }

        return newStats;
    }

    public String getGraphs() {
        if (attributes == null) return null;
        return attributes.get(Preprocessor.GRAPHS);
    }

    public List<HiCZoom> getBpZooms() {
        List<HiCZoom> zooms = new ArrayList<>(bpZooms);
        zooms.addAll(dynamicZooms);
        zooms.sort(Collections.reverseOrder());
        return zooms;
    }

    public void setBpZooms(int[] bpBinSizes) {

        bpZoomResolutions = Ints.asList(bpBinSizes);

        bpZooms = new ArrayList<>(bpBinSizes.length);
        for (int bpBinSize : bpZoomResolutions) {
            bpZooms.add(new HiCZoom(HiC.Unit.BP, bpBinSize));
        }
        dynamicZooms = new ArrayList<>();
    }

    public List<HiCZoom> getFragZooms() {
        return fragZooms;
    }

    public void setFragZooms(int[] fragBinSizes) {

        // Don't show fragments in restricted mode
//        if (MainWindow.isRestricted()) return;

        this.fragZooms = new ArrayList<>(fragBinSizes.length);
        for (int fragBinSize : fragBinSizes) {
            fragZooms.add(new HiCZoom(HiC.Unit.FRAG, fragBinSize));
        }
    }

    public boolean hasFrags() {
        return fragZooms != null && fragZooms.size() > 0;
    }

    public Map<String, Integer> getFragmentCounts() {
        return fragmentCounts;
    }

    public void setFragmentCounts(Map<String, Integer> map) {
        fragmentCounts = map;
    }
    
    /**
     * Return the "next" zoom level, relative to the current one, in the direction indicated
     *
     * @param zoom               - current zoom level
     * @param useIncreasingOrder -- direction, true == increasing resolution, false decreasing
     * @return Next zoom level
     */

    public HiCZoom getNextZoom(HiCZoom zoom, boolean useIncreasingOrder) {
        final HiC.Unit currentUnit = zoom.getUnit();
        List<HiCZoom> zoomList = currentUnit == HiC.Unit.BP ? getBpZooms() : fragZooms;

        // TODO MSS - is there a reason not to just rewrite this using indexOf? cleaner?
        if (useIncreasingOrder) {
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
            //eigenvector = reader.readEigenvector(chr.getName(), zoom, number, type.toString());

            ExpectedValueFunction df = getExpectedValues(zoom, type);
            Matrix m = getMatrix(chr, chr);
            MatrixZoomData mzd = m.getZoomData(zoom);
            if (df != null && mzd.getPearsons(df) != null && zoom.getBinSize() >= HiCGlobals.MAX_EIGENVECTOR_ZOOM) {
                eigenvector = mzd.computeEigenvector(df, number);
            } else {
                eigenvector = new double[0];
            }

            eigenvectorCache.put(key, eigenvector);
        }

        return eigenvectorCache.get(key);

    }

    public NormalizationVector getNormalizationVector(int chrIdx, HiCZoom zoom, NormalizationType type) {

        String key = NormalizationVector.getKey(type, chrIdx, zoom.getUnit().toString(), zoom.getBinSize());

        if (normalizationsVectorsOnlySavedInRAMCache.containsKey(key)) {
            return normalizationsVectorsOnlySavedInRAMCache.get(key);
        }

        if (type.equals(NormalizationHandler.NONE)) {
            return null;
        }  else if (!normalizationVectorCache.containsKey(key)) {
            try {
                NormalizationVector nv = reader.readNormalizationVector(type, chrIdx, zoom.getUnit(), zoom.getBinSize());
                normalizationVectorCache.put(key, nv);
            } catch (IOException e) {
                normalizationVectorCache.put(key, null);
            }
        }

        return normalizationVectorCache.get(key);
    }

    public NormalizationVector getPartNormalizationVector(int chrIdx, HiCZoom zoom, NormalizationType type, int bound1, int bound2) {
        String key = NormalizationVector.getKey(type, chrIdx, zoom.getUnit().toString(), zoom.getBinSize());
        NormalizationVector nv;

        if (type.equals(NormalizationHandler.NONE)) {
            return null;
        } else {
            try {
                nv = reader.readNormalizationVectorPart(type, chrIdx, zoom.getUnit(), zoom.getBinSize(), bound1, bound2);
            } catch (IOException e) {
                return null;
            }
        }

        return nv;
    }

    public void addNormalizationVectorDirectlyToRAM(NormalizationVector normalizationVector) {
        normalizationsVectorsOnlySavedInRAMCache.put(normalizationVector.getKey(), normalizationVector);
    }

    private String findRestrictionEnzyme(int sites) {
        if (genomeId == null) return null;

        if (Private.assessGenomeForRE(genomeId)) {
            if (sites == 13393) return "DpnII/MboI";
        } else if (Private.assessGenomeForRE3(genomeId)) {
            if (sites == 465673) return "DpnII/MboI";
        } else if (Private.assessGenomeForRE4(genomeId)) {
            if (sites == 801622) return "DpnII/MboI";
        } else if (genomeId.equals("canFam3")) {
            if (sites == 345776) return "DpnII/MboI";
        } else if (genomeId.equals("dMel")) {
            // arm_2L
            if (sites == 60924) return "DpnII/MboI";
            if (sites == 6742) return "HindIII";
            return Private.reForDMEL(sites);
        } else if (genomeId.equals("hg18")) {
            if (sites == 575605) return "DpnII/MboI";
            return Private.reForHG18(sites);
        } else if (genomeId.equals("hg19") || Private.assessGenomeForRE2(genomeId)) {
            if (sites == 576357) return "DpnII/MboI";
            if (sites == 64395) return "HindIII";
            if (sites == 59852) return "NcoI";
            return Private.reForHG19(sites);
        } else if (genomeId.equals("mm9")) {
            // chr1
            if (sites == 479082) return "DpnII/MboI";
            if (sites == 62882) return "HindIII";
            if (sites == 60953) return "NcoI";
            return Private.reForMM9(sites);
        } else if (genomeId.equals("mm10")) {
            if (sites == 480062) return "DpnII/MboI";
            if (sites == 63013) return "HindIII";
        } else if (genomeId.equals("Pf3D7")) {
            if (sites == 13) return "DpnII/MboI";
        } else if (genomeId.equals("sCerS288c")) {
            if (sites == 65) return "HindIII"; // chrI
        }
        return null;
    }

    public void setAttributes(String key, String value) {
        if (this.attributes == null) {
            attributes = new HashMap<>();
        }
        attributes.put(key, value);
    }

    public List<JCheckBox> getCheckBoxes(List<ActionListener> actionListeners) {
        return reader.getCheckBoxes(actionListeners);
    }

    public List<HiCZoom> getAllPossibleResolutions() {
        List<HiCZoom> resolutions = new ArrayList<>();
        resolutions.addAll(bpZooms);
        resolutions.addAll(dynamicZooms);
        resolutions.addAll(fragZooms);
        return resolutions;
    }

    public NormalizationHandler getNormalizationHandler() {
        return normalizationHandler;
    }

    public int getDepthBase() {
        return v9DepthBase;
    }

    public void clearCache(boolean onlyClearInter) {
        for (Matrix matrix : matrices.values()) {
            for (HiCZoom zoom : getBpZooms()) {
                try {
                    matrix.getZoomData(zoom).clearCache(onlyClearInter);
                } catch (Exception e) {
                    if (HiCGlobals.printVerboseComments) {
                        System.err.println("Clearing err: " + e.getLocalizedMessage());
                    }
                }
            }
        }
    }

    public void clearCache(boolean onlyClearInter, HiCZoom zoom) {
        for (Matrix matrix : matrices.values()) {
            try {
                matrix.getZoomData(zoom).clearCache(onlyClearInter);
            } catch (Exception e) {
                if (HiCGlobals.printVerboseComments) {
                    System.err.println("Clearing z_err: " + e.getLocalizedMessage());
                }
            }
        }
    }
}
