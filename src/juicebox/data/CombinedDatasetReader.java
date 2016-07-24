/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2016 Broad Institute, Aiden Lab
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
import juicebox.MainWindow;
import juicebox.matrix.BasicMatrix;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;

import javax.swing.*;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.ParseException;
import java.util.*;

//import java.util.List;

/**
 * @author jrobinso
 *         Date: 12/22/12
 *         Time: 10:10 AM
 */
public class CombinedDatasetReader implements DatasetReader {

    private final List<DatasetReaderV2> readers;
    private boolean hasFrags;
    private int version;

    public CombinedDatasetReader(List<DatasetReaderV2> readers) {
        this.readers = readers;
    }

    @Override
    public Dataset read() throws IOException {
        // Temporarily create a dataset for each reader, then merge them

        List<Dataset> tmpDatasets = new ArrayList<Dataset>();
        version = 100000;
        for (DatasetReader r : readers) {
            tmpDatasets.add(r.read());
            version = Math.min(version, r.getVersion());
        }

        return mergeDatasets(tmpDatasets);

    }


    @Override
    public int getVersion() {
        // Version is the minimum of all constituent datasets
        return version;
    }

    @Override
    public String getPath() {
        // we use this for peaks and blocks list, maybe the best thing to do is to somehow combine them
        return null;
    }

    @Override
    public String readStats() {
        // again we need to somehow combine from constituent datasets
        return null;
    }

    /**
     * @param key -- string identifier for matrix, concatenation of chromosome names
     * @return Merged matrices read in
     * @throws IOException
     */
    @Override

    public Matrix readMatrix(String key) throws IOException {
        //
        List<Matrix> tmpDatasets = new ArrayList<Matrix>();
        for (DatasetReader r : readers) {
            tmpDatasets.add(r.readMatrix(key));
        }

        return mergeMatrices(tmpDatasets);

    }

    @Override
    public Block readBlock(int blockNumber, MatrixZoomData zd) throws IOException {

        List<Block> blockList = new ArrayList<Block>();
        for (DatasetReader r : readers) {
            Block cb = r.readBlock(blockNumber, zd);
            if (cb != null) {
                blockList.add(cb);
            }
        }

        return (blockList.size() == 0) ? new Block(blockNumber) : mergeBlocks(blockList);

    }

    @Override
    public Block readNormalizedBlock(int blockNumber, MatrixZoomData zd, NormalizationType no) throws IOException {

        List<Block> blockList = new ArrayList<Block>();
        for (DatasetReader r : readers) {
            Block cb;
            cb = r.readNormalizedBlock(blockNumber, zd, no);
            if (cb != null) {
                blockList.add(cb);
            }
        }
        return blockList.size() == 0 ? new Block(blockNumber) : mergeBlocks(blockList);

    }

    /**
     * Return the block numbers of all occupied blocks.
     *
     * @param matrixZoomData Matrix
     * @return block numbers
     */
    @Override

    public List<Integer> getBlockNumbers(MatrixZoomData matrixZoomData) {

        Set<Integer> blockNumberSet = new HashSet<Integer>();
        for (DatasetReader r : readers) {
            blockNumberSet.addAll(r.getBlockNumbers(matrixZoomData));

        }
        List<Integer> blockNumbers = new ArrayList<Integer>(blockNumberSet);
        Collections.sort(blockNumbers);
        return blockNumbers;
    }

    @Override
    public double[] readEigenvector(String chr, HiCZoom zoom, int number, String type) {
        // Eigenvectors not supported for combined datasets
        return null;
    }

    @Override
    public void close() {
        for (DatasetReader r : readers) {
            r.close();
        }
    }

    @Override
    public NormalizationVector readNormalizationVector(NormalizationType type, int chrIdx, HiC.Unit unit, int binSize) throws IOException {
        return null; // Undefined for combined datasets
    }

    @Override
    public BasicMatrix readPearsons(String chr1Name, String chr2Name, HiCZoom zoom, NormalizationType type) throws IOException {
        // At this time combined datasets do not have precomputed pearsons.
        return null;
    }

    /**
     * Return a dataset that is an "intersection" of the supplied datasets.
     *
     * @param datasetList List of datasets to merge
     * @return new dataset
     */
    private Dataset mergeDatasets(List<Dataset> datasetList) {

        Dataset dataset = new Dataset(this);

        final Dataset firstDataset = datasetList.get(0);
        dataset.genomeId = firstDataset.getGenomeId();
        dataset.chromosomes = firstDataset.getChromosomes();

        String enzyme = firstDataset.getRestrictionEnzyme();
        hasFrags = enzyme != null;
        if (hasFrags) {
            for (Dataset ds : datasetList) {
                if (!ds.getRestrictionEnzyme().equals(enzyme)) {
                    hasFrags = false;
                    break;
                }
            }
        }

        // Assuming all datasets have same bp resolutions -- TODO check this
        dataset.bpZooms = firstDataset.getBpZooms();
        if (hasFrags) {
            dataset.fragZooms = firstDataset.getFragZooms();
            dataset.setFragmentCounts(firstDataset.getFragmentCounts());
        }

        // Expected values, just sum
        // Map key ==  unit_binSize
        Map<String, ExpectedValueFunction> dfMap = new HashMap<String, ExpectedValueFunction>();

        Collection<String> keys = firstDataset.getExpectedValueFunctionMap().keySet();
        Set<String> zoomsToRemove = new HashSet<String>();
        for (String key : keys) {
            if (!hasFrags && key.startsWith(HiC.Unit.FRAG.toString())) continue;
            List<ExpectedValueFunction> evFunctions = new ArrayList<ExpectedValueFunction>();
            boolean haveAll = true;
            for (Dataset ds : datasetList) {
                final ExpectedValueFunction e = ds.getExpectedValueFunctionMap().get(key);
                if (e == null) {
                    int idx = key.lastIndexOf("_");
                    String zoomKey = key.substring(0, idx);
                    zoomsToRemove.add(zoomKey);
                    haveAll = false;
                    break;
                } else {
                    evFunctions.add(e);
                }
            }
            if (haveAll) {
                ExpectedValueFunction combinedEV = mergeExpectedValues(evFunctions);
                dfMap.put(key, combinedEV);
            }
        }
        dataset.expectedValueFunctionMap = dfMap;

        if (zoomsToRemove.size() > 0) {
            List<HiCZoom> trimmedBpZooms = new ArrayList<HiCZoom>(dataset.bpZooms.size());
            for (HiCZoom zoom : dataset.bpZooms) {
                if (!zoomsToRemove.contains(zoom.getKey())) {
                    trimmedBpZooms.add(zoom);
                }
            }
            dataset.bpZooms = trimmedBpZooms;
            if (hasFrags) {
                List<HiCZoom> trimmedFragZooms = new ArrayList<HiCZoom>(dataset.bpZooms.size());
                for (HiCZoom zoom : dataset.fragZooms) {
                    if (!zoomsToRemove.contains(zoom.getKey())) {
                        trimmedFragZooms.add(zoom);
                    }
                }
                dataset.fragZooms = trimmedFragZooms;

            }
        }


        ArrayList<String> statisticsList = new ArrayList<String>();
        ArrayList<String> graphsList = new ArrayList<String>();
        HashSet<String> reList = new HashSet<String>();
        for (Dataset ds : datasetList) {
            try {
                statisticsList.add(ds.getStatistics());
                graphsList.add(ds.getGraphs());
                reList.add(ds.getRestrictionEnzyme());
            } catch (Exception e) {
                // TODO - test on hic file with no stats file specified
                e.printStackTrace();
                if (HiCGlobals.guiIsCurrentlyActive)
                    JOptionPane.showMessageDialog(MainWindow.getInstance(), "Unable to retrieve statistics for one of the maps.", "Error",
                            JOptionPane.ERROR_MESSAGE);
            }
        }


        Map<String, String> attributes = new HashMap<String, String>();
        attributes.put("statistics", mergeStatistics(statisticsList));
        attributes.put("graphs", mergeGraphs(graphsList));
        dataset.setAttributes(attributes);

        Iterator<?> it = reList.iterator();
        String newRestrictionEnzyme = "";
        while (it.hasNext()) newRestrictionEnzyme += it.next() + " ";
        dataset.restrictionEnzyme = newRestrictionEnzyme;

        // Set normalization types (for menu)
        LinkedHashSet<NormalizationType> normTypes = new LinkedHashSet<NormalizationType>();
        for (Dataset ds : datasetList) {
            List<NormalizationType> tmp = ds.getNormalizationTypes();
            if (tmp != null) normTypes.addAll(tmp);
        }
        for (Dataset ds : datasetList) {
            List<NormalizationType> tmp = ds.getNormalizationTypes();
            if (tmp != null) normTypes.retainAll(tmp);
        }
        dataset.setNormalizationTypes(new ArrayList<NormalizationType>(normTypes));

        return dataset;
    }


    private ExpectedValueFunction mergeExpectedValues(List<ExpectedValueFunction> densityFunctions) {

        try {

            ExpectedValueFunction protoFunction = densityFunctions.get(0);
            int binSize = protoFunction.getBinSize();
            HiC.Unit unit = protoFunction.getUnit();
            NormalizationType type = protoFunction.getNormalizationType();
            int len = protoFunction.getLength();

            for (ExpectedValueFunction df : densityFunctions) {
                if (df.getBinSize() != binSize || !df.getUnit().equals(unit) || df.getNormalizationType() != type) {
                    throw new RuntimeException("Attempt to merge incompatible expected values");
                }
                len = Math.min(df.getLength(), len);
            }
            double[] expectedValues = new double[len];

            for (ExpectedValueFunction df : densityFunctions) {
                double[] current = df.getExpectedValues();
                for (int i = 0; i < len; i++) {
                    expectedValues[i] += current[i];
                }
            }


            return new ExpectedValueFunctionImpl(protoFunction.getNormalizationType(), protoFunction.getUnit(), protoFunction.getBinSize(),
                    expectedValues, null);
        } catch (Exception e) {
            e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
            return null;
        }

    }

    private String mergeStatistics(List<String> statisticsList) {

        LinkedHashMap<String, String> statsMap = new LinkedHashMap<String, String>();


        int numberFiles = statisticsList.size();
        String newStatistics = "";
        for (String stats : statisticsList) {
            if (stats == null) return null;
            Scanner scanner = new Scanner(stats).useDelimiter("\n");
            while (scanner.hasNext()) {
                String[] results = scanner.next().split(":");
                if (results.length == 2) {
                    String key = results[0];
                    String prevValue = statsMap.get(key);
                    String value = results[1];
                    int index = value.indexOf("(");
                    if (index > -1) value = value.substring(0, index);

                    if (prevValue == null) {
                        statsMap.put(key, value);
                    } else {
                        statsMap.put(key, prevValue + ";" + value);
                    }
                }
            }
        }
        DecimalFormat decimalFormat = new DecimalFormat();
        for (Map.Entry<String, String> e : statsMap.entrySet()) {
            String key = e.getKey();
            String value = e.getValue();
            String[] results = value.split(";");
            if (results.length == numberFiles) {
                boolean add = true;
                long tmp = 0;
                for (int i = 0; i < numberFiles; i++) {
                    try {
                        tmp += decimalFormat.parse(results[i].trim()).longValue();
                    } catch (ParseException exception) {
                        add = false;

                    }
                    //tmp += Long.valueOf(results[i].trim());
                }
                if (add) newStatistics += key + ": " + decimalFormat.format(tmp) + "\n";
            }
        }
        return newStatistics;
    }

    private String mergeGraphs(List<String> graphsList) {
        long[] A = new long[2000];

        int[] mapq1 = new int[201];
        int[] mapq2 = new int[201];
        int[] mapq3 = new int[201];
        int[] inner = new int[100];
        int[] outer = new int[100];
        int[] right = new int[100];
        int[] left = new int[100];

        for (String graphs : graphsList) {
            if (graphs == null) {
                return null;
            }
            Scanner scanner = new Scanner(graphs);
            try {
                while (true) {
                    if (scanner.next().equals("[")) break;
                }
                //while (!scanner.next().equals("[")) ;
                for (int idx = 0; idx < 2000; idx++) {
                    A[idx] += scanner.nextLong();

                }

                while (true) {
                    if (scanner.next().equals("[")) break;
                }
                //while (!scanner.next().equals("[")) ;
                for (int idx = 0; idx < 201; idx++) {
                    mapq1[idx] += scanner.nextInt();
                    mapq2[idx] += scanner.nextInt();
                    mapq3[idx] += scanner.nextInt();

                }

                while (true) {
                    if (scanner.next().equals("[")) break;
                }
                //while (!scanner.next().equals("[")) ;
                for (int idx = 0; idx < 100; idx++) {
                    inner[idx] += scanner.nextInt();
                    outer[idx] += scanner.nextInt();
                    right[idx] += scanner.nextInt();
                    left[idx] += scanner.nextInt();
                }
            } catch (NoSuchElementException exception) {
                System.err.println("Graphing file improperly formatted");
                return null;
            }
        }
        String newGraph = "A = [\n";
        for (int idx = 0; idx < 2000; idx++) newGraph += A[idx] + " ";
        newGraph += "];\n";
        newGraph += "B = [\n";
        for (int idx = 0; idx < 201; idx++) newGraph += mapq1[idx] + " " + mapq2[idx] + " " + mapq3[idx] + "\n";
        newGraph += "];\n";
        newGraph += "D = [\n";
        for (int idx = 0; idx < 100; idx++)
            newGraph += inner[idx] + " " + outer[idx] + " " + right[idx] + " " + left[idx] + "\n";
        newGraph += "];\n";
        return newGraph;
    }


    private Matrix mergeMatrices(List<Matrix> matrixList) {

        Map<String, Double> averageCount = new HashMap<String, Double>();
        for (Matrix matrix : matrixList) {
            for (MatrixZoomData zd : matrix.bpZoomData) {
                String key = zd.getKey();
                Double avg = averageCount.get(key);
                if (avg == null) {
                    averageCount.put(key, zd.getAverageCount());
                } else if (avg >= 0) {
                    averageCount.put(key, avg + zd.getAverageCount());
                }
            }
            if (hasFrags) {
                for (MatrixZoomData zd : matrix.fragZoomData) {
                    String key = zd.getKey();
                    Double avg = averageCount.get(key);
                    if (avg == null) {
                        averageCount.put(key, zd.getAverageCount());
                    } else if (avg >= 0) {
                        averageCount.put(key, avg + zd.getAverageCount());
                    }
                }
            }
        }

        Matrix mergedMatrix = matrixList.get(0);

        for (MatrixZoomData zd : mergedMatrix.bpZoomData) {
            zd.reader = this;
            String key = zd.getKey();
            if (averageCount.containsKey(key)) {
                zd.setAverageCount(averageCount.get(key));
            }
        }
        if (hasFrags) {
            for (MatrixZoomData zd : mergedMatrix.fragZoomData) {
                zd.reader = this;
                String key = zd.getKey();
                if (averageCount.containsKey(key)) {
                    zd.setAverageCount(averageCount.get(key));
                }
            }
        } else {
            mergedMatrix.fragZoomData = null;
        }
        return mergedMatrix;
    }

    /**
     * Merge the contact records from multiple blocks to create a new block.  Contact records are sorted in row then
     * column order.
     *
     * @param blockList Blocks to merge
     * @return new Block
     */
    private Block mergeBlocks(List<Block> blockList) {
        // First combine contact records for all blocks
        final Block firstBlock = blockList.get(0);
        int repSize = firstBlock.getContactRecords().size();
        int blockNumber = firstBlock.getNumber(); // TODO -- this should be checked, all blocks should have same number

        HashMap<String, ContactRecord> mergedRecordMap = new HashMap<String, ContactRecord>(blockList.size() * repSize * 2);

        for (Block b : blockList) {
            Collection<ContactRecord> records = b.getContactRecords();
            for (ContactRecord rec : records) {
                String key = rec.getBinX() + "_" + rec.getBinY();
                ContactRecord mergedRecord = mergedRecordMap.get(key);
                if (mergedRecord == null) {
                    mergedRecord = new ContactRecord(rec.getBinX(), rec.getBinY(), rec.getCounts());
                    mergedRecordMap.put(key, mergedRecord);
                } else {
                    mergedRecord.incrementCount(rec.getCounts());
                }
            }
        }

        List<ContactRecord> mergedRecords = new ArrayList<ContactRecord>(mergedRecordMap.values());
        return new Block(blockNumber, mergedRecords);
    }

}
