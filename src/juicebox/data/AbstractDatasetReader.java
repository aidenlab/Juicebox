/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2015 Broad Institute, Aiden Lab
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
import juicebox.matrix.BasicMatrix;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import org.apache.log4j.Logger;
import org.broad.igv.util.FileUtils;
import org.broad.igv.util.ParsingUtils;
import org.broad.igv.util.collections.DoubleArrayList;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

/**
 * Abstract base class for methods that can be shared by V1 and V2 readers.
 *
 * @author jrobinso
 *         Date: 12/22/12
 *         Time: 10:15 AM
 */
public abstract class AbstractDatasetReader implements DatasetReader {

    private static final Logger log = Logger.getLogger(AbstractDatasetReader.class);

    final String path;

    AbstractDatasetReader(String path) {
        this.path = path;
    }

    public BasicMatrix readPearsons(String chr1Name, String chr2Name, HiCZoom zoom, NormalizationType type) throws IOException {

        // TODO -- need to use zoom unit (BP or FRAG)
        String rootPath = FileUtils.getParent(path);
        String folder = rootPath + "/" + chr1Name;
        String file = "pearsons" + "_" + chr1Name + "_" + chr2Name + "_" + zoom.getBinSize() + "_" + type + ".bin";
        String fullPath = folder + "/" + file;

        if (FileUtils.resourceExists(fullPath)) {
            return ScratchPad.readPearsons(fullPath);
        } else {
            return null;
        }

    }

    public String getPath() {
        return path;
    }

    @Override
    public double[] readEigenvector(String chrName, HiCZoom zoom, int number, String type) {


        double[] eigenvector = null;

        // If there's an eigenvector file load it
        String rootPath = FileUtils.getParent(path);
        String folder = rootPath + "/" + chrName;
        String eigenFile = "eigen" + "_" + chrName + "_" + chrName + "_" + zoom.getBinSize() + "_" + type + ".wig";
        String fullPath = folder + "/" + eigenFile;

        if (FileUtils.resourceExists(fullPath)) {
            System.out.println("Reading " + fullPath);


            //TODO Lots of assumptions made here about structure of wig file
            BufferedReader br = null;

            try {
                br = ParsingUtils.openBufferedReader(fullPath);
                br = new BufferedReader(new InputStreamReader(ParsingUtils.openInputStream(fullPath)), HiCGlobals.bufferSize);

                String nextLine = br.readLine();  // The track line, ignored
                DoubleArrayList arrayList = new DoubleArrayList(10000);  // TODO -- can size this exactly
                while ((nextLine = br.readLine()) != null) {
                    if (nextLine.startsWith("track") || nextLine.startsWith("fixedStep") || nextLine.startsWith("#")) {
                        continue;
                    }
                    try {
                        arrayList.add(Double.parseDouble(nextLine));
                    } catch (NumberFormatException e) {
                        arrayList.add(Double.NaN);
                    }
                }
                return arrayList.toArray();
            } catch (IOException e) {
                log.error("Error reading eigenvector", e);
            } finally {
                if (br != null) try {
                    br.close();
                } catch (IOException e) {
                    log.error("Error reading eigenvector", e);
                }
            }
        } else {
            System.out.println("Can't find eigenvector" + fullPath);
        }
        return null;

    }

    @Override
    public NormalizationVector readNormalizationVector(NormalizationType type, int chrIdx, HiC.Unit unit, int binSize) throws IOException {
        return null;  // Override as necessary
    }

    @Override
    public String readStats() throws IOException {
        return null; // Override for Combined Dataset Reader
    }

}
