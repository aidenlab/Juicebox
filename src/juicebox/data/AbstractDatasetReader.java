package juicebox.data;

import org.apache.log4j.Logger;
import juicebox.HiC;
import juicebox.HiCZoom;
import juicebox.NormalizationType;
import juicebox.matrix.BasicMatrix;
import org.broad.igv.util.FileUtils;
import org.broad.igv.util.ParsingUtils;
import org.broad.igv.util.collections.DoubleArrayList;

import java.io.BufferedReader;
import java.io.IOException;

/**
 * Abstract base class for methods that can be shared by V1 and V2 readers.
 *
 * @author jrobinso
 *         Date: 12/22/12
 *         Time: 10:15 AM
 */
public abstract class AbstractDatasetReader implements DatasetReader {

    private static Logger log = Logger.getLogger(AbstractDatasetReader.class);

    protected String path;

    public AbstractDatasetReader(String path) {
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
