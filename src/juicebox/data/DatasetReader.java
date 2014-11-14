package juicebox.data;

import juicebox.HiC;
import juicebox.HiCZoom;
import juicebox.NormalizationType;
import juicebox.matrix.BasicMatrix;

import java.io.IOException;
import java.util.List;

/**
 * @author jrobinso
 *         Date: 10/17/12
 *         Time: 8:38 AM
 */
public interface DatasetReader {


    int getVersion();

    Dataset read() throws IOException;

    Matrix readMatrix(String key) throws IOException;

    Block readBlock(int blockNumber, MatrixZoomData zd) throws IOException;

    Block readNormalizedBlock(int blockNumber, MatrixZoomData zd, NormalizationType no) throws IOException;

    /**
     * Return the list of occupied block numbers for the given matrix.
     *
     * @param matrixZoomData
     * @return
     */
    List<Integer> getBlockNumbers(MatrixZoomData matrixZoomData);

    double[] readEigenvector(String chrName, HiCZoom zoom, int number, String type);

    void close();

    NormalizationVector readNormalizationVector(NormalizationType type, int chrIdx, HiC.Unit unit, int binSize) throws IOException;

    public BasicMatrix readPearsons(String chr1Name, String chr2Name, HiCZoom zoom, NormalizationType type) throws IOException;

    public String getPath();

    public String readStats() throws IOException;

}
