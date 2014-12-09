/*
 * Copyright (C) 2011-2014 Aiden Lab - All Rights Reserved
 * SOFTWARE COPYRIGHT NOTICE
 * This software and its documentation are the copyright of the Aiden Lab All rights are reserved.
 *
 * This software is supplied without any warranty or guaranteed support whatsoever. Aiden Lab is not responsible for its use, misuse, or functionality.
 *
 * This software is licensed under the terms of the GNU Lesser General Public License (LGPL),
 * Version 2.1 which is available at http://www.opensource.org/licenses/lgpl-2.1.php.
 */

package juicebox.data;

import juicebox.HiC;
import juicebox.matrix.BasicMatrix;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;

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
