/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2019 Broad Institute, Aiden Lab
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
import juicebox.matrix.BasicMatrix;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;

import javax.swing.*;
import java.awt.event.ActionListener;
import java.io.IOException;
import java.util.List;

/**
 * @author jrobinso
 *         Date: 10/17/12
 *         Time: 8:38 AM
 */
public interface DatasetReader {

    boolean isActive();

    void setActive(boolean status);

    int getVersion();

    Dataset read() throws IOException;

    Matrix readMatrix(String key) throws IOException;

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

    BasicMatrix readPearsons(String chr1Name, String chr2Name, HiCZoom zoom, NormalizationType type) throws IOException;

    String getPath();

    String readStats() throws IOException;

    List<JCheckBox> getCheckBoxes(List<ActionListener> actionListeners);

    NormalizationVector getNormalizationVector(int chr1Idx, HiCZoom zoom, NormalizationType normalizationType);

}
