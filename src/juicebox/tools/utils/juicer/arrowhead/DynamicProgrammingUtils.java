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

package juicebox.tools.utils.juicer.arrowhead;

import juicebox.tools.utils.common.MatrixTools;
import org.apache.commons.math.linear.RealMatrix;

/**
 * Created by muhammadsaadshamim on 6/3/15.
 */
class DynamicProgrammingUtils {


    /**
     * Calculate cumulative sums across upper right matrix
     * iterative result is entry to left + entry below + orig value - diagonal down (else we double count)
     *
     * @param matrix
     * @param superDiagonal
     * @return
     */
    public static RealMatrix sum(RealMatrix matrix, int superDiagonal) {

        int n = Math.min(matrix.getRowDimension(), matrix.getColumnDimension());

        RealMatrix diagMatrix = MatrixTools.cleanArray2DMatrix(matrix.getRowDimension(), matrix.getColumnDimension());
        if (superDiagonal <= 0) {
            diagMatrix = MatrixTools.extractDiagonal(matrix);
            superDiagonal = 1;
        }

        // d = distance from diagonal
        for (int d = superDiagonal; d < n; d++) {
            // i = row, column is i +d;

            for (int i = 0; i < n - d; i++) {
                diagMatrix.setEntry(i, i + d,
                        diagMatrix.getEntry(i, i + d - 1) + diagMatrix.getEntry(i + 1, i + d) +
                                matrix.getEntry(i, i + d) - diagMatrix.getEntry(i + 1, i + d - 1));
            }
        }
        return diagMatrix;
    }

    /**
     * Dynamic programming to calculate "right" matrix
     * Initialize by setting the diagonal to the diagonal of original
     * Iterate to the right and up.
     *
     * @param matrix
     * @param maxSize
     * @return rightMatrix
     */
    public static RealMatrix right(RealMatrix matrix, int maxSize) {

        RealMatrix rightMatrix = MatrixTools.extractDiagonal(matrix);
        int n = rightMatrix.getRowDimension();

        // j is column, i is row
        for (int j = 1; j < n; j++) {
            int endPoint = Math.max(j - 1 - maxSize, 0);
            for (int i = j - 1; i >= endPoint; i--) {
                rightMatrix.setEntry(i, j, matrix.getEntry(i, j) + rightMatrix.getEntry(i + 1, j));
            }
        }
        return rightMatrix;
    }

    /**
     * Dynamic programming to calculate "upper" matrix
     * Initialize by setting the diagonal to the diagonal of original
     * Iterate down (for each row) and to the left.
     *
     * @param matrix
     * @param maxSize
     * @return upperMatrix
     */
    public static RealMatrix upper(RealMatrix matrix, int maxSize) {

        RealMatrix upperMatrix = MatrixTools.extractDiagonal(matrix);
        int n = upperMatrix.getRowDimension();

        // j is column, i is row
        for (int i = 0; i < n; i++) {
            int endPoint = Math.min(i + 1 + maxSize, n - 1);
            for (int j = i + 1; j <= endPoint; j++) {
                upperMatrix.setEntry(i, j, matrix.getEntry(i, j) + upperMatrix.getEntry(i, j - 1));
            }
        }
        return upperMatrix;
    }

}
