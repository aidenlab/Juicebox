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

package juicebox.matrix;

import org.apache.commons.math.linear.RealMatrix;
import org.apache.commons.math.stat.StatUtils;
import org.broad.igv.util.collections.DoubleArrayList;

/**
 * Wrpas a apache commons RealMatrix.  We don't expose the apache class so we can use other implementations.
 *
 * @author jrobinso
 *         Date: 7/13/12
 *         Time: 1:02 PM
 */
public class RealMatrixWrapper implements BasicMatrix {

    private final RealMatrix matrix;
    private float lowerValue = -1;
    private float upperValue = 1;

    public RealMatrixWrapper(RealMatrix matrix) {
        this.matrix = matrix;
        computePercentiles();
    }

    @Override
    public float getEntry(int row, int col) {
        return (float) matrix.getEntry(row, col);
    }

    @Override
    public int getRowDimension() {
        return matrix.getRowDimension();
    }

    @Override
    public int getColumnDimension() {
        return matrix.getColumnDimension();
    }

    @Override
    public float getLowerValue() {
        return lowerValue;
    }

    @Override
    public float getUpperValue() {
        return upperValue;
    }

    @Override
    public void setEntry(int i, int j, float corr) {

    }

    private void computePercentiles() {

        // Statistics, other attributes
        DoubleArrayList flattenedDataList = new DoubleArrayList(matrix.getColumnDimension() * matrix.getRowDimension());
        double min = 1;
        double max = -1;
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                double value = matrix.getEntry(i, j);
                if (!Double.isNaN(value) && value != 1) {
                    min = value < min ? value : min;
                    max = value > max ? value : max;
                    flattenedDataList.add(value);
                }
            }
        }

        // Stats
        double[] flattenedData = flattenedDataList.toArray();
        lowerValue = (float) StatUtils.percentile(flattenedData, 5);
        upperValue = (float) StatUtils.percentile(flattenedData, 95);
        System.out.println(lowerValue + "  " + upperValue);

    }


    public RealMatrix getMatrix() {
        return matrix;
    }
}
