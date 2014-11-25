package juicebox.matrix;


import org.apache.commons.math.stat.StatUtils;
import org.broad.igv.util.collections.DoubleArrayList;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * @author jrobinso
 *         Date: 2/28/14
 *         Time: 5:49 PM
 */
public class SymmetricMatrix implements BasicMatrix {

    final int dim;
    final float[] data;
    float lowerValue = Float.NaN;
    float upperValue = Float.NaN;
    final Set<Integer> nullColumns;

    public SymmetricMatrix(int dim) {
        this.dim = dim;
        int size = (dim * dim - dim) / 2 + dim;
        data = new float[size];
        nullColumns = new HashSet<Integer>();
    }


    public void nullColumn(int i) {
        nullColumns.add(i);
    }

    public Set<Integer> getNullColumns() {
        return nullColumns;
    }

    public void fill(float value) {
        Arrays.fill(data, value);
    }

    public void setEntry(int i, int j, float value) {
        data[getIdx(i, j)] = value;

    }

    int getIdx(int i, int j) {

        return (i < j) ?
                i * dim - (i - 1) * i / 2 + j - i :
                j * dim - (j - 1) * j / 2 + i - j;
    }


    public float getColumnMean(int j) {
        float sum = 0;
        int count = 0;
        for (int i = 0; i < dim; i++) {
            float value = getEntry(i, j);
            if (!Float.isNaN(value)) {
                sum += value;
                count++;
            }
        }
        return count == 0 ? Float.NaN : sum / count;
    }

    public float getRowMean(int i) {
        float sum = 0;
        int count = 0;
        for (int j = 0; j < dim; j++) {
            float value = getEntry(i, j);
            if (!Float.isNaN(value)) {
                sum += value;
                count++;
            }
        }
        return count == 0 ? Float.NaN : sum / count;
    }


    @Override
    public float getEntry(int i, int j) {
        int idx = getIdx(i, j);
        return idx < data.length ? data[idx]  : Float.NaN;
    }

    @Override
    public int getRowDimension() {
        return dim;
    }

    @Override
    public int getColumnDimension() {
        return dim;
    }

    @Override
    public float getLowerValue() {
        if(Float.isNaN(lowerValue)) {
            computePercentiles();
        }
        return lowerValue;
    }

    @Override
    public float getUpperValue() {
        if(Float.isNaN(upperValue)) {
            computePercentiles();
        }
        return upperValue;
    }

    void computePercentiles() {

        // Statistics, other attributes
        DoubleArrayList flattenedDataList = new DoubleArrayList(data.length);

        for (float value : data) {
            if (!Float.isNaN(value) && value != 1) {
                flattenedDataList.add(value);
            }
        }

        // Stats
        double[] flattenedData = flattenedDataList.toArray();
        lowerValue = (float) StatUtils.percentile(flattenedData, 5);
        upperValue = (float) StatUtils.percentile(flattenedData, 95);
    }

}
