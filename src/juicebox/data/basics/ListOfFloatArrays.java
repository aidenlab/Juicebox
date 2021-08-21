/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2021 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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

package juicebox.data.basics;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * can't use <T> because we need to instantiate the array, otherwise that would have been nice
 */
public class ListOfFloatArrays {

    private final long DEFAULT_LENGTH = Integer.MAX_VALUE - 10;
    private final long overallLength;
    private final List<float[]> internalList = new ArrayList<>();

    public ListOfFloatArrays(long length) {
        this.overallLength = length;
        long tempLength = length;
        while (tempLength > 0) {
            if (tempLength < DEFAULT_LENGTH) {
                internalList.add(new float[(int) tempLength]);
                break;
            } else {
                internalList.add(new float[(int) DEFAULT_LENGTH]);
                tempLength -= DEFAULT_LENGTH;
            }
        }
    }

    public ListOfFloatArrays(long totSize, float defaultValue) {
        this(totSize);
        for (float[] array : internalList) {
            Arrays.fill(array, defaultValue);
        }
    }

    public void clear() {
        internalList.clear();
    }

    public float get(long index) {
        if (index < overallLength) {
            int pseudoRow = (int) (index / DEFAULT_LENGTH);
            int pseudoCol = (int) (index % DEFAULT_LENGTH);
            return internalList.get(pseudoRow)[pseudoCol];
        } else {
            System.err.println("long index exceeds max size of list of arrays while getting: " + index + " " + overallLength);
            Exception ioe = new Exception();
            ioe.printStackTrace();
            return Float.NaN;
        }
    }

    public void set(long index, float value) {
        if (index < overallLength) {
            int pseudoRow = (int) (index / DEFAULT_LENGTH);
            int pseudoCol = (int) (index % DEFAULT_LENGTH);
            internalList.get(pseudoRow)[pseudoCol] = value;
        } else {
            System.err.println("long index exceeds max size of list of arrays while setting");
        }
    }

    public long getLength() {
        return overallLength;
    }

    public long getMaxRow() {
        long maxIndex = 0;
        float maxVal = 0;
        for (int index = 0; index < overallLength; index++) {
            int pseudoRow = (int) (index / DEFAULT_LENGTH);
            int pseudoCol = (int) (index % DEFAULT_LENGTH);
            if (maxVal < internalList.get(pseudoRow)[pseudoCol]) {
                maxVal = internalList.get(pseudoRow)[pseudoCol];
                maxIndex = index;
            }
        }
        return maxIndex;
    }

    public ListOfFloatArrays deepClone() {
        ListOfFloatArrays clone = new ListOfFloatArrays(overallLength);
        for (int k = 0; k < internalList.size(); k++) {
            System.arraycopy(internalList.get(k), 0, clone.internalList.get(k), 0, internalList.get(k).length);
        }
        return clone;
    }

    public void divideBy(long index, float value) {
        if (index < overallLength) {
            int pseudoRow = (int) (index / DEFAULT_LENGTH);
            int pseudoCol = (int) (index % DEFAULT_LENGTH);
            internalList.get(pseudoRow)[pseudoCol] /= value;
        } else {
            System.err.println("long index exceeds max size of list of arrays while dividing");
        }
    }

    public void multiplyBy(long index, float value) {
        if (index < overallLength) {
            int pseudoRow = (int) (index / DEFAULT_LENGTH);
            int pseudoCol = (int) (index % DEFAULT_LENGTH);
            internalList.get(pseudoRow)[pseudoCol] *= value;
        } else {
            System.err.println("long index exceeds max size of list of arrays while mutiplying");
        }
    }

    public void addTo(long index, float value) {
        if (index < overallLength) {
            int pseudoRow = (int) (index / DEFAULT_LENGTH);
            int pseudoCol = (int) (index % DEFAULT_LENGTH);
            try {
                internalList.get(pseudoRow)[pseudoCol] += value;
            } catch (Exception e) {
                System.err.println(index + " " + pseudoCol);
                e.printStackTrace();
            }
        } else {
            System.err.println("long index exceeds max size of list of arrays while adding: " + index + " " + overallLength);
            Exception ioe = new Exception();
            ioe.printStackTrace();
        }
    }

    public void addValuesFrom(ListOfFloatArrays other) {
        if (overallLength == other.overallLength) {
            for (int i = 0; i < internalList.size(); i++) {
                float[] array = internalList.get(i);
                float[] otherArray = other.internalList.get(i);
                for (int j = 0; j < array.length; j++) {
                    array[j] += otherArray[j];
                }
            }
        } else {
            System.err.println("Adding objects of different sizes!");
        }
    }

    public float getFirstValue() {
        return internalList.get(0)[0];
    }

    public float getLastValue() {
        float[] temp = internalList.get(internalList.size() - 1);
        return temp[temp.length - 1];
    }

    public List<float[]> getValues() {
        return internalList;
    }

    public void multiplyEverythingBy(double val) {
        for (float[] array : internalList) {
            for (int k = 0; k < array.length; k++) {
                array[k] *= val;
            }
        }
    }

    public ListOfDoubleArrays convertToDoubles() {
        ListOfDoubleArrays newList = new ListOfDoubleArrays(overallLength);
        for (int j = 0; j < internalList.size(); j++) {

            float[] array = internalList.get(j);
            double[] newArray = newList.getValues().get(j);

            for (int k = 0; k < array.length; k++) {
                newArray[k] = array[k];
            }
        }
        return newList;
    }
}
