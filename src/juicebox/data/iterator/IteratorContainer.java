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

package juicebox.data.iterator;

import juicebox.data.ContactRecord;
import juicebox.data.basics.ListOfDoubleArrays;
import juicebox.data.basics.ListOfFloatArrays;

import java.util.Iterator;

public abstract class IteratorContainer {

    private final long matrixSize;
    private long numberOfContactRecords = -1;
    public static int numCPUMatrixThreads = 10;

    public IteratorContainer(long matrixSize) {
        this.matrixSize = matrixSize;
    }

    abstract public Iterator<ContactRecord> getNewContactRecordIterator();

    protected void setNumberOfContactRecords(long numberOfContactRecords) {
        this.numberOfContactRecords = numberOfContactRecords;
    }

    public long getNumberOfContactRecords() {
        if (numberOfContactRecords > 0) return numberOfContactRecords;

        numberOfContactRecords = 0;
        Iterator<ContactRecord> iterator = getNewContactRecordIterator();
        while (iterator.hasNext()) {
            iterator.next();
            numberOfContactRecords++;
        }

        return numberOfContactRecords;
    }

    public long getMatrixSize() {
        return matrixSize;
    }

    public boolean getIsThereEnoughMemoryForNormCalculation() {
        // when using an iterator, we basically only worry
        // about the vector of row sums
        // float is 4 bytes; one for each row
        return matrixSize * 4 < Runtime.getRuntime().maxMemory();
    }

    public abstract ListOfFloatArrays sparseMultiply(ListOfFloatArrays vector, long vectorLength);

    public abstract void clear();

    protected static ListOfFloatArrays[] getArrayOfFloatVectors(int size, long vectorLength) {
        ListOfFloatArrays[] array = new ListOfFloatArrays[size];
        for (int i = 0; i < size; i++) {
            array[i] = new ListOfFloatArrays(vectorLength);
        }
        return array;
    }

    protected static ListOfDoubleArrays[] getArrayOfDoubleVectors(int size, long vectorLength) {
        ListOfDoubleArrays[] array = new ListOfDoubleArrays[size];
        for (int i = 0; i < size; i++) {
            array[i] = new ListOfDoubleArrays(vectorLength);
        }
        return array;
    }
}
