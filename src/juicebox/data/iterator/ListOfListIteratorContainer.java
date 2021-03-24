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

import java.util.Iterator;
import java.util.List;

public class ListOfListIteratorContainer extends IteratorContainer {

    private final List<List<ContactRecord>> allContactRecords;

    public ListOfListIteratorContainer(List<List<ContactRecord>> allContactRecords, long matrixSize,
                                       long totalNumberOfContacts) {
        super(matrixSize);
        setNumberOfContactRecords(totalNumberOfContacts);
        this.allContactRecords = allContactRecords;
    }

    @Override
    public Iterator<ContactRecord> getNewContactRecordIterator() {
        return new ListOfListIterator(allContactRecords);
    }

    @Override
    public boolean getIsThereEnoughMemoryForNormCalculation() {
        // float is 4 bytes; one for each row (row sums)
        // 12 bytes (2 ints, 1 float) for contact record
        return 4 * getMatrixSize() + 12 * getNumberOfContactRecords() < Runtime.getRuntime().maxMemory();
    }
}
