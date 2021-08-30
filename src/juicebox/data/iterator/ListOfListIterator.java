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

public class ListOfListIterator implements Iterator<ContactRecord> {

    private final BigContactRecordList allContactRecords;
    private Iterator<ContactRecord> currentIterator = null;
    private int currentListIndex = 0;

    public ListOfListIterator(BigContactRecordList allContactRecords) {
        this.allContactRecords = allContactRecords;
        getNextIterator();
    }

    @Override
    public boolean hasNext() {
        if (currentIterator.hasNext()) {
            return true;
        } else {
            currentListIndex++;
        }
        return getNextIterator();
    }

    private boolean getNextIterator() {
        while (currentListIndex < allContactRecords.getNumLists()) {
            currentIterator = allContactRecords.getSubList(currentListIndex).iterator();
            if (currentIterator.hasNext()) {
                return true;
            }
            currentListIndex++;
        }
        return false;
    }

    @Override
    public ContactRecord next() {
        return currentIterator.next();
    }
}
