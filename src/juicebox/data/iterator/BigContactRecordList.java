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

import java.util.*;

public class BigContactRecordList {

    private static final int MAX_LIMIT = Integer.MAX_VALUE - 10;
    private List<List<ContactRecord>> internalList = new ArrayList<>();
    private long numOfContactRecords = 0;

    public static BigContactRecordList populateListOfListsFromSingleIterator(Iterator<ContactRecord> iterator) {
        BigContactRecordList allRecords = new BigContactRecordList();
        List<ContactRecord> tempList = new ArrayList<>();
        int counter = 0;
        while (iterator.hasNext()) {
            tempList.add(iterator.next());
            counter++;
            if (counter > MAX_LIMIT) {
                allRecords.addSubList(tempList);
                tempList = new ArrayList<>();
                counter = 0;
            }
        }
        if (tempList.size() > 0) {
            allRecords.addSubList(tempList);
        }
        return allRecords;
    }

    public void addAllSubLists(BigContactRecordList other) {
        internalList.addAll(other.internalList);
        for (List<ContactRecord> records : other.internalList) {
            numOfContactRecords += records.size();
        }
    }

    private void addSubList(List<ContactRecord> cList) {
        internalList.add(cList);
        numOfContactRecords += cList.size();
    }

    public long getTotalSize() {
        return numOfContactRecords;
    }

    public int getNumLists() {
        return internalList.size();
    }

    public List<ContactRecord> getSubList(int index) {
        return internalList.get(index);
    }

    public void clear() {
        for (List<ContactRecord> cList : internalList) {
            cList.clear();
        }
        internalList.clear();
        internalList = new ArrayList<>();
        numOfContactRecords = 0;
    }

    public void sort() {
        internalList.sort(Comparator.comparing(o -> o.get(0)));
    }

    public void collapse() {
        System.out.println("Was n " + internalList.size());

        int numFinList = (int) Math.max(numOfContactRecords / 200000000, 20);
        List<List<ContactRecord>> newInternalList = new ArrayList<>();
        int[] countForList = new int[numFinList];
        Arrays.fill(countForList, 0);
        for (int z = 0; z < numFinList; z++) {
            newInternalList.add(new ArrayList<>());
        }

        for (List<ContactRecord> subList : internalList) {
            int whichIndexToAddTo = getIndexOfMin(countForList);
            countForList[whichIndexToAddTo] += subList.size();
            newInternalList.get(whichIndexToAddTo).addAll(subList);
        }

        internalList.clear();
        internalList = newInternalList;

        System.out.println("Now is n " + internalList.size());
    }

    private int getIndexOfMin(int[] counts) {
        int minIndex = 0;
        for (int k = 1; k < counts.length; k++) {
            if (counts[k] < counts[minIndex]) {
                minIndex = k;
            }
        }
        return minIndex;
    }
}
