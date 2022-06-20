/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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

package juicebox.tools.utils.original;

import java.awt.*;
import java.util.HashMap;
import java.util.Map;

/**
 * Representation of a sparse matrix block used for preprocessing.
 */
class BlockPP {

    private final int number;

    // Key to the map is a Point representing the x,y coordinate for the cell.
    private final Map<Point, ContactCount> contactRecordMap;


    BlockPP(int number) {
        this.number = number;
        this.contactRecordMap = new HashMap<>();
    }

    BlockPP(int number, Map<Point, ContactCount> contactRecordMap) {
        this.number = number;
        this.contactRecordMap = contactRecordMap;
    }


    int getNumber() {
        return number;
    }

    int getNumRecords() {return contactRecordMap.size();}

    void incrementCount(int col, int row, float score) {
        Point p = new Point(col, row);
        ContactCount rec = contactRecordMap.get(p);
        if (rec == null) {
            rec = new ContactCount(score);
            contactRecordMap.put(p, rec);

        } else {
            rec.incrementCount(score);
        }
    }

    Map<Point, ContactCount> getContactRecordMap() {
        return contactRecordMap;
    }

    void merge(BlockPP other) {

        for (Map.Entry<Point, ContactCount> entry : other.getContactRecordMap().entrySet()) {

            Point point = entry.getKey();
            ContactCount otherValue = entry.getValue();

            ContactCount value = contactRecordMap.get(point);
            if (value == null) {
                contactRecordMap.put(point, otherValue);
            } else {
                value.incrementCount(otherValue.getCounts());
            }

        }
    }

    void clear() {
        contactRecordMap.clear();
    }
}
