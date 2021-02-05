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

package juicebox.data.censoring;

import java.util.List;

class OneDimSearchUtils {

    /**
     * Searches the specified list for the specified object or its closest
     * counter part using a modified binary search algorithm.  The list must
     * be sorted into ascending order according to the natural ordering of
     * its elements prior to making this call.  If it is not sorted, the
     * results are undefined.  If the list contains multiple elements equal
     * to the specified object, there is no guarantee which one will be found.
     *
     * @param <T>      the class of the objects in the list
     * @param list     the list to be searched.
     * @param key      the key to be searched for.
     * @param useFloor to get the lesser of two indices given no exact match
     * @return the index of the search key, if it is contained in the list;
     */
    public static <T> int indexedBinaryNearestSearch(List<? extends Comparable<? super T>> list, T key, boolean useFloor) {
        int low = 0;
        int high = list.size() - 1;

        if (useFloor && list.get(low).compareTo(key) > 0) {
            return low;
        } else if (!useFloor && list.get(high).compareTo(key) < 0) {
            return high;
        }

        while (low < high) {
            int mid = (low + high) >>> 1;
            int cmp = list.get(mid).compareTo(key);
            int cmpPlus1 = list.get(mid + 1).compareTo(key);

            if (cmpPlus1 < 0) {
                low = Math.min(high, mid + 1);
            } else if (cmp > 0) {
                high = Math.max(low, mid - 1);
            } else if (cmp == 0) {
                return mid;
            } else if (cmpPlus1 == 0) {
                return mid + 1;
            } else if (useFloor) {
                return mid;
            } else {
                return mid + 1;
            }
            if (high == low) {
                return low;
            }
        }
        System.err.println("something went wrong " + low + " " + high + " " + useFloor);
        return -(low);  // key not found
    }
}
