/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2019 Broad Institute, Aiden Lab
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

package juicebox.tools.utils.dev;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class LocalGenomeRegion {

    final int initialIndex;
    private List<Neighbor> neighbors = new ArrayList<>();

    public LocalGenomeRegion(int initialIndex) {
        this.initialIndex = initialIndex;
    }

    public void addNeighbor(int y, float counts) {
        neighbors.add(new Neighbor(y, counts));
    }

    public void filterDownValues(int numNeighborsAllowed) {
        Collections.sort(neighbors, Collections.reverseOrder());
        neighbors = neighbors.subList(0, Math.min(numNeighborsAllowed, neighbors.size()));
    }

    public boolean notConnectedWith(int index) {
        for (Neighbor neighbor : neighbors) {
            if (neighbor.index == index) {
                return false;
            }
        }
        return true;
    }

    public int getOutlierContacts(boolean isBadUpstream, int cliqueSize) {

        Collections.sort(neighbors, Collections.reverseOrder());

        for (Neighbor neighbor : neighbors) {
            if (Math.abs(neighbor.index - initialIndex) > cliqueSize) {
                if (isBadUpstream) {
                    if (neighbor.index < initialIndex) {
                        return neighbor.index;
                    }
                } else { // is bad downstream
                    if (neighbor.index > initialIndex) {
                        return neighbor.index;
                    }
                }
            }
        }

        return -1;
    }


    @Override
    public String toString() {
        String nei = "";
        for (Neighbor neighbor : neighbors) {
            nei += neighbor.index + "-" + neighbor.value + "__";
        }

        return initialIndex + " - " + nei;
    }

    private class Neighbor implements Comparable<Neighbor> {
        int index;
        Float value;

        Neighbor(int index, float value) {
            this.index = index;
            this.value = value;
        }

        @Override
        public int compareTo(Neighbor o) {
            return value.compareTo(o.value);
        }
    }
}
