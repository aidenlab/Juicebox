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

package juicebox.data.censoring;

import juicebox.data.anchor.MotifAnchor;
import org.broad.igv.util.Pair;

public class RegionPair {

    public final int xI;
    public final int yI;
    public final MotifAnchor xRegion;
    public final MotifAnchor xTransRegion;
    public final MotifAnchor yRegion;
    public final MotifAnchor yTransRegion;

    private RegionPair(int xI, Pair<MotifAnchor, MotifAnchor> xLocalRegion,
                       int yI, Pair<MotifAnchor, MotifAnchor> yLocalRegion) {
        this.xI = xI;
        this.yI = yI;
        this.xRegion = xLocalRegion.getFirst();
        this.xTransRegion = xLocalRegion.getSecond();
        this.yRegion = yLocalRegion.getFirst();
        this.yTransRegion = yLocalRegion.getSecond();
    }

    public static RegionPair generateRegionPair(Pair<MotifAnchor, MotifAnchor> xRegion, Pair<MotifAnchor, MotifAnchor> yRegion) {
        int xI = xRegion.getFirst().getChr();
        int yI = yRegion.getFirst().getChr();

        // todo debug for diff custom chrs against each other
        //  return new RegionPair(xI, xRegion, yI, yRegion);

        if (xI <= yI) {
            return new RegionPair(xI, xRegion, yI, yRegion);
        } else {
            return new RegionPair(yI, yRegion, xI, xRegion);
        }
    }

    public String getDescription() {
        return "" + xI + "_" + yI + xRegion.toString() + xTransRegion.toString() + yRegion.toString() + yTransRegion.toString();
    }

    public int[] getOriginalGenomeRegion() {
        return new int[]{
                xRegion.getX1(), xRegion.getX2(),
                yRegion.getX1(), yRegion.getX2()};
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj instanceof RegionPair) {
            RegionPair o = (RegionPair) obj;

            return xI == o.xI
                    && yI == o.yI
                    && xRegion.equals(o.xRegion)
                    && xTransRegion.equals(o.xTransRegion)
                    && yRegion.equals(o.yRegion)
                    && yTransRegion.equals(o.yTransRegion);
        }
        return false;
    }

    @Override
    public int hashCode() {
        int hash = 29 * xI + 31 * yI;
        hash *= xRegion.hashCode() + xTransRegion.hashCode();
        hash *= yRegion.hashCode() + yTransRegion.hashCode();
        return hash;
    }
}
