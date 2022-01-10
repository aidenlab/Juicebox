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

import juicebox.data.ChromosomeHandler;
import juicebox.data.anchor.GenericLocus;
import juicebox.data.anchor.MotifAnchor;
import org.broad.igv.util.Pair;

import java.util.Objects;

public class RegionPair {
	
	public final int xI;
	public final int yI;
	public final GenericLocus xRegion;
	public final GenericLocus xTransRegion;
	public final GenericLocus yRegion;
	public final GenericLocus yTransRegion;

    private RegionPair(int xI, Pair<GenericLocus, GenericLocus> xLocalRegion,
                       int yI, Pair<GenericLocus, GenericLocus> yLocalRegion) {
        this.xI = xI;
        this.yI = yI;
        this.xRegion = xLocalRegion.getFirst();
        this.xTransRegion = xLocalRegion.getSecond();
        this.yRegion = yLocalRegion.getFirst();
        this.yTransRegion = yLocalRegion.getSecond();
    }

    public static RegionPair generateRegionPair(Pair<GenericLocus, GenericLocus> xRegion, Pair<GenericLocus, GenericLocus> yRegion, ChromosomeHandler handler) {
        int xI = handler.getChromosomeFromName(xRegion.getFirst().getChr()).getIndex();
        int yI = handler.getChromosomeFromName(yRegion.getFirst().getChr()).getIndex();

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
	
	public long[] getOriginalGenomeRegion() {
		return new long[]{
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
		return Objects.hash(xI, yI, xRegion.hashCode(), xTransRegion.hashCode(), yRegion.hashCode(), yTransRegion.hashCode());
    }
}
