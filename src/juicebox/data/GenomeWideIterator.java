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

package juicebox.data;

import juicebox.data.basics.Chromosome;
import juicebox.windowui.HiCZoom;

import java.util.Iterator;

public class GenomeWideIterator implements Iterator<ContactRecord> {

    private final Chromosome[] chromosomes;
    private final boolean includeIntra;
    private final HiCZoom zoom;
    private final Dataset dataset;
    private final boolean iterationIsDone = false;
    private Iterator<ContactRecord> currentIterator = null;

    private int recentAddX = 0;
    private int recentAddY = 0;
    private int c1i = 0, c2i = 0;

    public GenomeWideIterator(Dataset dataset, ChromosomeHandler handler,
                              HiCZoom zoom, boolean includeIntra) {
        this.chromosomes = handler.getChromosomeArrayWithoutAllByAll();
        this.includeIntra = includeIntra;
        this.zoom = zoom;
        this.dataset = dataset;
        getNextIterator();
    }

    @Override
    public boolean hasNext() {
        if (currentIterator.hasNext()) {
            return true;
        } else {
            recentAddY += chromosomes[c2i].getLength() / zoom.getBinSize() + 1;
            c2i++;
        }
        return getNextIterator();
    }

    private boolean getNextIterator() {
        while (c1i < chromosomes.length) {
            Chromosome c1 = chromosomes[c1i];
            while (c2i < chromosomes.length) {
                Chromosome c2 = chromosomes[c2i];

                if (c1.getIndex() < c2.getIndex() || (c1.equals(c2) && includeIntra)) {
                    MatrixZoomData zd = HiCFileTools.getMatrixZoomData(dataset, c1, c2, zoom);
                    if (zd != null) {
                        currentIterator = zd.getIteratorContainer().getNewContactRecordIterator();
                        if (currentIterator != null && currentIterator.hasNext()) {
                            return true;
                        }
                    }
                }
                recentAddY += c2.getLength() / zoom.getBinSize() + 1;
                c2i++;
            }
            recentAddX += c1.getLength() / zoom.getBinSize() + 1;
            recentAddY = 0;
            c1i++;
            c2i = 0;
        }
        return false;
    }

    @Override
    public ContactRecord next() {
        ContactRecord cr = currentIterator.next();
        int binX = cr.getBinX() + recentAddX;
        int binY = cr.getBinY() + recentAddY;
        return new ContactRecord(binX, binY, cr.getCounts());
    }
}
