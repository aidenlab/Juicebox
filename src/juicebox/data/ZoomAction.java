/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2017 Broad Institute, Aiden Lab
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

package juicebox.data;

import juicebox.HiC;
import juicebox.windowui.HiCZoom;

/**
 * Created by ranganmostofa on 7/8/17.
 */
public class ZoomAction {
    private String chromosomeX, chromosomeY;
    private HiCZoom hiCZoom;
    private int genomeX, genomeY;
    private double scaleFactor;
    private boolean resetZoom;
    private HiC.ZoomCallType zoomCallType;
    private int resolutionLocked;

    public ZoomAction(String chromosomeX, String chromosomeY, HiCZoom hiCZoom, int genomeX, int genomeY,
                      double scaleFactor, boolean resetZoom, HiC.ZoomCallType zoomCallType, int resolutionLocked) {
        this.chromosomeX = chromosomeX;
        this.chromosomeY = chromosomeY;
        this.hiCZoom = hiCZoom;
        this.genomeX = genomeX;
        this.genomeY = genomeY;
        this.scaleFactor = scaleFactor;
        this.resetZoom = resetZoom;
        this.zoomCallType = zoomCallType;
        this.resolutionLocked = resolutionLocked;
    }

    public boolean equals(ZoomAction other) {
        if (sameObject(other)) return true;
        if (other != null) {
            return chromosomeX.equals(other.getChromosomeX()) && chromosomeY.equals(other.getChromosomeY()) &&
                    hiCZoom.equals(other.getHiCZoom()) && genomeX == other.getGenomeX() && genomeY == other.getGenomeY()
                    && scaleFactor == other.getScaleFactor() && resetZoom == other.getResetZoom() &&
                    zoomCallType == other.getZoomCallType() && resolutionLocked == other.getResolutionLocked();
        }
        return false;
    }

    public boolean sameObject(ZoomAction other) {
        return this == other;
    }

    public ZoomAction deepCopy() {
        return new ZoomAction(chromosomeX, chromosomeY, hiCZoom.clone(), genomeX, genomeY, scaleFactor,
                resetZoom, zoomCallType, resolutionLocked);
    }

    public String getChromosomeX() {
        return this.chromosomeX;
    }

    public String getChromosomeY() {
        return this.chromosomeY;
    }

    public HiCZoom getHiCZoom() {
        return this.hiCZoom;
    }

    public int getGenomeX() {
        return this.genomeX;
    }

    public int getGenomeY() {
        return this.genomeY;
    }

    public double getScaleFactor() {
        return this.scaleFactor;
    }

    public void setScaleFactor(double newScaleFactor) {
        this.scaleFactor = newScaleFactor;
    }

    public boolean getResetZoom() {
        return this.resetZoom;
    }

    public HiC.ZoomCallType getZoomCallType() {
        return this.zoomCallType;
    }

    public int getResolutionLocked() {
        return this.resolutionLocked;
    }

}
