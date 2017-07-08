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

package juicebox.mapcolorui;

import juicebox.HiC;
import juicebox.windowui.HiCZoom;

/**
 * Created by ranganmostofa on 7/8/17.
 */
public class ZoomState {
    private String chromosomeX;
    private String chromosomeY;
    private HiCZoom hiCZoom;
    private int genomeX;
    private int genomeY;
    private double scaleFactor;
    private boolean resetZoom;
    private HiC.ZoomCallType zoomCallType;
    private boolean allowLocationBroadcast;

    public ZoomState(String chromosomeX, String chromosomeY, HiCZoom hiCZoom, int genomeX, int genomeY,
                     double scaleFactor, boolean resetZoom, HiC.ZoomCallType zoomCallType, boolean allowLocationBroadcast) {
        this.chromosomeX = chromosomeX;
        this.chromosomeY = chromosomeY;
        this.hiCZoom = hiCZoom;
        this.genomeX = genomeX;
        this.genomeY = genomeY;
        this.scaleFactor = scaleFactor;
        this.resetZoom = resetZoom;
        this.zoomCallType = zoomCallType;
        this.allowLocationBroadcast = allowLocationBroadcast;
    }

    public boolean equals(ZoomState other) {
        if (sameObject(other)) return true;
        if (other != null) {
            if (this.chromosomeX.equals(other.getChromosomeX())) {
                if (this.chromosomeY.equals(other.getChromosomeY())) {
                    if (this.hiCZoom.equals(other.getHiCZoom())) {
                        if (this.genomeX == other.getGenomeX()) {
                            if (this.genomeY == other.getGenomeY()) {
                                if (this.scaleFactor == other.getScaleFactor()) {
                                    if (this.resetZoom == other.getResetZoom()) {
                                        if (this.zoomCallType == other.getZoomCallType()) {
                                            if (this.allowLocationBroadcast == other.getAllowLocationBroadcast()) {
                                                return true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return false;
    }

    public boolean sameObject(ZoomState other) {
        return this == other;
    }

    public ZoomState deepCopy() {
        return new ZoomState(chromosomeX, chromosomeY, hiCZoom, genomeX, genomeY, scaleFactor,
                resetZoom, zoomCallType, allowLocationBroadcast);
    }

    private String getChromosomeX() {
        return this.chromosomeX;
    }

    private String getChromosomeY() {
        return this.chromosomeY;
    }

    private HiCZoom getHiCZoom() {
        return this.hiCZoom;
    }

    private int getGenomeX() {
        return this.genomeX;
    }

    private int getGenomeY() {
        return this.genomeY;
    }

    private double getScaleFactor() {
        return this.scaleFactor;
    }

    private boolean getResetZoom() {
        return this.resetZoom;
    }

    private HiC.ZoomCallType getZoomCallType() {
        return this.zoomCallType;
    }

    private boolean getAllowLocationBroadcast() {
        return this.allowLocationBroadcast;
    }
}