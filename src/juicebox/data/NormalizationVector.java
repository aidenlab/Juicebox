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

import juicebox.HiC;
import juicebox.data.basics.Chromosome;
import juicebox.data.basics.ListOfDoubleArrays;
import juicebox.data.basics.ListOfFloatArrays;
import juicebox.data.iterator.IteratorContainer;
import juicebox.tools.utils.norm.ZeroScale;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;

/**
 * @author jrobinso
 *         Date: 2/10/13
 *         Time: 9:19 AM
 */
public class NormalizationVector {

    private final NormalizationType type;
    private final int chrIdx;
    private final HiC.Unit unit;
    private final int resolution;
    private final ListOfDoubleArrays data;
    private boolean needsToBeScaledTo = false;
    
    public NormalizationVector(NormalizationType type, int chrIdx, HiC.Unit unit, int resolution, ListOfDoubleArrays data) {
        this.type = type;
        this.chrIdx = chrIdx;
        this.unit = unit;
        this.resolution = resolution;
        this.data = data;
    }
    
    public NormalizationVector(NormalizationType type, int chrIdx, HiC.Unit unit, int resolution, ListOfDoubleArrays data, boolean needsToBeScaledTo) {
        this(type, chrIdx, unit, resolution, data);
        this.needsToBeScaledTo = needsToBeScaledTo;
    }

    public static String getKey(NormalizationType type, int chrIdx, String unit, int resolution) {
        return type + "_" + chrIdx + "_" + unit + "_" + resolution;
    }

    public int getChrIdx() {
        return chrIdx;
    }

    public int getResolution() {
        return resolution;
    }

    public String getKey() {
        return NormalizationVector.getKey(type, chrIdx, unit.toString(), resolution);
    }
    
    public ListOfDoubleArrays getData() {
        return data;
    }

    public boolean doesItNeedToBeScaledTo() {
        return needsToBeScaledTo;
    }

    public NormalizationVector mmbaScaleToVector(Dataset ds) {
        Chromosome chromosome = ds.getChromosomeHandler().getChromosomeFromIndex(chrIdx);
        MatrixZoomData zd = HiCFileTools.getMatrixZoomData(ds, chromosome, chromosome, new HiCZoom(unit, resolution));
        if (zd == null) return null;
        return mmbaScaleToVector(zd.getIteratorContainer());
    }

    public NormalizationVector mmbaScaleToVector(IteratorContainer ic) {

        ListOfFloatArrays newNormVector = ZeroScale.scale(ic, data.convertToFloats(), getKey());
        if (newNormVector != null) {
            newNormVector = ZeroScale.normalizeVectorByScaleFactor(newNormVector, ic);
        }
        ListOfDoubleArrays newDoubleNormVector = newNormVector.convertToDoubles();
        return new NormalizationVector(type, chrIdx, unit, resolution, newDoubleNormVector);
    }
}
