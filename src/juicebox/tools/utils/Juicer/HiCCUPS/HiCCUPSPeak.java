/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2015 Broad Institute, Aiden Lab
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

package juicebox.tools.utils.Juicer.HiCCUPS;

/**
 * Created by muhammadsaadshamim on 5/12/15.
 */
public class HiCCUPSPeak {

    private float observed;
    private float peak;
    private int rowPosition;
    private int columnPosition;
    private float expectedBL;
    private float expectedDonut;
    private float expectedH;
    private float expectedV;
    private float binBL;
    private float binDonut;
    private float binH;
    private float binV;
    private float fdrBL;
    private float fdrDonut;
    private float fdrH;
    private float fdrV;

    private String chrName;

    public HiCCUPSPeak(float observedVal, float peakVal, int rowPosition, int columnPosition,
                       float expectedBLVal, float expectedDonutVal, float expectedHVal, float expectedVVal,
                       float binBLVal, float binDonutVal, float binHVal, float binVVal) {
        this.chrName = chrName;
        this.observed = observedVal;
        this.peak = peakVal;
        this.rowPosition = rowPosition;
        this.columnPosition = columnPosition;
        this.expectedBL = expectedBLVal;
        this.expectedDonut = expectedDonutVal;
        this.expectedH = expectedHVal;
        this.expectedV = expectedVVal;
        this.binBL = binBLVal;
        this.binDonut = binDonutVal;
        this.binH = binHVal;
        this.binV = binVVal;
    }

    public void calculateFDR(float[][] fdrLogBL, float[][] fdrLogDonut, float[][] fdrLogH, float[][] fdrLogV) {
        fdrBL = fdrLogBL[(int)binBL][(int)observed];
        fdrDonut = fdrLogDonut[(int)binDonut][(int)observed];
        fdrH = fdrLogH[(int)binH][(int)observed];
        fdrV = fdrLogV[(int)binV][(int)observed];
    }

    @Override
    public String toString(){
        return chrName + "\t" + rowPosition + "\t" + chrName + "\t" + columnPosition + "\t" + observed + "\t" +
                expectedBL + "\t" + expectedDonut + "\t" + expectedH + "\t" + expectedV + "\t" +
                binBL + "\t" + binDonut + "\t" + binH + "\t" + binV + "\t" +
                fdrBL + "\t" + fdrDonut + "\t" + fdrH + "\t" + fdrV;
    }


}
