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

import juicebox.data.Matrix;
import juicebox.track.Feature.Feature2D;

import java.awt.*;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by muhammadsaadshamim on 6/2/15.
 */
public class HiCCUPSUtils {

    private static String OBSERVED = "observed";
    private static String PEAK = "peak";

    private static String EXPECTEDBL = "expectedBL";
    private static String EXPECTEDDONUT = "expectedDonut";
    private static String EXPECTEDH = "expectedH";
    private static String EXPECTEDV = "expectedV";

    private static String BINBL = "binBL";
    private static String BINDONUT = "binDonut";
    private static String BINH = "binH";
    private static String BINV = "binV";

    private static String FDRBL = "fdrBL";
    private static String FDRDONUT = "fdrDonut";
    private static String FDRH = "fdrH";
    private static String FDRV = "fdrV";

    /**
     * Generate a Feature2D peak for a possible peak location from HiCCUPS
     * @param chrName
     * @param observed
     * @param peak
     * @param rowPos
     * @param colPos
     * @param expectedBL
     * @param expectedDonut
     * @param expectedH
     * @param expectedV
     * @param binBL
     * @param binDonut
     * @param binH
     * @param binV
     * @return feature
     */
    public static Feature2D generatePeak(String chrName, float observed, float peak, int rowPos, int colPos,
                                         float expectedBL, float expectedDonut, float expectedH, float expectedV,
                                         float binBL, float binDonut, float binH, float binV) {

        Map<String, String> attributes = new HashMap<String, String>();

        attributes.put(OBSERVED, String.valueOf(observed));
        attributes.put(PEAK, String.valueOf(peak));

        attributes.put(EXPECTEDBL, String.valueOf(expectedBL));
        attributes.put(EXPECTEDDONUT, String.valueOf(expectedDonut));
        attributes.put(EXPECTEDH, String.valueOf(expectedH));
        attributes.put(EXPECTEDV, String.valueOf(expectedV));

        attributes.put(BINBL, String.valueOf(binBL));
        attributes.put(BINDONUT, String.valueOf(binDonut));
        attributes.put(BINH, String.valueOf(binH));
        attributes.put(BINV, String.valueOf(binV));

        int pos1 = Math.min(rowPos, colPos);
        int pos2 = Math.max(rowPos, colPos);

        return new Feature2D(Feature2D.peak, chrName, pos1, pos1+1, chrName, pos2, pos2 + 1, Color.black, attributes);
    }

    /**
     * Calculate fdr values for a given peak
     * @param feature
     * @param fdrLogBL
     * @param fdrLogDonut
     * @param fdrLogH
     * @param fdrLogV
     */
    public static void calculateFDR(Feature2D feature, float[][] fdrLogBL, float[][] fdrLogDonut, float[][] fdrLogH, float[][] fdrLogV) {

        int observed = (int) feature.getFloatAttribute(OBSERVED);
        int binBL = (int) feature.getFloatAttribute(BINBL);
        int binDonut = (int) feature.getFloatAttribute(BINDONUT);
        int binH = (int) feature.getFloatAttribute(BINH);
        int binV = (int) feature.getFloatAttribute(BINV);

        if(binBL >= 0  && binDonut >= 0  && binH >= 0  && binV >= 0  && observed >= 0) {
            feature.addFeature(FDRBL, String.valueOf(fdrLogBL[binBL][observed]));
            feature.addFeature(FDRDONUT, String.valueOf(fdrLogDonut[binDonut][observed]));
            feature.addFeature(FDRH, String.valueOf(fdrLogH[binH][observed]));
            feature.addFeature(FDRV, String.valueOf(fdrLogV[binV][observed]));
        }
        else{
            System.out.println("Error in calculateFDR binBL=" + binBL + " binDonut=" + binDonut +" binH=" + binH +
                    " binV="+ binV + " observed="+ observed);
        }

    }
}
