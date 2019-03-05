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

package juicebox.windowui;

import java.util.ArrayList;
import java.util.List;

public class NormalizationHandler {
    public static final NormalizationType NONE = new NormalizationType("NONE", "None");
    public static final NormalizationType VC = new NormalizationType("VC", "Coverage");
    public static final NormalizationType VC_SQRT = new NormalizationType("VC_SQRT", "Coverage (Sqrt)");
    public static final NormalizationType KR = new NormalizationType("KR", "Balanced");
    public static final NormalizationType GW_KR = new NormalizationType("GW_KR", "Genome-wide balanced");
    public static final NormalizationType INTER_KR = new NormalizationType("INTER_KR", "Inter balanced");
    public static final NormalizationType GW_VC = new NormalizationType("GW_VC", "Genome-wide coverage");
    public static final NormalizationType INTER_VC = new NormalizationType("INTER_VC", "Inter coverage");//,

    private final static List<NormalizationType> currentlyAvailableNorms = new ArrayList<>();

    public NormalizationHandler() {
        currentlyAvailableNorms.add(NONE);
        currentlyAvailableNorms.add(KR);
        currentlyAvailableNorms.add(VC);
        currentlyAvailableNorms.add(VC_SQRT);
        currentlyAvailableNorms.add(GW_KR);
        currentlyAvailableNorms.add(GW_VC);
        currentlyAvailableNorms.add(INTER_KR);
        currentlyAvailableNorms.add(INTER_VC);
    }

    public static List<NormalizationType> getAllNormTypes() {
        return currentlyAvailableNorms;
    }

    public NormalizationType getNormTypeFromString(String text) {
        if (text != null && text.length() > 0) {
            for (NormalizationType norm : currentlyAvailableNorms) {
                if (text.equalsIgnoreCase(norm.getLabel()) || text.equalsIgnoreCase(norm.getDescription())) {
                    return norm;
                }
            }
        }
        NormalizationType newNormType = new NormalizationType(text, text);
        currentlyAvailableNorms.add(newNormType);
        return newNormType;
    }

    public NormalizationType getNormTypeFromString(String text, String description) {
        if (text != null && text.length() > 0) {
            for (NormalizationType norm : currentlyAvailableNorms) {
                if (text.equalsIgnoreCase(norm.getLabel()) || text.equalsIgnoreCase(norm.getDescription()) || description.equalsIgnoreCase(norm.getDescription())) {
                    return norm;
                }
            }
        }
        NormalizationType newNormType = new NormalizationType(text, description);
        currentlyAvailableNorms.add(newNormType);
        return newNormType;
    }
}
