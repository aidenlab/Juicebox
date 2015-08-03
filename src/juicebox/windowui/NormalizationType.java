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

package juicebox.windowui;

/**
 * @author jrobinso Date: 8/31/13  9:47 PM
 */
public enum NormalizationType {
    NONE("None"),
    VC("Coverage"),
    VC_SQRT("Coverage (Sqrt)"),
    KR("Balanced"),
    GW_KR("Genome-wide balanced"),
    INTER_KR("Inter balanced"),
    GW_VC("Genome-wide coverage"),
    INTER_VC("Inter coverage"),
    LOADED("Loaded");
    private final String label;

    NormalizationType(String label) {
        this.label = label;
    }

    public static NormalizationType enumValueFromString(String text) {
        if (text != null) {
            for (NormalizationType norm : NormalizationType.values()) {
                if (text.equalsIgnoreCase(norm.label)) {
                    return norm;
                }
            }
        }
        return null;
    }

    public String getLabel() {
        return label;
    }

}
