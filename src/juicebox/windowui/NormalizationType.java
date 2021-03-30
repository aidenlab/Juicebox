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

package juicebox.windowui;


import java.util.Objects;

/**
 * @author jrobinso Date: 8/31/13  9:47 PM
 */
public class NormalizationType {
	//LOADED("Loaded");
	private final String label;
	private final String description;
	
	public NormalizationType(String label, String description) {
        this.label = label.toUpperCase();
        String description2 = description;
        if (label.endsWith("SCALE")) {
            description2 = description2.replaceAll("Fast scaling", "Balanced++");
            description2 = description2.replaceAll("fast scaling", "Balanced++");
        }

        this.description = description2;
    }

    public String getLabel() {
        return label;
    }

    public String getDescription() {
        return description;
    }

    @Override
    public String toString() {
        return label;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        } else if (obj instanceof NormalizationType) {
            NormalizationType norm2 = (NormalizationType) obj;
            return label.equals(norm2.getLabel());
        }
        return false;
    }

    @Override
    public int hashCode() {
        return Objects.hash(label.hashCode(), description.hashCode());
    }

    public boolean usesKR() {
        return label.contains("KR");
    }

    public boolean usesVC() {
        return label.contains("VC");
    }

    public boolean usesSCALE() {
        return label.contains("SCALE");
    }

    public boolean isNONE() {
        return label.equals("NONE");
    }
}
