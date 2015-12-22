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


public enum MatrixType {
    OBSERVED("Observed"),
    OE("O/E"),
    PEARSON("Pearson"),
    EXPECTED("Expected"),
    RATIO("Observed/Control"),
    CONTROL("Control");
    private final String value;

    MatrixType(String value) {
        this.value = value;
    }

    public static MatrixType enumValueFromString(String text) {
        if (text != null) {
            for (MatrixType matrix : MatrixType.values()) {
                if (text.equalsIgnoreCase(matrix.value)) {
                    return matrix;
                }
            }
        }
        return null;
    }

    public static boolean isSimpleType(MatrixType displayOption) {
        return displayOption == MatrixType.OBSERVED || displayOption == MatrixType.EXPECTED || displayOption == MatrixType.CONTROL;
    }

    public static boolean isComparisonType(MatrixType displayOption) {
        return displayOption == MatrixType.OE || displayOption == MatrixType.RATIO;
    }

    public String toString() {
        return value;
    }


}
