/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab
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
    EXPECTED("Expected"),
    OE("Observed/Expected"),
    OP1EP1("(Observed+1)/(Expected+1)"),
    OME("Observed-Expected"),
    PEARSON("Observed Pearson"),
    NORM2("Observed Norm^2"),
    CONTROL("Control"),
    OECTRL("Control/ExpectedC"),
    OP1EP1CTRL("(Control+1)/(ExpectedC+1)"),
    CME("Control-ExpectedC"),
    PEARSONCTRL("Control Pearson"),
    NORM2CTRL("Control Norm^2"),
    RATIO("Observed/Control * (AvgC/AvgO)"),
    RATIOP1("(Observed+1)/(Control+1) * (AvgC+1)/(AvgO+1)"),
    RATIO0("Observed/Control * (ExpC0/Exp0)"),
    RATIO0P1("(Observed+1)/(Control+1) * (ExpC0+1)/(Exp0+1)"),
    VS("Observed vs Control"),
    OEVS("Observed/Expected vs Control/ExpectedC"),
    OEVSP1("(Observed+1)/(Expected+1) vs (Control+1)/(ExpectedC+1)"),
    OERATIO("(Observed/Expected) / (Control/ExpectedC)"),
    OERATIOP1("((Observed+1)/(Expected+1)) / ((Control+1)/(ExpectedC+1))"),
    OERATIOMINUS("(Observed/Expected) - (Control/ExpectedC)"),
    OERATIOMINUSP1("(Observed+1)/(Expected+1) - (Control+1)/(ExpectedC+1)"),
    OCMEVS("Observed-Expected vs Control-Expected"),
    PEARSONVS("Observed Pearson vs Control Pearson"),
    DIFF("Observed-Control"),
    NORM("Norm"),
    EIGENVECTOR("Eigenvector"),
    NORM2OBSVSCTRL("Observed Norm^2 vs Control Norm^2");
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
            if (text.equalsIgnoreCase("oe")) {
                return OE;
            }
        }
        return null;
    }

    /**
     * @param option
     * @return true is the option is generally available all maps or resolutions
     */
    public static boolean isObservedOrControl(MatrixType option) {
        return option == OBSERVED || option == CONTROL;
    }

    /**
     * @param option
     * @return true is the option is generally available all maps or resolutions
     */
    public static boolean isSimpleType(MatrixType option) {
        return isSimpleObservedOrControlType(option) || option == EXPECTED || option == NORM2
                || option == NORM2CTRL || option == NORM2OBSVSCTRL;
    }

    /**
     * @param option
     * @return true is the option is generally available for maps, but does not use expected vector
     */
    public static boolean isSimpleObservedOrControlType(MatrixType option) {
        return option == OBSERVED || option == CONTROL || option == VS;
    }

    /**
     * @param option
     * @return true is the option can be manipulated by the color range slider
     */
    public static boolean isColorScaleType(MatrixType option) {
        return isComparisonType(option) || isSimpleObservedOrControlType(option) || option == NORM2
                || option == NORM2CTRL || option == NORM2OBSVSCTRL;
    }



    /**
     * @param option
     * @return true if the option should allowed in genome-wide view
     */
    public static boolean isValidGenomeWideOption(MatrixType option) {
        return option == OBSERVED || isSimpleControlType(option);
    }

    /**
     * @param option
     * @return true if the option requires control map but not expected vector
     */
    public static boolean isSimpleControlType(MatrixType option) {
        return option == CONTROL || option == VS || option == DIFF || option == RATIO || option == RATIOP1
                || option == RATIO0 || option == RATIO0P1;
    }


    /**
     * @param option
     * @return true if the option involves comparison/divis (but not pearsons)
     */
    public static boolean isComparisonType(MatrixType option) {
        return option == OE || option == OP1EP1 || option == RATIO || option == RATIOP1 || option == RATIO0
                || option == RATIO0P1 || option == OECTRL || option == OP1EP1CTRL
                || option == OEVS || option == OEVSP1 || option == OERATIO || option == OERATIOP1
                || isSubtactType(option);
    }

    public static boolean isSubtactType(MatrixType option) {
        return option == DIFF || option == OCMEVS || option == OME || option == CME || option == OERATIOMINUS || option == OERATIOMINUSP1;
    }

    /**
     * @param option
     * @return true if the option only works for intrachromosomal, not interchromosomal (genomewide may still be allowed)
     */
    public static boolean isOnlyIntrachromosomalType(MatrixType option) {
        return isPearsonType(option) || isVSTypeDisplay(option);
    }

    /**
     * @param option
     * @return true if the option requires the expected vector
     */
    public static boolean isExpectedValueType(MatrixType option) {
        return option == OE || option == OP1EP1 || isPearsonType(option) || isControlExpectedUsedType(option)
                || option == OCMEVS || option == OME || option == CME;
    }

    /**
     * @param option
     * @return true if the option uses pearson's
     */
    public static boolean isPearsonType(MatrixType option) {
        return option == PEARSON || isControlPearsonType(option);
    }

    /**
     * @param option
     * @return true if the option is dumped (clt) as a vector
     */
    public static boolean isDumpVectorType(MatrixType option) {
        return option == NORM || option == EXPECTED;
    }

    /**
     * @param option
     * @return true if the option is dumped (clt) as a matrix
     */
    public static boolean isDumpMatrixType(MatrixType option) {
        return option == OE || option == OBSERVED;
    }

    public static boolean isVSTypeDisplay(MatrixType option) {
        return option == MatrixType.VS || option == MatrixType.PEARSONVS || option == MatrixType.OEVS
                || option == MatrixType.OEVSP1 || option == OCMEVS;
    }

    public static boolean isControlPearsonType(MatrixType option) {
        return option == PEARSONVS || option == PEARSONCTRL;
    }

    private static boolean isControlExpectedUsedType(MatrixType option) {
        return option == OECTRL || option == OP1EP1CTRL || option == OEVS || option == OEVSP1 || option == OCMEVS
                || option == CME || option == OERATIO || option == OERATIOP1 || option == OERATIOMINUS || option == OERATIOMINUSP1;
    }

    public static boolean isOnlyControlType(MatrixType option) {
        return option == CONTROL || option == OECTRL || option == OP1EP1CTRL || option == PEARSONCTRL;
    }

    public String toString() {
        return value;
    }
}
