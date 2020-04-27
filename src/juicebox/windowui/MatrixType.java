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
    OEV2("Log(Observed/Expected)"),
    OEP1("(Observed+1)/(Expected+1)"),
    OEP1V2("Log((Observed+1)/(Expected+1))"),
    OME("Observed-Expected"),
    PEARSON("Observed Pearson"),
    LOG("Log(Observed+1)"),
    LOGEO("Log(Observed+1)/Log(Expected+1)"),
    METALOGEO("(Log(Observed+1)+1)/(Log(Expected+1)+1)"),
    EXPLOGEO("e^(Log(Observed+1)/Log(Expected+1))"),
    EXPLOGEOINV("e^(Log(Expected+1)/Log(Observed+1))"),
    NORM2("Observed Norm^2"),
    CONTROL("Control"),
    OECTRL("Control/ExpectedC"),
    OECTRLP1("(Control+1)/(ExpectedC+1)"),
    CME("Control-ExpectedC"),
    PEARSONCTRL("Control Pearson"),
    LOGC("Log(Control+1)"),
    LOGCEO("Log(Control+1)/Log(ExpectedC+1)"),
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
    LOGVS("Log(Observed/AvgO+1) vs Log(Control/AvgC+1)"),
    LOGEOVS("Log(Observed+1)/Log(Expected+1) vs Log(Control+1)/Log(ExpectedC+1)"),
    LOGRATIO("Log(Observed/AvgO+1)/Log(Control/AvgC+1)"),
    LOGEORATIO("(Log(Observed+1)/Log(Expected+1)) / (Log(Control+1)/Log(ExpectedC+1))"),
    DIFF("Observed-Control"),
    NORM("Norm"),
    EIGENVECTOR("Eigenvector"),
    NORM2OBSVSCTRL("Observed Norm^2 vs Control Norm^2");

    public static final MatrixType[] enabledMatrixTypesWithControl = new MatrixType[]{
            OBSERVED, EXPECTED, OE, OEP1, OME, PEARSON, LOG, LOGEO,
            CONTROL, OECTRL, OECTRLP1, CME, PEARSONCTRL, LOGC, LOGCEO,
            VS, RATIO, RATIOP1, RATIO0, RATIO0P1,
            OERATIO, OERATIOP1, OERATIOMINUS, OERATIOMINUSP1,
            OEVS, OEVSP1, OCMEVS, PEARSONVS, DIFF,
            LOGVS, LOGEOVS, LOGRATIO, LOGEORATIO, // todo LOGMINUS, LOGEOMINUS
    };

    public static final MatrixType[] enabledMatrixTypesNoControl =
            new MatrixType[]{OBSERVED, EXPECTED, OE, OEV2, OEP1, OME, PEARSON, LOG, LOGEO, METALOGEO, EXPLOGEO,
                    EXPLOGEOINV};

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
    public static boolean isSimpleColorscaleType(MatrixType option) {
        return isSimpleObservedOrControlType(option) || option == EXPECTED || option == NORM2
                || option == NORM2CTRL || option == NORM2OBSVSCTRL
                || option == LOGEO || option == METALOGEO
                || option == LOGCEO || option == LOGEOVS || option == EXPLOGEO
                || option == EXPLOGEOINV || option == OE;
    }

    /**
     * @param option
     * @return true is the option is generally available for maps, but does not use expected vector
     */
    public static boolean isSimpleObservedOrControlType(MatrixType option) {
        return option == OBSERVED || option == CONTROL || option == VS
                || option == LOG || option == LOGC || option == LOGVS;
    }

    /**
     * @param option
     * @return true is the option can be manipulated by the color range slider
     */
    public static boolean isColorScaleType(MatrixType option) {
        return isOEColorScaleType(option) || option == NORM2
                || option == NORM2CTRL || option == NORM2OBSVSCTRL || isSimpleColorscaleType(option);
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
                || option == RATIO0 || option == RATIO0P1 || option == LOGC || option == LOGVS || option == LOGRATIO;
    }


    /**
     * @param option
     * @return true if the option involves comparison/divis (but not pearsons)
     */
    public static boolean isOEColorScaleType(MatrixType option) {
        return option == OEV2 || option == OEP1V2 || option == RATIO || option == RATIOP1 || option == RATIO0
                || option == RATIO0P1 || option == OECTRL || option == OECTRLP1
                || option == OEVS || option == OEVSP1 || option == OERATIO || option == OERATIOP1
                || option == LOGRATIO || option == LOGEORATIO
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
        return option == OE || option == OEV2 || option == OEP1 || option == LOGEO || option == LOGCEO || isPearsonType(option) || isControlExpectedUsedType(option)
                || option == OCMEVS || option == OME || option == CME || option == METALOGEO;
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
                || option == MatrixType.OEVSP1 || option == OCMEVS || option == LOGVS || option == LOGEOVS;
    }

    public static boolean isControlPearsonType(MatrixType option) {
        return option == PEARSONVS || option == PEARSONCTRL;
    }

    private static boolean isControlExpectedUsedType(MatrixType option) {
        return option == OECTRL || option == OECTRLP1 || option == OEVS || option == OEVSP1 || option == OCMEVS
                || option == CME || option == OERATIO || option == OERATIOP1 || option == OERATIOMINUS
                || option == OERATIOMINUSP1 || option == LOGCEO || option == LOGEORATIO || option == LOGEOVS;
    }

    public static boolean isOnlyControlType(MatrixType option) {
        return option == CONTROL || option == OECTRL || option == OECTRLP1 || option == PEARSONCTRL;
    }

    public String toString() {
        return value;
    }
}
