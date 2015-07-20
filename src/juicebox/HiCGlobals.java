/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2014 Broad Institute, Aiden Lab
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

package juicebox;

import java.awt.*;

/**
 * @author Muhammad Shamim
 * @date 11/25/14
 */
public class HiCGlobals {

    // whether MatrixZoomData should cache or not
    public static final boolean useCache = true;

    // Changes Data Output Mode
    public static final boolean isRestricted = true;

    // Enable black border
    public static final boolean isBlackBorderActivated = false;

    // Juicebox version (for display purposes only)
    private static double versionNum = 1.1;

    // Juicebox title
    public static String juiceboxTitle = "[Juicebox "+versionNum+"] Hi-C Map: ";

    // for plotting
    public static String topChromosomeColor = "#0000FF";
    public static String leftChromosomeColor = "#009900";

    public static final Color backgroundColor = new  Color(204,204,204);

    public static final String stateFileName = "CurrentJuiceboxStates";
    public static final String xmlFileName = "JuiceboxStatesForExport.xml";
    // Feature2D hover text
    public static boolean allowSpacingBetweenFeatureText = true;

    public static void verifySupportedHiCFileVersion(int version) throws RuntimeException {
        if (version < 5) {
            throw new RuntimeException("This file is version " + version +
                    ". Only versions 5 and greater are supported at this time.");
        }
    }
}
