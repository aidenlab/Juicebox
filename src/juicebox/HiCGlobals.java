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

package juicebox;

import juicebox.windowui.MatrixType;

import java.awt.*;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Muhammad Shamim
 * @since 11/25/14
 */
public class HiCGlobals {

    // Juicebox version (for display and header purposes only)
    public static final String versionNum = "1.10.10"; //

    // Changes Data Output Mode
    public static final boolean isRestricted = false;
    // Enable black border
    public static final boolean isBlackBorderActivated = false;
    // MainWindow variables
    public static final Color RULER_LINE_COLOR = new Color(0, 0, 230, 100);
    public static final Color DARKULA_RULER_LINE_COLOR = new Color(200, 200, 250, 100);

    public static final int BIN_PIXEL_WIDTH = 1;
    // for plotting
    public static final String topChromosomeColor = "#0000FF";
    public static final String leftChromosomeColor = "#009900";
    public static final Color backgroundColor = new Color(204, 204, 204);
    public static final String BACKUP_FILE_STEM = "unsaved_hic_annotations_backup_";
    // for state saving
    public static File stateFile;
    public static File xmlSavedStatesFile;
    // Feature2D hover text
    public static final boolean allowSpacingBetweenFeatureText = true;
    public static final ArrayList<String> savedStatesList = new ArrayList<>();
    // min hic file version supported
    public static final int minVersion = 6; // todo redundant calls to this should be removed
    public static final int bufferSize = 2097152;

    public static final MatrixType[] enabledMatrixTypesNoControl = new MatrixType[]{
            MatrixType.OBSERVED, MatrixType.EXPECTED, MatrixType.OE, MatrixType.PEARSON};
    public static final MatrixType[] enabledMatrixTypesWithControl = new MatrixType[]{
            MatrixType.OBSERVED, MatrixType.EXPECTED, MatrixType.OE, MatrixType.PEARSON,
            MatrixType.CONTROL, MatrixType.OECTRL, MatrixType.PEARSONCTRL,
            MatrixType.VS, MatrixType.RATIO, MatrixType.OEVS, MatrixType.PEARSONVS, MatrixType.DIFF};
    public static final String defaultPropertiesURL = "http://hicfiles.tc4ga.com/juicebox.properties";
    // Juicebox title
    // TODO decide on title displayed in Juicebox
    public static final String juiceboxTitle = "[Juicebox " + versionNum + "] Hi-C Map ";
    public static Color HIC_MAP_COLOR = Color.RED;
    public static final Color HIGHLIGHT_COLOR = Color.BLACK;
    public static final Color SELECT_FEATURE_COLOR = Color.DARK_GRAY;
    public static int MAX_PEARSON_ZOOM = 500000;
    public static double hicMapScale = 1; //TODO implement Map scaling with this global variable
    // whether MatrixZoomData should cache or not
    public static boolean useCache = true;
    public static boolean guiIsCurrentlyActive = false;
    public static boolean printVerboseComments = false;
    public static boolean slideshowEnabled = false;
    public static boolean splitModeEnabled = false;
    public static boolean translationInProgress = false;
    public static boolean displayTiles = false;
    public static boolean isDarkulaModeEnabled = false;
    public static boolean isAssemblyMatCheck = false;


    // whether instance was linked before mouse press or not
    public static boolean wasLinkedBeforeMousePress = false;
    public static boolean isLegacyOutputPrintingEnabled = false;
    public static final boolean isDevAssemblyToolsAllowedPublic = true;
    public static final boolean isDevCustomChromosomesAllowedPublic = true;
    public static final Color diffGrayColor = new Color(238, 238, 238);

    public static void verifySupportedHiCFileVersion(int version) throws RuntimeException {
        if (version < minVersion) {
            throw new RuntimeException("This file is version " + version +
                    ". Only versions " + minVersion + " and greater are supported at this time.");
        }
    }

    public static List<Color> createNewPreDefMapColorGradient() {
        List<Color> colors = new ArrayList<>();
        colors.add(new Color(255, 242, 255));
        colors.add(new Color(255, 242, 255));
        colors.add(new Color(255, 230, 242));
        colors.add(new Color(255, 222, 230));
        colors.add(new Color(250, 218, 234));
        colors.add(new Color(255, 206, 226));
        colors.add(new Color(238, 198, 210));
        colors.add(new Color(222, 186, 182));
        colors.add(new Color(226, 174, 165));
        colors.add(new Color(214, 157, 145));
        colors.add(new Color(194, 141, 125));
        colors.add(new Color(218, 157, 121));
        colors.add(new Color(234, 182, 129));
        colors.add(new Color(242, 206, 133));
        colors.add(new Color(238, 222, 153));
        colors.add(new Color(242, 238, 161));
        colors.add(new Color(222, 238, 161));
        colors.add(new Color(202, 226, 149));
        colors.add(new Color(178, 214, 117));
        colors.add(new Color(149, 190, 113));
        colors.add(new Color(117, 170, 101));
        colors.add(new Color(113, 153, 89));
        colors.add(new Color(18, 129, 242));
        colors.add(new Color(255, 0, 0));
        colors.add(new Color(0, 0, 0));
        return colors;
    }

    public static Font font(int size, boolean isBold) {
        if (isBold)
            return new Font("Arial", Font.BOLD, size);
        return new Font("Arial", Font.PLAIN, size);
    }

    public enum menuType {MAP, LOCATION, STATE}
}
