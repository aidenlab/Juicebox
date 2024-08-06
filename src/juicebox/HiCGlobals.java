/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2024 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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

package juicebox;

import java.awt.*;
import java.io.File;
import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * @author Muhammad Shamim
 * @since 11/25/14
 */
public class HiCGlobals {

    public static final String versionNum = "2.24.00";
    public static final String juiceboxTitle = "[Juicebox " + versionNum + "] Hi-C Map ";

    // MainWindow variables
    public static final Color RULER_LINE_COLOR = new Color(0, 0, 230, 100);
    public static final Color DARKULA_RULER_LINE_COLOR = new Color(200, 200, 250, 100);

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
    public static final int minVersion = 6;
    public static final int writingVersion = 9;
    public static final int bufferSize = 2097152;
    public static final String defaultPropertiesURL = "http://hicfiles.tc4ga.com/juicebox.properties";
    public static final Color diffGrayColor = new Color(238, 238, 238);
    // for state saving
    public static int MAX_PEARSON_ZOOM = 50000;
    public static int MAX_EIGENVECTOR_ZOOM = 250000;
    // implement Map scaling with this global variable
    public static double hicMapScale = 1;
    // whether MatrixZoomData should cache or not
    public static boolean useCache = true;
    public static boolean guiIsCurrentlyActive = false;
    public static boolean allowDynamicBlockIndex = true;
    public static boolean printVerboseComments = false;
    public static boolean slideshowEnabled = false;
    public static boolean splitModeEnabled = false;
    public static boolean translationInProgress = false;
    public static boolean displayTiles = false;
    public static boolean isDarkulaModeEnabled = false;
    public static boolean isAssemblyMatCheck = false;

    // whether instance was linked before mouse press or not

    public static boolean phasing = false;
    public static boolean noSortInPhasing = false;
    public static boolean wasLinkedBeforeMousePress = false;
    public static boolean isLegacyOutputPrintingEnabled = false;
    public static final boolean isDevAssemblyToolsAllowedPublic = true;
    public static final boolean isDevCustomChromosomesAllowedPublic = true;
    public static boolean HACK_COLORSCALE = false;
    public static boolean HACK_COLORSCALE_EQUAL = false;
    public static boolean HACK_COLORSCALE_LINEAR = false;

    // for norm/pre, save contact records into memory
    public static boolean USE_ITERATOR_NOT_ALL_IN_RAM = false;
    public static boolean CHECK_RAM_USAGE = false;

    public static void verifySupportedHiCFileVersion(int version) throws RuntimeException {
        if (version < minVersion) {
            throw new RuntimeException("This file is version " + version +
                    ". Only versions " + minVersion + " and greater are supported at this time.");
        }
    }

    public static void verifySupportedHiCFileWritingVersion(int version) throws RuntimeException {
        if (version < writingVersion) {
            throw new RuntimeException("This file is version " + version +
                    ". Only versions " + writingVersion + " and greater can be edited using this jar.");
        }
    }

    public static Font font(int size, boolean isBold) {
        if (isBold)
            return new Font("Arial", Font.BOLD, size);
        return new Font("Arial", Font.PLAIN, size);
    }

    public static int getIdealThreadCount() {
        return Math.max(1, Runtime.getRuntime().availableProcessors());
    }

    public static ExecutorService newFixedThreadPool() {
        return Executors.newFixedThreadPool(getIdealThreadCount());
    }

    public enum menuType {MAP, LOCATION, STATE}
}
