/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2018 Broad Institute, Aiden Lab
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

package juicebox.assembly;

import juicebox.HiC;
import juicebox.gui.SuperAdapter;
import juicebox.track.feature.Feature2D;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by nathanielmusial on 7/10/17.
 */
public class AssemblyOperationExecutor {
    private static String assemblyTrackingSaveLocation;
    private static boolean assemblyTrackingEnabled = false;

    public static void splitContig(Feature2D originalContig, Feature2D debrisContig, SuperAdapter superAdapter, HiC hic, boolean moveTo) {
        AssemblyScaffoldHandler assemblyScaffoldHandler = superAdapter.getAssemblyStateTracker().getNewAssemblyHandler();
        assemblyScaffoldHandler.editScaffold(originalContig, debrisContig);
        performAssemblyAction(superAdapter, assemblyScaffoldHandler, true);
    }

    public static void invertSelection(SuperAdapter superAdapter, List<Feature2D> selectedFeatures) {
        if (selectedFeatures != null && !selectedFeatures.isEmpty()) {
            AssemblyScaffoldHandler assemblyScaffoldHandler = superAdapter.getAssemblyStateTracker().getNewAssemblyHandler();
            assemblyScaffoldHandler.invertSelection(selectedFeatures);
            performAssemblyAction(superAdapter, assemblyScaffoldHandler, true);
        }
    }

    public static void moveSelection(SuperAdapter superAdapter, List<Feature2D> selectedFeatures, Feature2D featureOrigin) {
        if (selectedFeatures != null && !selectedFeatures.isEmpty()) {
            AssemblyScaffoldHandler assemblyScaffoldHandler = superAdapter.getAssemblyStateTracker().getNewAssemblyHandler();
            assemblyScaffoldHandler.moveSelection(selectedFeatures, featureOrigin);
            performAssemblyAction(superAdapter, assemblyScaffoldHandler, true);
        }
    }

    public static void toggleGroup(SuperAdapter superAdapter, Feature2D upstreamFeature2D, Feature2D downstreamFeature2D) {
        if (upstreamFeature2D != null && downstreamFeature2D != null) {
            AssemblyScaffoldHandler assemblyScaffoldHandler = superAdapter.getAssemblyStateTracker().getNewAssemblyHandler();
            assemblyScaffoldHandler.toggleGroup(upstreamFeature2D, downstreamFeature2D);
            performAssemblyAction(superAdapter, assemblyScaffoldHandler, false);
        }
    }


    private static void performAssemblyAction(final SuperAdapter superAdapter, final AssemblyScaffoldHandler assemblyScaffoldHandler, final Boolean refreshMap) {

        superAdapter.getAssemblyStateTracker().assemblyActionPerformed(assemblyScaffoldHandler, refreshMap);
        if (assemblyTrackingEnabled) {
            AssemblyFileExporter assemblyFileExporter = new AssemblyFileExporter(assemblyScaffoldHandler, assemblyTrackingSaveLocation + "/" + "test_" + System.currentTimeMillis());
            assemblyFileExporter.exportCpropsAndAsm();
        }
        if (refreshMap) superAdapter.safeClearAllMZDCache();

    }

    public static void enableAssemblyTracking(String saveLocation) {
        assemblyTrackingSaveLocation = saveLocation;
        assemblyTrackingEnabled = true;
    }

    public static void loadAssemblyTracking(SuperAdapter superAdapter, String directory) {
        File[] files = new File(directory).listFiles();
//If this pathname does not denote a directory, then listFiles() returns null.
        List<String> results = new ArrayList<String>();
        for (File file : files) {
            if (file.isFile()) {
                if (file.getName().contains(".assembly"))
                    results.add(file.getPath());
            }
        }

        Collections.sort(results);


        for (String assemblyPath : results) {
            AssemblyFileImporter assemblyFileImporter;
            assemblyFileImporter = new AssemblyFileImporter(assemblyPath, true);
            assemblyFileImporter.importAssembly();

            AssemblyScaffoldHandler modifiedAssemblyScaffoldHandler = assemblyFileImporter.getAssemblyScaffoldHandler();
            superAdapter.getAssemblyStateTracker().assemblyActionPerformed(modifiedAssemblyScaffoldHandler, true);
            superAdapter.safeClearAllMZDCache();
            superAdapter.refresh();
        }

        while (superAdapter.getAssemblyStateTracker().checkUndo()) {
            superAdapter.getAssemblyStateTracker().undo();
        }
    }
}