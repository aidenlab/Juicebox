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

package juicebox.assembly;

import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.gui.SuperAdapter;
import juicebox.track.feature.Feature2D;

import java.util.List;

/**
 * Created by nathanielmusial on 7/10/17.
 */
public class AssemblyOperationExecutor {

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

    public static void moveAndDisperseSelection(SuperAdapter superAdapter, List<Feature2D> selectedFeatures, Feature2D featureOrigin) {
        if (selectedFeatures != null && !selectedFeatures.isEmpty()) {
            AssemblyScaffoldHandler assemblyScaffoldHandler = superAdapter.getAssemblyStateTracker().getNewAssemblyHandler();
            assemblyScaffoldHandler.moveSelection(selectedFeatures, featureOrigin);
            assemblyScaffoldHandler.multiSplit(selectedFeatures);
            performAssemblyAction(superAdapter, assemblyScaffoldHandler, true);
        }
    }

    public static void toggleGroup(SuperAdapter superAdapter, Feature2D upstreamFeature2D, Feature2D downstreamFeature2D) {
        if (upstreamFeature2D != null && downstreamFeature2D != null) {
            AssemblyScaffoldHandler assemblyScaffoldHandler = superAdapter.getAssemblyStateTracker().getNewAssemblyHandler();
            assemblyScaffoldHandler.toggleGroup(upstreamFeature2D, downstreamFeature2D);
            performAssemblyAction(superAdapter, assemblyScaffoldHandler, HiCGlobals.phasing);
        }
    }

    public static void multiMerge(SuperAdapter superAdapter, List<Feature2D> selectedFeatures) {
        if (selectedFeatures != null && !selectedFeatures.isEmpty()) {
            AssemblyScaffoldHandler assemblyScaffoldHandler = superAdapter.getAssemblyStateTracker().getNewAssemblyHandler();
            assemblyScaffoldHandler.multiMerge(selectedFeatures.get(0), selectedFeatures.get(selectedFeatures.size() - 1));
            performAssemblyAction(superAdapter, assemblyScaffoldHandler, false);
        }
    }

    public static void multiSplit(SuperAdapter superAdapter, List<Feature2D> selectedFeatures) {
        if (selectedFeatures != null && !selectedFeatures.isEmpty()) {
            AssemblyScaffoldHandler assemblyScaffoldHandler = superAdapter.getAssemblyStateTracker().getNewAssemblyHandler();
            assemblyScaffoldHandler.multiSplit(selectedFeatures);
            performAssemblyAction(superAdapter, assemblyScaffoldHandler, false);
        }
    }

    public static void phaseMerge(SuperAdapter superAdapter, List<Integer> selectedFeatures) {
        if (selectedFeatures != null && !selectedFeatures.isEmpty()) {
            AssemblyScaffoldHandler assemblyScaffoldHandler = superAdapter.getAssemblyStateTracker().getNewAssemblyHandler();
            assemblyScaffoldHandler.phaseMerge(selectedFeatures);
            performAssemblyAction(superAdapter, assemblyScaffoldHandler, true);
        }
    }

    private static void performAssemblyAction(final SuperAdapter superAdapter, final AssemblyScaffoldHandler assemblyScaffoldHandler, final Boolean refreshMap) {

        superAdapter.getAssemblyStateTracker().assemblyActionPerformed(assemblyScaffoldHandler, refreshMap);
        if (refreshMap) superAdapter.safeClearAllMZDCache();

    }
}
