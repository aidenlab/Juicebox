/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2017 Broad Institute, Aiden Lab
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

import java.util.List;

/**
 * Created by nathanielmusial on 7/10/17.
 */
public class AssemblyOperationExecutor {

    public static void splitContig(Feature2D originalContig, Feature2D debrisContig, SuperAdapter superAdapter, HiC hic, boolean moveTo) {
        AssemblyFragmentHandler assemblyFragmentHandler = superAdapter.getAssemblyStateTracker().getNewAssemblyHandler();
        assemblyFragmentHandler.editContig(originalContig, debrisContig);
        performAssemblyAction(superAdapter, assemblyFragmentHandler, false);
    }

    public static void invertSelection(SuperAdapter superAdapter, List<Feature2D> selectedFeatures) {

        AssemblyFragmentHandler assemblyFragmentHandler = superAdapter.getAssemblyStateTracker().getNewAssemblyHandler();
        assemblyFragmentHandler.invertSelection(selectedFeatures);
        performAssemblyAction(superAdapter, assemblyFragmentHandler, true);
    }

    public static void moveSelection(SuperAdapter superAdapter, List<Feature2D> selectedFeatures, Feature2D featureOrigin) {
        AssemblyFragmentHandler assemblyFragmentHandler = superAdapter.getAssemblyStateTracker().getNewAssemblyHandler();
        assemblyFragmentHandler.moveSelection(selectedFeatures, featureOrigin);
        performAssemblyAction(superAdapter, assemblyFragmentHandler, true);
    }

    public static void moveDebrisToEnd(SuperAdapter superAdapter) {
        AssemblyFragmentHandler assemblyFragmentHandler = superAdapter.getAssemblyStateTracker().getNewAssemblyHandler();
        assemblyFragmentHandler.moveDebrisToEnd();
        performAssemblyAction(superAdapter, assemblyFragmentHandler, true);
    }

    public static void toggleGroup(SuperAdapter superAdapter, Feature2D upstreamFeature2D, Feature2D downstreamFeature2D) {
        AssemblyFragmentHandler assemblyFragmentHandler = superAdapter.getAssemblyStateTracker().getNewAssemblyHandler();
        assemblyFragmentHandler.toggleGroup(upstreamFeature2D, downstreamFeature2D);
        superAdapter.refresh();
    }

    public static void performAssemblyAction(final SuperAdapter superAdapter, final AssemblyFragmentHandler assemblyFragmentHandler, final Boolean refeshMap) {
        Runnable runnable = new Runnable() {
            public void run() {
                superAdapter.getAssemblyStateTracker().assemblyActionPerformed(assemblyFragmentHandler);
                if (refeshMap) {
                    superAdapter.clearAllMatrixZoomCache();
                    superAdapter.refresh();
                }
            }
        };
        superAdapter.getMainWindow().executeLongRunningTask(runnable, "AssemblyAction");
    }
}
