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
import juicebox.track.feature.AnnotationLayerHandler;
import juicebox.track.feature.Feature2D;

import java.util.List;

/**
 * Created by nathanielmusial on 7/10/17.
 */
public class AssemblyOperationExecutor {

    private static SuperAdapter superAdapter;

    public static void splitContig(Feature2D originalContig, Feature2D debrisContig, SuperAdapter superAdapter, HiC hic) {
        AssemblyHandler assemblyHandler = superAdapter.getAssemblyStateTracker().getNewAssemblyHandler();
        assemblyHandler.editContig(originalContig, debrisContig);
        superAdapter.getAssemblyStateTracker().assemblyActionPerformed(assemblyHandler);
    }


    public static void splitGroup(List<Feature2D> selectedFeatures) {
        AssemblyHandler assemblyHandler = superAdapter.getAssemblyStateTracker().getNewAssemblyHandler();
        assemblyHandler.splitGroup(selectedFeatures);
        superAdapter.getAssemblyStateTracker().assemblyActionPerformed(assemblyHandler);
    }

    public static void mergeGroup(List<Feature2D> selectedFeatures) {
        String attributeName = "Scaffold Number";
        AnnotationLayerHandler groupLayer = superAdapter.getActiveLayerHandler(); //todo make check for group layer
        int startingIndex = Integer.parseInt(selectedFeatures.get(0).getAttribute(attributeName));
        System.out.println(startingIndex);
        AssemblyHandler assemblyHandler = superAdapter.getAssemblyStateTracker().getNewAssemblyHandler();
        assemblyHandler.mergeGroup(startingIndex, selectedFeatures);
        superAdapter.getAssemblyStateTracker().assemblyActionPerformed(assemblyHandler);
    }

    public static void invertSelection(List<Feature2D> selectedFeatures) {
        AssemblyHandler assemblyHandler = superAdapter.getAssemblyStateTracker().getNewAssemblyHandler();
        assemblyHandler.invertSelection(selectedFeatures);
        superAdapter.getAssemblyStateTracker().assemblyActionPerformed(assemblyHandler);

    }

    public static void moveSelectedFeatures(List<Feature2D> selectedFeatures, Feature2D featureOrigin) {
        AssemblyHandler assemblyHandler = superAdapter.getAssemblyStateTracker().getNewAssemblyHandler();
        assemblyHandler.translateSelection(selectedFeatures, featureOrigin);
        superAdapter.getAssemblyStateTracker().assemblyActionPerformed(assemblyHandler);
    }
}
