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

import java.util.ArrayList;
import java.util.List;

/**
 * Created by nathanielmusial on 7/10/17.
 */
public class AssemblyOperationExecutor {

    private static SuperAdapter superAdapter;

    public static void splitContig(Feature2D originalContig, Feature2D debrisContig, SuperAdapter superAdapter, HiC hic) {
        AssemblyFragmentHandler assemblyFragmentHandler = superAdapter.getAssemblyStateTracker().getNewAssemblyHandler();
        assemblyFragmentHandler.editContig(originalContig, debrisContig);
        superAdapter.getAssemblyStateTracker().assemblyActionPerformed(assemblyFragmentHandler);
    }


    public static void splitGroup(SuperAdapter superAdapter, List<Feature2D> selectedFeatures) {
        AssemblyFragmentHandler assemblyFragmentHandler = superAdapter.getAssemblyStateTracker().getNewAssemblyHandler();
        assemblyFragmentHandler.splitGroup(selectedFeatures);
        superAdapter.getAssemblyStateTracker().assemblyActionPerformed(assemblyFragmentHandler);
    }

    public static void mergeGroup(SuperAdapter superAdapter, List<Feature2D> selectedFeatures) {
        String attributeName = "Scaffold Number";
        AnnotationLayerHandler groupLayer = superAdapter.getActiveLayerHandler(); //todo make check for group layer
        int startingIndex = Integer.parseInt(selectedFeatures.get(0).getAttribute(attributeName));
        System.out.println(startingIndex);
        AssemblyFragmentHandler assemblyFragmentHandler = superAdapter.getAssemblyStateTracker().getNewAssemblyHandler();
        assemblyFragmentHandler.mergeGroup(startingIndex, selectedFeatures);
        superAdapter.getAssemblyStateTracker().assemblyActionPerformed(assemblyFragmentHandler);
    }

    public static void invertSelection(SuperAdapter superAdapter, List<Feature2D> selectedFeatures, List<Feature2D> contigs, int startIndex, int endIndex) {
        List<Feature2D> duplicateSelectedFeatures = new ArrayList<>();
        for (Feature2D feature2D : selectedFeatures) {
            duplicateSelectedFeatures.add(feature2D.deepCopy());
        }

        AssemblyHeatmapHandler.invertMultipleContiguousEntriesAt(contigs, startIndex, endIndex);
        AssemblyHeatmapHandler.recalculateAllAlterations(contigs);

        superAdapter.getContigLayer().getAnnotationLayer().getFeatureHandler().remakeRTree();
        superAdapter.refresh();

        AssemblyFragmentHandler assemblyFragmentHandler = superAdapter.getAssemblyStateTracker().getNewAssemblyHandler();
        assemblyFragmentHandler.invertSelection(duplicateSelectedFeatures);
        superAdapter.getAssemblyStateTracker().assemblyActionPerformed(assemblyFragmentHandler, true);
    }

    public static void moveSelectedFeatures(SuperAdapter superAdapter, List<Feature2D> selectedFeatures, Feature2D featureOrigin) {
        AssemblyFragmentHandler assemblyFragmentHandler = superAdapter.getAssemblyStateTracker().getNewAssemblyHandler();
        assemblyFragmentHandler.translateSelection(selectedFeatures, featureOrigin);
        superAdapter.getAssemblyStateTracker().assemblyActionPerformed(assemblyFragmentHandler);
    }
}
