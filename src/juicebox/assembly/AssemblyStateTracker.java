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

import juicebox.track.feature.AnnotationLayer;
import juicebox.track.feature.AnnotationLayerHandler;

import java.util.Stack;

/**
 * Created by nathanielmusial on 7/5/17.
 */
public class AssemblyStateTracker {
    private Stack<AssemblyFragmentHandler> undoStack;
    private Stack<AssemblyFragmentHandler> redoStack;
    private AnnotationLayerHandler contigLayerHandler;
    private AnnotationLayerHandler scaffoldLayerHandler;

    public AssemblyStateTracker(AssemblyFragmentHandler assemblyFragmentHandler, AnnotationLayerHandler contigLayerHandler, AnnotationLayerHandler scaffoldLayerHandler) {

        undoStack = new Stack<>();
        undoStack.push(assemblyFragmentHandler);
        this.contigLayerHandler = contigLayerHandler;
        this.scaffoldLayerHandler = scaffoldLayerHandler;
        redoStack = new Stack<AssemblyFragmentHandler>();
    }

    public AssemblyFragmentHandler getAssemblyHandler() {
        return undoStack.peek();
    }

    public AssemblyFragmentHandler getNewAssemblyHandler() {
        AssemblyFragmentHandler newAssemblyFragmentHandler = new AssemblyFragmentHandler(undoStack.peek());
        return newAssemblyFragmentHandler;
    }

    public void assemblyActionPerformed(AssemblyFragmentHandler assemblyFragmentHandler, boolean regenerateLayers) {
        redoStack.clear();
        undoStack.push(assemblyFragmentHandler);
        if (regenerateLayers) {
            regenerateLayers();
        }
    }

    public void assemblyActionPerformed(AssemblyFragmentHandler assemblyFragmentHandler) {
        redoStack.clear();
        undoStack.push(assemblyFragmentHandler);
        regenerateLayers();
    }

    public void regenerateLayers() {
        AssemblyFragmentHandler assemblyFragmentHandler = undoStack.peek();
        assemblyFragmentHandler.generateContigsAndScaffolds();

        AnnotationLayer scaffoldLayer = new AnnotationLayer(assemblyFragmentHandler.getScaffolds());
        scaffoldLayer.setLayerType(AnnotationLayer.LayerType.GROUP);
        scaffoldLayerHandler.setAnnotationLayer(scaffoldLayer);

        AnnotationLayer contigLayer = new AnnotationLayer(assemblyFragmentHandler.getContigs());
        contigLayer.setLayerType(AnnotationLayer.LayerType.MAIN);
        contigLayerHandler.setAnnotationLayer(contigLayer);
    }

    public boolean checkUndo() {
        return undoStack.size() > 1;
    }

    public void undo() {
        if (checkUndo()) {
            AssemblyHeatmapHandler.getSuperAdapter().clearAllMatrixZoomCache();
            redoStack.push(undoStack.pop());
            regenerateLayers();
        }
    }

    public boolean checkRedo() {
        return !redoStack.empty();
    }

    public void redo() {
        if (checkRedo()) {
            AssemblyHeatmapHandler.getSuperAdapter().clearAllMatrixZoomCache();
            undoStack.push(redoStack.pop());
            regenerateLayers();
        }
    }

}