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
    private Stack<AssemblyHandler> assemblyHandlers;
    private Stack<AssemblyHandler> revertedChanges;
    private AnnotationLayerHandler contigLayerHandler;
    private AnnotationLayerHandler scaffoldLayerHandler;

    public AssemblyStateTracker(AssemblyHandler assemblyHandler, AnnotationLayerHandler contigLayerHandler, AnnotationLayerHandler scaffoldLayerHandler) {

        assemblyHandlers = new Stack<>();
        assemblyHandlers.push(assemblyHandler);
        this.contigLayerHandler = contigLayerHandler;
        this.scaffoldLayerHandler = scaffoldLayerHandler;
        revertedChanges = new Stack<AssemblyHandler>();
    }

    public AssemblyHandler getAssemblyHandler() {
        return assemblyHandlers.peek();
    }

    public AssemblyHandler getNewAssemblyHandler() {
        AssemblyHandler newAssemblyHandler = new AssemblyHandler(assemblyHandlers.peek());
        return newAssemblyHandler;
    }

    public void regenerateLayers(AssemblyHandler assemblyHandler) {
        assemblyHandlers.push(assemblyHandler);
        assemblyHandler.generateContigsAndScaffolds();

        AnnotationLayer scaffoldLayer = new AnnotationLayer(assemblyHandler.getScaffolds());
        scaffoldLayer.setLayerType(AnnotationLayer.LayerType.GROUP);
        scaffoldLayerHandler.setAnnotationLayer(scaffoldLayer);

        AnnotationLayer contigLayer = new AnnotationLayer(assemblyHandler.getContigs());
        contigLayer.setLayerType(AnnotationLayer.LayerType.MAIN);
        contigLayerHandler.setAnnotationLayer(contigLayer);


    }

    public boolean checkUndo() {
        return assemblyHandlers.size() > 1;
    }

    public void undo() {
        if (checkUndo()) {
            revertedChanges.push(assemblyHandlers.pop());
            regenerateLayers(assemblyHandlers.peek());
        }
    }

    public boolean checkRedo() {
        return !revertedChanges.empty();
    }

    public void redo() {
        if (checkRedo()) {
            assemblyHandlers.push(revertedChanges.pop());
            regenerateLayers(assemblyHandlers.peek());
        }
    }

}