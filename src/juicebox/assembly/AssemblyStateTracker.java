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

import juicebox.gui.SuperAdapter;

import java.util.Stack;

/**
 * Created by nathanielmusial on 7/5/17.
 */
public class AssemblyStateTracker {
    private Stack<AssemblyScaffoldHandler> undoStack;
    private Stack<AssemblyScaffoldHandler> redoStack;
    private AssemblyScaffoldHandler initialAssemblyScaffoldHandler;
    private SuperAdapter superAdapter;

    public AssemblyStateTracker(AssemblyScaffoldHandler assemblyScaffoldHandler) {

        undoStack = new Stack<>();
<<<<<<< HEAD
        undoStack.push(assemblyFragmentHandler);
        this.contigLayerHandler = contigLayerHandler;
        this.scaffoldLayerHandler = scaffoldLayerHandler;
        redoStack = new Stack<>();
        this.initialAssemblyFragmentHandler = assemblyFragmentHandler;
=======
        undoStack.push(assemblyScaffoldHandler);
        redoStack = new Stack<AssemblyScaffoldHandler>();
        this.initialAssemblyScaffoldHandler = assemblyScaffoldHandler;
>>>>>>> AggregateProcessingDevelopment
    }

    public AssemblyScaffoldHandler getAssemblyHandler() {
        return undoStack.peek();
    }

    public AssemblyScaffoldHandler getNewAssemblyHandler() {
        AssemblyScaffoldHandler newAssemblyScaffoldHandler = new AssemblyScaffoldHandler(undoStack.peek());
        //AssemblyScaffoldHandler newAssemblyScaffoldHandler = undoStack.peek();
        return newAssemblyScaffoldHandler;
    }

    public void resetState() {
        undoStack.clear();
<<<<<<< HEAD
        redoStack.clear();
        undoStack.push(initialAssemblyFragmentHandler);
        regenerateLayers();
        executeClearAllMZDCache(AssemblyHeatmapHandler.getSuperAdapter());
=======
        assemblyActionPerformed(this.initialAssemblyScaffoldHandler, true);
        executeLongRunningTask(this.superAdapter);
>>>>>>> AggregateProcessingDevelopment
    }

    public AssemblyScaffoldHandler getInitialAssemblyScaffoldHandler() {
        return this.initialAssemblyScaffoldHandler;
    }

    public void assemblyActionPerformed(AssemblyScaffoldHandler assemblyScaffoldHandler, boolean refreshMap) {
        redoStack.clear();
        undoStack.push(assemblyScaffoldHandler);
        assemblyScaffoldHandler.updateAssembly(refreshMap);
        regenerateLayers(refreshMap);
        //     System.out.println(assemblyScaffoldHandler.toString());
    }

<<<<<<< HEAD
    public void regenerateLayers() {
        AssemblyFragmentHandler assemblyFragmentHandler = undoStack.peek();
        assemblyFragmentHandler.updateAssembly();
        scaffoldLayerHandler.getFeatureHandler().setLoopList(assemblyFragmentHandler.getSuperscaffoldFeature2DList());
        contigLayerHandler.getFeatureHandler().setLoopList(assemblyFragmentHandler.getScaffoldFeature2DList());
=======
    public void regenerateLayers(boolean refreshMap) {
        AssemblyScaffoldHandler assemblyScaffoldHandler = undoStack.peek();
        if (refreshMap) {
            superAdapter.getMainLayer().getFeatureHandler().loadLoopList(assemblyScaffoldHandler.getScaffoldFeature2DHandler().getAllVisibleLoops(), true);
        }
        superAdapter.getGroupLayer().getFeatureHandler().loadLoopList(assemblyScaffoldHandler.getSuperscaffoldFeature2DHandler().getAllVisibleLoops(), false);
>>>>>>> AggregateProcessingDevelopment
    }

    public boolean checkUndo() {
        return undoStack.size() > 1;
    }

    public void undo() {
        if (checkUndo()) {
            redoStack.push(undoStack.pop());
<<<<<<< HEAD
            regenerateLayers();
            executeClearAllMZDCache(AssemblyHeatmapHandler.getSuperAdapter());
=======
            undoStack.peek().updateAssembly(true);
            regenerateLayers(true);
            executeLongRunningTask(this.superAdapter);
>>>>>>> AggregateProcessingDevelopment
        }
    }

    public boolean checkRedo() {
        return !redoStack.empty();
    }

    public void redo() {
        if (checkRedo()) {
            undoStack.push(redoStack.pop());
<<<<<<< HEAD
            regenerateLayers();
            executeClearAllMZDCache(AssemblyHeatmapHandler.getSuperAdapter());
=======
            undoStack.peek().updateAssembly(true);
            regenerateLayers(true);
            executeLongRunningTask(this.superAdapter);
>>>>>>> AggregateProcessingDevelopment
        }
    }

    public void executeClearAllMZDCache(final SuperAdapter superAdapter) {
        Runnable runnable = new Runnable() {
            public void run() {
                superAdapter.clearAllMatrixZoomCache(); //split clear current zoom and put the rest in background? Seems to taking a lot of time
                superAdapter.refresh();
            }
        };
        superAdapter.executeLongRunningTask(runnable, "Assembly clear MZD cache");
    }

    public void setSuperAdapter(SuperAdapter superAdapter) {
        this.superAdapter = superAdapter;
    }
}