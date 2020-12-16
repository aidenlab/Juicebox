/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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

package juicebox.data;

import java.util.Stack;

/**
 * Created by ranganmostofa on 7/8/17.
 */
public class ZoomActionTracker {
    private final int stackSizeLimit = 100;
    private ZoomAction currentZoomAction;
    private final Stack<ZoomAction> undoZoomActions = new Stack<>();
    private final Stack<ZoomAction> redoZoomActions = new Stack<>();

    public void undoZoom() {
        if (validateUndoZoom()) {
            redoZoomActions.push(undoZoomActions.pop());
            setCurrentZoomAction(undoZoomActions.peek());
        }
    }

    public void redoZoom() {
        if (validateRedoZoom()) {
            undoZoomActions.push(redoZoomActions.pop());
            setCurrentZoomAction(undoZoomActions.peek());
        }
    }

    public boolean validateUndoZoom() {
        return undoZoomActions.size() > 1;
    }

    public boolean validateRedoZoom() {
        return !redoZoomActions.isEmpty();
    }

    public void addZoomState(ZoomAction newZoomAction) {
        undoZoomActions.add(newZoomAction);
        setCurrentZoomAction(undoZoomActions.peek());
        redoZoomActions.clear();
        if (undoZoomActions.size() > stackSizeLimit) {
            undoZoomActions.remove(0);
        }
    }

    public void clear() {
        this.currentZoomAction = null;
        this.undoZoomActions.clear();
        this.redoZoomActions.clear();
    }

    public boolean equals(ZoomActionTracker other) {
        if (this == other) return true;
        if (other != null) {
            if (this.undoZoomActions.equals(other.getUndoZoomActions())) {
                if (this.redoZoomActions.equals(other.getRedoZoomActions())) {
                    return this.currentZoomAction.equals(other.getCurrentZoomAction());
                }
            }
        }
        return false;
    }

    public ZoomAction getCurrentZoomAction() {
        return this.currentZoomAction;
    }

    private void setCurrentZoomAction(ZoomAction zoomAction) {
        this.currentZoomAction = zoomAction;
    }

    private Stack<ZoomAction> getUndoZoomActions() {
        return this.undoZoomActions;
    }

    private Stack<ZoomAction> getRedoZoomActions() {
        return this.redoZoomActions;
    }
}
