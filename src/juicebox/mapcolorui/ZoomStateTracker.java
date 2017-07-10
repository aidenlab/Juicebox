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

package juicebox.mapcolorui;

import java.util.Stack;

/**
 * Created by ranganmostofa on 7/8/17.
 */
public class ZoomStateTracker {
    private ZoomState currentZoomState;
    private Stack<ZoomState> undoZoomStates = new Stack<>();
    private Stack<ZoomState> redoZoomStates = new Stack<>();

    public ZoomStateTracker(ZoomState currentZoomState) {
        this.currentZoomState = currentZoomState;
        undoZoomStates.add(currentZoomState);
    }

    public void undoZoom() {
        if (validateUndoZoom()) {
            redoZoomStates.push(undoZoomStates.pop());
            setCurrentZoomState(undoZoomStates.peek());
        }
    }

    public void redoZoom() {
        if (validateRedoZoom()) {
            undoZoomStates.push(redoZoomStates.pop());
            setCurrentZoomState(undoZoomStates.peek());
        }
    }

    private boolean validateUndoZoom() {
        return undoZoomStates.size() > 1;
    }

    private boolean validateRedoZoom() {
        return !redoZoomStates.isEmpty();
    }

    public void addZoomState(ZoomState newZoomState) {
        undoZoomStates.add(newZoomState);
        redoZoomStates.clear();
    }

    public boolean equals(ZoomStateTracker other) {
        if (sameObject(other)) return true;
        if (other != null) {
            if (this.undoZoomStates.equals(other.getUndoZoomStates())) {
                if (this.redoZoomStates.equals(other.getRedoZoomStates())) {
                    if (this.currentZoomState.equals(other.getCurrentZoomState())) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    public boolean sameObject(ZoomStateTracker other) {
        return this == other;
    }

    @SuppressWarnings("MethodDoesntCallSuperMethod")
    public ZoomStateTracker deepCopy() {
        return new ZoomStateTracker(this.currentZoomState);
    }

    public ZoomState getCurrentZoomState() {
        return this.currentZoomState;
    }

    private void setCurrentZoomState(ZoomState zoomState) {
        this.currentZoomState = zoomState;
    }

    private Stack<ZoomState> getUndoZoomStates() {
        return this.undoZoomStates;
    }

    private Stack<ZoomState> getRedoZoomStates() {
        return this.redoZoomStates;
    }
}
