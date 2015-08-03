/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2015 Broad Institute, Aiden Lab
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


package juicebox.tools.utils.juicer.arrowhead.connectedcomponents;

import java.awt.*;
import java.util.HashSet;
import java.util.Set;

/**
 * Created by muhammadsaadshamim on 6/5/15.
 */
class IndexNode {
    private final int n;
    private final Set<IndexNode> connectedNodes = new HashSet<IndexNode>();
    private final Set<Point> matrixIndices = new HashSet<Point>();
    private boolean hasNotBeenIndexed = true;

    public IndexNode(int n) {
        this.n = n;
    }

    public void addConnections(IndexNode node) {
        connectedNodes.add(node);
    }

    public void addPoint(Point point) {
        matrixIndices.add(point);
    }

    public Set<Point> getMatrixIndices() {
        return matrixIndices;
    }

    public void index() {
        hasNotBeenIndexed = false;
    }

    public boolean hasNotBeenIndexed() {
        return hasNotBeenIndexed;
    }

    public Set<IndexNode> getConnectedNodes() {
        return connectedNodes;
    }

    public int getIndex() {
        return n;
    }
}
