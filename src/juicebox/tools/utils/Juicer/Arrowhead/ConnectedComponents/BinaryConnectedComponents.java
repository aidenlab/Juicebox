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

package juicebox.tools.utils.Juicer.Arrowhead.ConnectedComponents;

import juicebox.tools.utils.Common.MatrixTools;
import org.apache.commons.math.linear.RealMatrix;

import java.awt.*;
import java.util.*;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * Created by muhammadsaadshamim on 6/5/15.
 */
public class BinaryConnectedComponents {

    public static List<Set<Point>> detection(RealMatrix image) {
        int r = image.getRowDimension();
        int c = image.getColumnDimension();

        RealMatrix labels = MatrixTools.cleanArray2DMatrix(r, c);

        List<IndexNode> indices = new ArrayList<IndexNode>();
        indices.add(new IndexNode(-1));
        int nextLabel = 1;
        RealMatrix neighborLabels;


        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                if (image.getEntry(i, j) > 0) {

                    // 8 - point connectivity
                    neighborLabels = labels.getSubMatrix(
                            Math.max(i - 1, 0), Math.min(i + 1, r - 1), Math.max(j - 1, 0), Math.min(j + 1, c - 1));

                    // 0 means none found
                    int lowestLabel = (int) MatrixTools.minimumPositive(neighborLabels);
                    Set<Integer> allPosVals = positiveValues(neighborLabels);

                    if (lowestLabel <= 0) {
                        lowestLabel = nextLabel;
                        indices.add(new IndexNode(lowestLabel));
                        nextLabel++;
                    }

                    labels.setEntry(i, j, lowestLabel);
                    IndexNode current = indices.get(lowestLabel);
                    current.addPoint(new Point(i, j));

                    for (Integer k : allPosVals) {
                        IndexNode other = indices.get(k);
                        other.addConnections(current);
                        current.addConnections(other);
                    }
                }
            }
        }


        List<Set<Point>> cleanedUpComponents = new ArrayList<Set<Point>>();
        for (int i = 1; i < nextLabel; i++) {
            IndexNode current = indices.get(i);
            if (current.hasNotBeenIndexed()) {
                Queue<IndexNode> queue = new LinkedBlockingQueue<IndexNode>();
                Set<Point> points = new HashSet<Point>(current.getMatrixIndices());
                queue.addAll(current.getConnectedNodes());
                current.index();

                while (!queue.isEmpty()) {
                    IndexNode node = queue.poll();
                    if (node.hasNotBeenIndexed()) {
                        points.addAll(node.getMatrixIndices());
                        node.index();
                        for (IndexNode node2 : node.getConnectedNodes()) {
                            if (node2.hasNotBeenIndexed())
                                queue.add(node2);
                        }
                    }
                }
                cleanedUpComponents.add(points);
            }
        }

        return cleanedUpComponents;
    }

    private static Set<Integer> positiveValues(RealMatrix matrix) {
        Set<Integer> values = new HashSet<Integer>();
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (double val : matrix.getRow(i)) {
                if (val > 0) {
                    values.add((int) val);
                }
            }
        }
        return values;
    }
}
