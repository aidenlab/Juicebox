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
import java.util.*;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * Implementation of 2-pass algorithm for finding connected components
 */
public class BinaryConnectedComponents {

    // unique labels for components, start at 0
    private static Integer nextLabel = 0;

    /**
     * @param image
     * @param threshold
     * @return list of connected components in image
     */
    public static List<Set<Point>> detection(double[][] image, double threshold) {
        int r = image.length;
        int c = image[0].length;

        // pixel label matrix
        int[][] labels = new int[r][c];

        List<IndexNode> indices = new ArrayList<IndexNode>();
        indices.add(new IndexNode(-1));
        nextLabel = 1;

        // 1st pass
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                if (image[i][j] > threshold) {
                    processNeighbors(labels, indices, i, j, Math.max(i - 1, 0), Math.min(i + 1, r - 1), Math.max(j - 1, 0), Math.min(j + 1, c - 1));
                }
            }
        }
        return processLabeledIndices(indices);
    }

    /**
     * 2nd pass of algorithm
     *
     * @param indices
     * @return connected components
     */
    private static List<Set<Point>> processLabeledIndices(List<IndexNode> indices) {
        List<Set<Point>> components = new ArrayList<Set<Point>>();
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
                components.add(points);
            }
        }
        return components;
    }

    /**
     * Label the neighbors of the pixel
     *
     * @param labels
     * @param indices
     * @param i
     * @param j
     * @param left
     * @param right
     * @param top
     * @param bottom
     */
    private static void processNeighbors(int[][] labels, List<IndexNode> indices, int i, int j,
                                         int left, int right, int top, int bottom) {

        // 8 - point connectivity
        int[][] neighborLabels = getSubMatrix(labels, left, right, top, bottom);

        // 0 - means none found
        Set<Integer> allPosVals = positiveValues(neighborLabels);

        int lowestLabel = 0;
        if (allPosVals.size() > 0)
            lowestLabel = Collections.min(new ArrayList<Integer>(allPosVals));

        if (lowestLabel <= 0) {
            lowestLabel = nextLabel;
            indices.add(new IndexNode(lowestLabel));
            nextLabel++;
        }

        labels[i][j] = lowestLabel;
        IndexNode current = indices.get(lowestLabel);
        current.addPoint(new Point(i, j));

        for (Integer k : allPosVals) {
            IndexNode other = indices.get(k);
            other.addConnections(current);
            current.addConnections(other);
        }

    }

    private static int[][] getSubMatrix(int[][] matrix, int left, int right, int top, int bottom) {
        int[][] subMatrix = new int[right - left + 1][bottom - top + 1];
        for (int i = 0; i < subMatrix.length; i++) {
            System.arraycopy(matrix[left + i], top, subMatrix[i], 0, subMatrix[0].length);
        }
        return subMatrix;
    }

    private static Set<Integer> positiveValues(int[][] matrix) {
        Set<Integer> values = new HashSet<Integer>();
        for (int[] row : matrix) {
            for (int val : row) {
                if (val > 0) {
                    values.add(val);
                }
            }
        }
        return values;
    }
}
