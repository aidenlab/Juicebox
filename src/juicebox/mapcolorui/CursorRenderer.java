/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2021 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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
 *  FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.mapcolorui;

import java.awt.*;

public class CursorRenderer {
    private final HeatmapPanel parent;

    public CursorRenderer(HeatmapPanel heatmapPanel) {
        this.parent = heatmapPanel;
    }

    public void drawCursors(Graphics2D g, Point cursorPoint, Point diagonalCursorPoint,
                            double binOriginX, double binOriginY, double scaleFactor, Color colorForRuler,
                            double bRight, double bBottom) {
        if (cursorPoint != null) {
            g.setColor(colorForRuler);
            g.drawLine(cursorPoint.x, 0, cursorPoint.x, parent.getHeight());
            g.drawLine(0, cursorPoint.y, parent.getWidth(), cursorPoint.y);
        } else if (diagonalCursorPoint != null) {
            g.setColor(colorForRuler);
            // quadrant 4 signs in plotting equal to quadrant 1 flipped across x in cartesian plane
            // y = -x + b
            // y + x = b
            int b = diagonalCursorPoint.x + diagonalCursorPoint.y;
            // at x = 0, y = b unless y exceeds height
            int pHeight = parent.getHeight();
            int pWidth = parent.getWidth();


            int leftEdgeY = Math.min(b, pHeight);
            int leftEdgeX = b - leftEdgeY;
            // at y = 0, x = b unless x exceeds width
            int rightEdgeX = Math.min(b, pWidth);
            int rightEdgeY = b - rightEdgeX;
            g.drawLine(leftEdgeX, leftEdgeY, rightEdgeX, rightEdgeY);

            // now we need to draw the perpendicular
            // line which intersects this at the mouse
            // m = -1, neg reciprocal is 1
            // y2 = x2 + b2
            // y2 - x2 = b2
            int b2 = diagonalCursorPoint.y - diagonalCursorPoint.x;
            // at x2 = 0, y2 = b2 unless y less than 0
            int leftEdgeY2 = Math.max(b2, 0);
            int leftEdgeX2 = leftEdgeY2 - b2;
            // at x2 = width, y2 = width+b2 unless x exceeds height
            int rightEdgeY2 = Math.min(pWidth + b2, pHeight);
            int rightEdgeX2 = rightEdgeY2 - b2;
            g.drawLine(leftEdgeX2, leftEdgeY2, rightEdgeX2, rightEdgeY2);

            // find a point on the diagonal (binx = biny)
            double binXYOrigin = Math.max(binOriginX, binOriginY);
            // ensure diagonal is in view
            if (binXYOrigin < bRight && binXYOrigin < bBottom) {
                int xDiag = (int) ((binXYOrigin - binOriginX) * scaleFactor);
                int yDiag = (int) ((binXYOrigin - binOriginY) * scaleFactor);
                // see if new point is above the line y2 = x2 + b2
                // y' less than due to flipped topography
                int vertDisplacement = yDiag - (xDiag + b2);
                // displacement takes care of directionality of diagonal
                // being above/below is the second line we drew
                int b3 = b2 + (2 * vertDisplacement);
                // at x2 = 0, y2 = b2 unless y less than 0
                int leftEdgeY3 = Math.max(b3, 0);
                int leftEdgeX3 = leftEdgeY3 - b3;
                // at x2 = width, y2 = width+b2 unless x exceeds height
                int rightEdgeY3 = Math.min(pWidth + b3, pHeight);
                int rightEdgeX3 = rightEdgeY3 - b3;
                g.drawLine(leftEdgeX3, leftEdgeY3, rightEdgeX3, rightEdgeY3);
            }
        }
    }
}
