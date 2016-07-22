/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2016 Broad Institute, Aiden Lab
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

import juicebox.MainWindow;
import juicebox.gui.MainViewPanel;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.plaf.basic.BasicSliderUI;
import java.awt.*;
import java.awt.event.MouseEvent;

/**
 * UI delegate for the RangeSlider component.  RangeSliderUI paints two thumbs,
 * one for the lower value and one for the upper value.
 *
 * @author Ernest Yu, Muhammad S Shamim
 */
class RangeSliderUI extends BasicSliderUI {

    /**
     * Color of selected range.
     */

    private final Color rangeColorBlank = Color.gray;
    private final Color[] gradientColorsBlank = {Color.gray, Color.gray};
    private final float[] fractionsBlank = {0.0f, 1.0f};

    private final Color rangeColor = Color.RED;
    private final Color[] gradientColors = {Color.WHITE, MainWindow.hicMapColor};
    private final float[] fractions = {0.0f, 1.0f};

    private final Color[] gradientColorsOE = {Color.BLUE, Color.WHITE, Color.RED};
    private final float[] fractionsOE = {0.0f, 0.5f, 1.0f};
    private final int oeColorMax;
    private Color[] gradientColorsPreDef =
            {new Color(255, 242, 255),
                    new Color(255, 230, 242),
                    new Color(255, 222, 230),
                    new Color(250, 218, 234),
                    new Color(255, 206, 226),
                    new Color(238, 198, 210),
                    new Color(222, 186, 182),
                    new Color(226, 174, 165),
                    new Color(214, 157, 145),
                    new Color(194, 141, 125),
                    new Color(218, 157, 121),
                    new Color(234, 182, 129),
                    new Color(242, 206, 133),
                    new Color(238, 222, 153),
                    new Color(242, 238, 161),
                    new Color(222, 238, 161),
                    new Color(202, 226, 149),
                    new Color(178, 214, 117),
                    new Color(149, 190, 113),
                    new Color(117, 170, 101),
                    new Color(113, 153, 89),
                    new Color(18, 129, 242),
                    new Color(255, 0, 0)
            };
    //            {new Color(18, 129, 242),
//            new Color(113, 153, 89),
//            new Color(117, 170, 101),
//            new Color(149, 190, 113),
//            new Color(178, 214, 117),
//            new Color(202, 226, 149),
//            new Color(222, 238, 161),
//            new Color(242, 238, 161),
//            new Color(238, 222, 153),
//            new Color(242, 206, 133),
//            new Color(234, 182, 129),
//            new Color(218, 157, 121),
//            new Color(194, 141, 125),
//            new Color(214, 157, 145),
//            new Color(226, 174, 165),
//            new Color(222, 186, 182),
//            new Color(238, 198, 210),
//            new Color(255, 206, 226),
//            new Color(250, 218, 234),
//            new Color(255, 222, 230),
//            new Color(255, 230, 242),
//            new Color(255, 242, 255),
//            new Color(255,0,0)};
    private float[] fractionsPreDef =
            {0.0f,
                    0.15f,
                    0.2f,
                    0.25f,
                    0.3f,
                    0.325f,
                    0.35f,
                    0.375f,
                    0.4f,
                    0.425f,
                    0.45f,
                    0.5f,
                    0.55f,
                    0.6f,
                    0.65f,
                    0.7f,
                    0.75f,
                    0.8f,
                    0.85f,
                    0.9f,
                    0.925f,
                    0.95f,
                    1.0f};
    private boolean colorIsOE = false;
    private boolean colorIsBlank = false;
    private boolean colorIsPreDef = false;
    private int preDefColorMax;

    /**
     * Location and size of thumb for upper value.
     */
    private Rectangle upperThumbRect;
    /**
     * Indicator that determines whether upper thumb is selected.
     */
    private boolean upperThumbSelected;

    /**
     * Indicator that determines whether lower thumb is being dragged.
     */
    private transient boolean lowerDragging;
    /**
     * Indicator that determines whether upper thumb is being dragged.
     */
    private transient boolean upperDragging;

    /**
     * Constructs a RangeSliderUI for the specified slider component.
     *
     * @param b RangeSlider
     */
    public RangeSliderUI(RangeSlider b) {
        super(b);
        oeColorMax = OEColorScale.defaultMaxOEVal;
    }

    /**
     * Installs this UI delegate on the specified component.
     */
    @Override
    public void installUI(JComponent c) {
        upperThumbRect = new Rectangle();
        super.installUI(c);
    }

    /**
     * Creates a listener to handle track events in the specified slider.
     */
    @Override
    protected TrackListener createTrackListener(JSlider slider) {
        return new RangeTrackListener();
    }

    /**
     * Creates a listener to handle change events in the specified slider.
     */
    @Override
    protected ChangeListener createChangeListener(JSlider slider) {
        return new ChangeHandler();
    }

    /**
     * Updates the dimensions for both thumbs.
     */
    @Override
    protected void calculateThumbSize() {
        // Call superclass method for lower thumb size.
        super.calculateThumbSize();

        // Set upper thumb size.
        upperThumbRect.setSize(thumbRect.width, thumbRect.height);
    }

    /**
     * Updates the locations for both thumbs.
     */
    @Override
    protected void calculateThumbLocation() {
        // Call superclass method for lower thumb location.
        super.calculateThumbLocation();

        // Adjust upper value to snap to ticks if necessary.
        if (slider.getSnapToTicks()) {
            int upperValue = slider.getValue() + slider.getExtent();
            int snappedValue = upperValue;
            int majorTickSpacing = slider.getMajorTickSpacing();
            int minorTickSpacing = slider.getMinorTickSpacing();
            int tickSpacing = 0;

            if (minorTickSpacing > 0) {
                tickSpacing = minorTickSpacing;
            } else if (majorTickSpacing > 0) {
                tickSpacing = majorTickSpacing;
            }

            if (tickSpacing != 0) {
                // If it's not on a tick, change the value
                if ((upperValue - slider.getMinimum()) % tickSpacing != 0) {
                    float temp = (float) (upperValue - slider.getMinimum()) / (float) tickSpacing;
                    int whichTick = Math.round(temp);
                    snappedValue = slider.getMinimum() + (whichTick * tickSpacing);
                }

                if (snappedValue != upperValue) {
                    slider.setExtent(snappedValue - slider.getValue());
                }
            }
        }

        // Calculate upper thumb location.  The thumb is centered over its
        // value on the track.
        if (slider.getOrientation() == JSlider.HORIZONTAL) {
            int upperPosition = xPositionForValue(slider.getValue() + slider.getExtent());
            upperThumbRect.x = upperPosition - (upperThumbRect.width / 2);
            upperThumbRect.y = trackRect.y;

        } else {
            int upperPosition = yPositionForValue(slider.getValue() + slider.getExtent());
            upperThumbRect.x = trackRect.x;
            upperThumbRect.y = upperPosition - (upperThumbRect.height / 2);
        }
    }

    /**
     * Returns the size of a thumb.
     */
    @Override
    protected Dimension getThumbSize() {
        return new Dimension(12, 12);
    }


    public String getColorsAsText() {
        String tmpStr = "";
        for (int idx = 0; idx < gradientColorsPreDef.length - 1; idx++) {
            tmpStr = tmpStr + "Color(" + gradientColorsPreDef[idx].getRed() + "," + gradientColorsPreDef[idx].getGreen() + "," + gradientColorsPreDef[idx].getBlue() + "),";
        }
        return tmpStr;
    }

    /**
     * Paints the slider.  The selected thumb is always painted on top of the
     * other thumb.
     */
    @Override
    public void paint(Graphics g, JComponent c) {
        super.paint(g, c);

        Rectangle clipRect = g.getClipBounds();
        if (upperThumbSelected) {
            // Paint lower thumb first, then upper thumb.
            if (clipRect.intersects(thumbRect)) {
                paintLowerThumb(g);
            }
            if (clipRect.intersects(upperThumbRect)) {
                paintUpperThumb(g);
            }

        } else {
            // Paint upper thumb first, then lower thumb.
            if (clipRect.intersects(upperThumbRect)) {
                paintUpperThumb(g);
            }
            if (clipRect.intersects(thumbRect)) {
                paintLowerThumb(g);
            }
        }
    }

    /**
     * Paints the track.
     */
    @Override
    public void paintTrack(Graphics g) {
        // Draw track.
        super.paintTrack(g);

        Rectangle trackBounds = trackRect;

        if (slider.getOrientation() == JSlider.HORIZONTAL) {
            // Save color
            Color oldColor = g.getColor();

            // parameters for recolored track
            int subTrackWidth = upperThumbRect.x - thumbRect.x;
            int leftArrowX = thumbRect.x;
            int rightArrowX = leftArrowX + subTrackWidth;
            int leftArrowY = trackRect.y + trackRect.height / 4;
            int redTrackWidth = trackRect.x + trackRect.width - rightArrowX;

            Rectangle subRect = new Rectangle(leftArrowX, leftArrowY, subTrackWidth, trackRect.height / 2);
            Rectangle leftSide = new Rectangle(trackRect.x, leftArrowY, leftArrowX - trackRect.x, subRect.height);
            Rectangle rightSide = new Rectangle(rightArrowX, leftArrowY, redTrackWidth, subRect.height);

            Point startP = new Point(subRect.x, subRect.y);
            Point endP = new Point(subRect.x + subRect.width, subRect.y + subRect.height);

            if (colorIsBlank) {
                LinearGradientPaint gradient = new LinearGradientPaint(startP, endP, fractionsBlank, gradientColorsBlank);
                drawSubTrackRectangles((Graphics2D) g, gradient, subRect, Color.gray, leftSide, Color.gray, rightSide);
                oldColor = rangeColorBlank;
            } else if (colorIsOE) {
                LinearGradientPaint gradient = new LinearGradientPaint(startP, endP, fractionsOE, gradientColorsOE);
                drawSubTrackRectangles((Graphics2D) g, gradient, subRect, Color.BLUE, leftSide, Color.RED, rightSide);
            } else if (colorIsPreDef) {
                LinearGradientPaint gradient = new LinearGradientPaint(startP, endP, fractionsPreDef, gradientColorsPreDef);
                drawSubTrackRectangles((Graphics2D) g, gradient, subRect, gradientColorsPreDef[0], leftSide, gradientColorsPreDef[gradientColorsPreDef.length - 1], rightSide);
            } else {
                LinearGradientPaint gradient = new LinearGradientPaint(startP, endP, fractions, new Color[]{Color.WHITE, MainWindow.hicMapColor});
                drawSubTrackRectangles((Graphics2D) g, gradient, subRect, Color.WHITE, leftSide, MainWindow.hicMapColor, rightSide);
            }


            g.setColor(oldColor);

        } else {
            // Determine position of selected range by moving from the middle
            // of one thumb to the other.
            int lowerY = thumbRect.x + (thumbRect.width / 2);
            int upperY = upperThumbRect.x + (upperThumbRect.width / 2);

            // Determine track position.
            int cx = (trackBounds.width / 2) - 2;

            // Save color and shift position.
            Color oldColor = g.getColor();
            g.translate(trackBounds.x + cx, trackBounds.y);

            // TODO - this should eventually be changed to accommodate vertical gradients (see the horizontal gradients code above)
            // Draw selected range.
            g.setColor(rangeColor);
            for (int x = 0; x <= 3; x++) {
                g.drawLine(x, lowerY - trackBounds.y, x, upperY - trackBounds.y);
            }

            // Restore position and color.
            g.translate(-(trackBounds.x + cx), -trackBounds.y);
            g.setColor(oldColor);
        }
    }

    private void drawSubTrackRectangles(Graphics2D g, LinearGradientPaint gradientColor, Rectangle gradientRect,
                                        Paint leftColor, Rectangle leftRect,
                                        Paint rightColor, Rectangle rightRect) {
        g.setPaint(gradientColor);
        g.fill(gradientRect);

        g.setPaint(leftColor);
        g.fill(leftRect);

        g.setPaint(rightColor);
        g.fill(rightRect);
    }

    /**
     * Overrides superclass method to do nothing.  Thumb painting is handled
     * within the <code>paint()</code> method.
     */
    @Override
    public void paintThumb(Graphics g) {
        // Do nothing.
    }

    /**
     * Paints the thumb for the lower value using the specified graphics object.
     */
    private void paintLowerThumb(Graphics g) {
        Rectangle knobBounds = thumbRect;
        super.paintThumb(g);
    }

    /**
     * Paints the thumb for the upper value using the specified graphics object.
     */
    private void paintUpperThumb(Graphics g) {
        Rectangle tmp = thumbRect;
        thumbRect = upperThumbRect;
        super.paintThumb(g);
        thumbRect = tmp;

    }


    /**
     * Sets the location of the upper thumb, and repaints the slider.  This is
     * called when the upper thumb is dragged to repaint the slider.  The
     * <code>setThumbLocation()</code> method performs the same task for the
     * lower thumb.
     */
    private void setUpperThumbLocation(int x, int y) {
        Rectangle upperUnionRect = new Rectangle();
        upperUnionRect.setBounds(upperThumbRect);

        upperThumbRect.setLocation(x, y);

        SwingUtilities.computeUnion(upperThumbRect.x, upperThumbRect.y, upperThumbRect.width, upperThumbRect.height, upperUnionRect);
        slider.repaint(upperUnionRect.x, upperUnionRect.y, upperUnionRect.width, upperUnionRect.height);
    }

    /**
     * Moves the selected thumb in the specified direction by a block preview.
     * This method is called when the user presses the Page Up or Down keys.
     */
    public void scrollByBlock(int direction) {
        synchronized (slider) {
            int blockIncrement = (slider.getMaximum() - slider.getMinimum()) / 10;
            if (blockIncrement <= 0 && slider.getMaximum() > slider.getMinimum()) {
                blockIncrement = 1;
            }
            int delta = blockIncrement * ((direction > 0) ? POSITIVE_SCROLL : NEGATIVE_SCROLL);

            if (upperThumbSelected) {
                int oldValue = ((RangeSlider) slider).getUpperValue();
                ((RangeSlider) slider).setUpperValue(oldValue + delta);
            } else {
                int oldValue = slider.getValue();
                slider.setValue(oldValue + delta);
            }
        }
    }

    /**
     * Moves the selected thumb in the specified direction by a unit preview.
     * This method is called when the user presses one of the arrow keys.
     */
    public void scrollByUnit(int direction) {
        synchronized (slider) {
            int delta = ((direction > 0) ? POSITIVE_SCROLL : NEGATIVE_SCROLL);

            if (upperThumbSelected) {
                int oldValue = ((RangeSlider) slider).getUpperValue();
                ((RangeSlider) slider).setUpperValue(oldValue + delta);
            } else {
                int oldValue = slider.getValue();
                slider.setValue(oldValue + delta);
            }
        }
    }

    public void setDisplayToOE(boolean isOE) {
        this.colorIsOE = isOE;
    }

    public void setDisplayToPreDef(boolean isPreDef) {

        this.colorIsPreDef = isPreDef;
        if (MainViewPanel.preDefMapColorFractions.size() == 0) {
            return;
        }
        int colorArraySize = MainViewPanel.preDefMapColorFractions.size();
        gradientColorsPreDef = MainViewPanel.preDefMapColorGradient.toArray(new Color[colorArraySize]);

        fractionsPreDef = new float[colorArraySize];
        int i = 0;

        float fractionSize = 1.0f / colorArraySize;
        for (; i < colorArraySize; i++) {
            fractionsPreDef[i] = fractionSize * i;
        }

        this.preDefColorMax = PreDefColorScale.defaultMaxPreDefVal;
    }

    public void setDisplayToBlank(boolean isBlank) {

        this.colorIsBlank = isBlank;
    }


    /**
     * Listener to handle model change events.  This calculates the thumb
     * locations and repaints the slider if the value change is not caused by
     * dragging a thumb.
     */
    private class ChangeHandler implements ChangeListener {
        public void stateChanged(ChangeEvent arg0) {
            if (!lowerDragging && !upperDragging) {
                calculateThumbLocation();
                slider.repaint();
            }
        }
    }

    /**
     * Listener to handle mouse movements in the slider track.
     */
    public class RangeTrackListener extends TrackListener {

        @Override
        public void mousePressed(MouseEvent e) {
            if (!slider.isEnabled()) {
                return;
            }

            currentMouseX = e.getX();
            currentMouseY = e.getY();

            if (slider.isRequestFocusEnabled()) {
                slider.requestFocus();
            }

            // Determine which thumb is pressed.  If the upper thumb is
            // selected (last one dragged), then check its position first;
            // otherwise check the position of the lower thumb first.
            boolean lowerPressed = false;
            boolean upperPressed = false;
            if (upperThumbSelected) {
                if (upperThumbRect.contains(currentMouseX, currentMouseY)) {
                    upperPressed = true;
                } else if (thumbRect.contains(currentMouseX, currentMouseY)) {
                    lowerPressed = true;
                }
            } else {
                if (thumbRect.contains(currentMouseX, currentMouseY)) {
                    lowerPressed = true;
                } else if (upperThumbRect.contains(currentMouseX, currentMouseY)) {
                    upperPressed = true;
                }
            }

            // Handle lower thumb pressed.
            if (lowerPressed) {
                switch (slider.getOrientation()) {
                    case JSlider.VERTICAL:
                        offset = currentMouseY - thumbRect.y;
                        break;
                    case JSlider.HORIZONTAL:
                        offset = currentMouseX - thumbRect.x;
                        break;
                }
                upperThumbSelected = false;
                lowerDragging = true;
                return;
            }
            lowerDragging = false;

            // Handle upper thumb pressed.
            if (upperPressed) {
                switch (slider.getOrientation()) {
                    case JSlider.VERTICAL:
                        offset = currentMouseY - upperThumbRect.y;
                        break;
                    case JSlider.HORIZONTAL:
                        offset = currentMouseX - upperThumbRect.x;
                        break;
                }
                upperThumbSelected = true;
                upperDragging = true;
                return;
            }
            upperDragging = false;
        }

        @Override
        public void mouseReleased(MouseEvent e) {
            if (!slider.isEnabled()) {
                return;
            }
            slider.setValueIsAdjusting(false);
            lowerDragging = false;
            upperDragging = false;
            //super.mouseReleased(e);
        }

        @Override
        public void mouseDragged(MouseEvent e) {

            if (!slider.isEnabled()) {
                return;
            }

            currentMouseX = e.getX();
            currentMouseY = e.getY();

            if (lowerDragging) {
                slider.setValueIsAdjusting(true);
                moveLowerThumb();

            } else if (upperDragging) {
                slider.setValueIsAdjusting(true);
                moveUpperThumb();
            }
        }

        @Override
        public boolean shouldScroll(int direction) {
            return false;
        }

        /**
         * Moves the location of the lower thumb, and sets its corresponding
         * value in the slider.
         */
        private void moveLowerThumb() {
            int thumbMiddle = 0;

            switch (slider.getOrientation()) {
                case JSlider.VERTICAL:
                    int halfThumbHeight = thumbRect.height / 2;
                    int thumbTop = currentMouseY - offset;
                    int trackTop = trackRect.y;
                    int trackBottom = trackRect.y + (trackRect.height - 1);
                    int vMax = yPositionForValue(slider.getValue() + slider.getExtent());

                    // Apply bounds to thumb position.
                    if (drawInverted()) {
                        trackBottom = vMax;
                    } else {
                        trackTop = vMax;
                    }
                    thumbTop = Math.max(thumbTop, trackTop - halfThumbHeight);
                    thumbTop = Math.min(thumbTop, trackBottom - halfThumbHeight);

                    setThumbLocation(thumbRect.x, thumbTop);

                    // Update slider value.
                    thumbMiddle = thumbTop + halfThumbHeight;
                    slider.setValue(valueForYPosition(thumbMiddle));
                    break;

                case JSlider.HORIZONTAL:
                    int halfThumbWidth = thumbRect.width / 2;
                    int thumbLeft = currentMouseX - offset;
                    int trackLeft = trackRect.x;
                    int trackRight = trackRect.x + (trackRect.width - 1);
                    int hMax = xPositionForValue(slider.getValue() + slider.getExtent());

                    thumbLeft = Math.max(thumbLeft, trackLeft - halfThumbWidth);
                    thumbLeft = Math.min(thumbLeft, trackRight - halfThumbWidth);

                    // Apply bounds to thumb position.
                    if (drawInverted()) {
                        thumbLeft = Math.max(thumbLeft, hMax);
                    } else {
                        thumbLeft = Math.min(thumbLeft, hMax);
                    }

                    if (colorIsOE) {
                        int midpoint = (trackRight - trackLeft) / 2 + trackLeft;
                        thumbLeft = Math.min(thumbLeft, midpoint - halfThumbWidth);
                    }

                    setThumbLocation(thumbLeft, thumbRect.y);

                    // Update slider value.
                    thumbMiddle = thumbLeft + halfThumbWidth;
                    slider.setValue(valueForXPosition(thumbMiddle));

                    if (colorIsOE) {
                        int val = ((RangeSlider) slider).getLowerValue();
                        if (val == 0) {
                            val = -1;
                            ((RangeSlider) slider).setLowerValue(val);
                        }
                        ((RangeSlider) slider).setUpperValue(-val);
                    }

                    break;

                default:
            }
        }

        /**
         * Moves the location of the upper thumb, and sets its corresponding
         * value in the slider.
         */
        private void moveUpperThumb() {
            int thumbMiddle = 0;

            switch (slider.getOrientation()) {
                case JSlider.VERTICAL:
                    int halfThumbHeight = thumbRect.height / 2;
                    int thumbTop = currentMouseY - offset;
                    int trackTop = trackRect.y;
                    int trackBottom = trackRect.y + (trackRect.height - 1);
                    int vMin = yPositionForValue(slider.getValue());

                    // Apply bounds to thumb position.
                    if (drawInverted()) {
                        trackTop = vMin;
                    } else {
                        trackBottom = vMin;
                    }
                    thumbTop = Math.max(thumbTop, trackTop - halfThumbHeight);
                    thumbTop = Math.min(thumbTop, trackBottom - halfThumbHeight);

                    setUpperThumbLocation(thumbRect.x, thumbTop);

                    // Update slider extent.
                    thumbMiddle = thumbTop + halfThumbHeight;
                    slider.setExtent(valueForYPosition(thumbMiddle) - slider.getValue());
                    break;

                case JSlider.HORIZONTAL:
                    int halfThumbWidth = thumbRect.width / 2;
                    int thumbLeft = currentMouseX - offset;
                    int trackLeft = trackRect.x;
                    int trackRight = trackRect.x + (trackRect.width - 1);
                    int hMin = xPositionForValue(slider.getValue());

                    // Apply bounds to thumb position.
                    if (drawInverted()) {
                        trackRight = hMin;
                    } else {
                        trackLeft = hMin;
                    }
                    thumbLeft = Math.max(thumbLeft, trackLeft - halfThumbWidth);
                    thumbLeft = Math.min(thumbLeft, trackRight - halfThumbWidth);

                    if (colorIsOE) {
                        int midpoint = (trackRight - trackLeft) / 2 + trackLeft;
                        thumbLeft = Math.max(thumbLeft, midpoint - halfThumbWidth);
                    }

                    setUpperThumbLocation(thumbLeft, thumbRect.y);

                    // Update slider extent.
                    thumbMiddle = thumbLeft + halfThumbWidth;
                    slider.setExtent(valueForXPosition(thumbMiddle) - slider.getValue());

                    if (colorIsOE) {
                        int val = ((RangeSlider) slider).getUpperValue();
                        if (val == 0) {
                            val = 1;
                            ((RangeSlider) slider).setUpperValue(val);
                        }
                        ((RangeSlider) slider).setLowerValue(-val);
                    }

                    break;

                default:
            }
        }
    }
}