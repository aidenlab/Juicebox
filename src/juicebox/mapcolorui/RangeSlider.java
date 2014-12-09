/*
 * Copyright (C) 2011-2014 Aiden Lab - All Rights Reserved
 * SOFTWARE COPYRIGHT NOTICE
 * This software and its documentation are the copyright of the Aiden Lab All rights are reserved.
 *
 * This software is supplied without any warranty or guaranteed support whatsoever. Aiden Lab is not responsible for its use, misuse, or functionality.
 *
 * This software is licensed under the terms of the GNU Lesser General Public License (LGPL),
 * Version 2.1 which is available at http://www.opensource.org/licenses/lgpl-2.1.php.
 */

package juicebox.mapcolorui;

import javax.swing.*;
import java.io.Serializable;

/**
 * An extension of JSlider to select a range of values using two thumb controls.
 * The thumb controls are used to select the lower and upper value of a range
 * with predetermined minimum and maximum values.
 * <p/>
 * <p>Note that RangeSlider makes use of the default BoundedRangeModel, which
 * supports an inner range defined by a value and an extent.  The upper value
 * returned by RangeSlider is simply the lower value plus the extent.</p>
 *
 * @author Ernest Yu
 * @modified Jim Robinson
 * @modified Muhammad S Shamim
 */
public class RangeSlider extends JSlider implements Serializable {

    private RangeSliderUI rangeSliderUI;

    private boolean colorIsOE = false;

    /**
     * Constructs a RangeSlider with default minimum and maximum values of 0
     * and 100.
     */
    public RangeSlider() {
        initSlider();
    }

    /**
     * Constructs a RangeSlider with the specified default minimum and maximum
     * values.
     */
    public RangeSlider(int min, int max) {
        super(min, max);
        initSlider();
    }

    /**
     * Initializes the slider by setting default properties.
     */
    private void initSlider() {
        setOrientation(HORIZONTAL);
    }

    /**
     * Overrides the superclass method to install the UI delegate to draw two
     * thumbs.
     */
    @Override
    public void updateUI() {
        rangeSliderUI = new RangeSliderUI(this);
        setUI(rangeSliderUI);
        // Update UI for slider labels.  This must be called after updating the
        // UI of the slider.  Refer to JSlider.updateUI().
        updateLabelUIs();
    }


    public int getLowerValue() {
        return getValue();
    }

    public void setLowerValue(int value) {

        setValue(value);
    }

    /**
     * Sets the lower value in the range.
     */
    @Override
    public void setValue(int value) {
        int oldValue = getValue();
        if (oldValue == value) {
            return;
        }

        // Compute new value and extent to maintain upper value.
        int oldExtent = getExtent();
        int newValue = Math.min(Math.max(getMinimum(), value), oldValue + oldExtent);
        int newExtent = oldExtent + oldValue - newValue;

        // Set new value and extent, and fire a single change event.
        getModel().setRangeProperties(newValue, newExtent, getMinimum(),
                getMaximum(), getValueIsAdjusting());
    }

    /**
     * Returns the upper value in the range.
     */
    public int getUpperValue() {
        return getValue() + getExtent();
    }

    /**
     * Sets the upper value in the range.
     */
    public void setUpperValue(int value) {
        // Compute new extent.
        int lowerValue = getValue();
        int newExtent = Math.min(Math.max(0, value - lowerValue), getMaximum() - lowerValue);

        // Set extent to set upper value.
        setExtent(newExtent);
    }

    public void setDisplayToOE(boolean colorIsOE) {
        this.colorIsOE = colorIsOE;
        rangeSliderUI.setDisplayToOE(colorIsOE);
    }

}