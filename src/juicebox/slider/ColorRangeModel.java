package juicebox.slider;

import javax.swing.DefaultBoundedRangeModel;

/**
 * Created by nchernia on 11/3/14.
 */
public class ColorRangeModel extends DefaultBoundedRangeModel {
    boolean isObserved = true;
    int observedMin = 0;
    int observedMax = 100;
    int observedExtent = 1;
    int observedValue = 50;
    int oeMax = 5;
    int oeValue = 1;
    int oeExtent = 1;
    int oeMin = 0;

    public void setRangeProperties(int newValue, int newExtent, int newMin, int newMax, boolean adjusting) {
        super.setRangeProperties(newValue,newExtent,newMin,newMax,adjusting);
        if (isObserved) {
            observedMin = newMin;
            observedMax = newMax;
            observedExtent = newExtent;
            observedValue = newValue;
        }
        else {
            oeMin = newMin;
            oeMax = newMax;
            oeExtent = newExtent;
            oeValue = newValue;
        }
    }

    public void setObserved(boolean isObserved) {
        if (isObserved != this.isObserved) {
            this.isObserved = isObserved;
            if (isObserved) {
                setRangeProperties(observedValue, observedExtent, observedMin, observedMax, false);
            }
            else {
                setRangeProperties(oeValue, oeExtent, oeMin, oeMax, false);
            }
        }

    }
}
