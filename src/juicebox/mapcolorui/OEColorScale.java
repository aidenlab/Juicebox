package juicebox.mapcolorui;

import org.broad.igv.renderer.ColorScale;

import java.awt.*;

/**
 * @author jrobinso
 *         Date: 11/11/12
 *         Time: 11:32 PM
 */
public class OEColorScale implements ColorScale {

    public static int defaultMaxOEVal = 5;
    private static double max;

    public OEColorScale() {
        super();
        resetMax();
    }

    public static void resetMax() {
        max = Math.log(defaultMaxOEVal);
    }

    public Color getColor(float score) {
/*
        int R = (int) (255 * Math.min(score/max, 1));
        int G = 0;
        int B = (int) (255 * Math.min(min * (1.0/score), 1));
  */
        double value = Math.log(score);
        int R, G, B;
        if (value > 0) {
            R = 255;
            value = Math.min(value, max);
            G = (int) (255 * (max - value) / max);
            B = (int) (255 * (max - value) / max);
        } else {
            value = -value;
            value = Math.min(value, max);
            B = 255;
            R = (int) (255 * (max - value) / max);
            G = (int) (255 * (max - value) / max);

        }

        return new Color(R, G, B);

    }

    public Color getColor(String symbol) {
        return null;  //To change body of implemented methods use File | Settings | File Templates.
    }

    public Color getNoDataColor() {
        return null;  //To change body of implemented methods use File | Settings | File Templates.
    }

    public String asString() {
        return null;  //To change body of implemented methods use File | Settings | File Templates.
    }

    public boolean isDefault() {
        return false;  //To change body of implemented methods use File | Settings | File Templates.
    }

    public void setMax(double max) {
        OEColorScale.max = Math.log(max);
    }
}
