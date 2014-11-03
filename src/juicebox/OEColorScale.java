package juicebox;

import org.broad.igv.renderer.ColorScale;

import java.awt.*;

/**
 * @author jrobinso
 *         Date: 11/11/12
 *         Time: 11:32 PM
 */
public class OEColorScale implements ColorScale {

    private double max;

    public OEColorScale() {
        super();
        double m = 5;
        max = Math.log(m);
    }

    public Color getColor(float score) {
/*
        int R = (int) (255 * Math.min(score/max, 1));
        int G = 0;
        int B = (int) (255 * Math.min(min * (1.0/score), 1));
  */
        double value = Math.log(score);
        int R,G,B;
        if (value > 0) {
            R = 255;
            value = Math.min(value, max);
            G = (int) (255 * (max-value)/max);
            B = (int) (255 * (max-value)/max);
        }
        else {
            value = -value;
            value = Math.min(value, max);
            B = 255;
            R = (int) (255 * (max-value)/max);
            G = (int) (255 * (max-value)/max);

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
        this.max = Math.log(max);
    }
}
