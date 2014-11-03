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
    private double min;

    public OEColorScale() {
        super();
        max = 5;
        min = 1/max;
    }

    public Color getColor(float score) {

        int R = (int) (255 * Math.min(score/max, 1));
        int G = 0;
        int B = (int) (255 * Math.min(min * (1.0/score), 1));
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
        this.max = max;
        this.min = 1/max;
    }
}
