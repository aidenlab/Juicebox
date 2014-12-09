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

import java.awt.*;

/**
 * @author Neva Cherniavsky
 * @since 3/22/12
 */
public class HiCColorScale implements org.broad.igv.renderer.ColorScale {

    private float min = -1f;
    private float max = 1f;

    public HiCColorScale() {
    }

    public void setMin(float min) {
        this.min = min;
    }

    public void setMax(float max) {
        this.max = max;
    }

    public Color getColor(float score) {

        if (score > 0) {
            score = score / max;
            int R = (int) (255 * Math.min(score, 1));
            int G = 0;
            int B = 0;
            return new Color(R, G, B);
        } else if (score < 0) {
            score = score / min;
            if (score < 0) score = -score; // this shouldn't happen but seems to be happening.
            int R = 0;
            int G = 0;
            int B = (int) (255 * Math.min(score, 1));
            return new Color(R, G, B);
        } else {
            // Nan ?
            return Color.black;
        }

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

}