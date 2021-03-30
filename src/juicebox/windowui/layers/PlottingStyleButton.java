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

package juicebox.windowui.layers;

import juicebox.mapcolorui.FeatureRenderer;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.IOException;

public class PlottingStyleButton extends JButton {

    private static final long serialVersionUID = 9000050;
    private final ImageIcon iconActive1, iconTransition1, iconInactive1,
            iconActive2, iconTransition2, iconInactive2, iconActive3, iconTransition3, iconInactive3;

    public PlottingStyleButton() throws IOException {
        super();

        // full
        String url1 = "/images/layer/full_clicked.png";
        BufferedImage imageActive1 = ImageIO.read(getClass().getResource(url1));
        iconActive1 = new ImageIcon(imageActive1);
        iconTransition1 = new ImageIcon(LayerPanelButtons.translucentImage(imageActive1, 0.6f));
        iconInactive1 = new ImageIcon(LayerPanelButtons.translucentImage(imageActive1, 0.2f));

        // ll
        String url2 = "/images/layer/ll_clicked.png";
        BufferedImage imageActive2 = ImageIO.read(getClass().getResource(url2));
        iconActive2 = new ImageIcon(imageActive2);
        iconTransition2 = new ImageIcon(LayerPanelButtons.translucentImage(imageActive2, 0.6f));
        iconInactive2 = new ImageIcon(LayerPanelButtons.translucentImage(imageActive2, 0.2f));

        // ur
        String url3 = "/images/layer/ur_clicked.png";
        BufferedImage imageActive3 = ImageIO.read(getClass().getResource(url3));
        iconActive3 = new ImageIcon(imageActive3);
        iconTransition3 = new ImageIcon(LayerPanelButtons.translucentImage(imageActive3, 0.6f));
        iconInactive3 = new ImageIcon(LayerPanelButtons.translucentImage(imageActive3, 0.2f));
    }

    public void setCurrentState(FeatureRenderer.PlottingOption state) {
        if (state == FeatureRenderer.PlottingOption.ONLY_LOWER_LEFT) {
            setIcon(iconActive2);
            setRolloverIcon(iconTransition3);
            setPressedIcon(iconActive3);
            setDisabledIcon(iconInactive2);
        } else if (state == FeatureRenderer.PlottingOption.ONLY_UPPER_RIGHT) {
            setIcon(iconActive3);
            setRolloverIcon(iconTransition1);
            setPressedIcon(iconActive1);
            setDisabledIcon(iconInactive3);
        } else if (state == FeatureRenderer.PlottingOption.EVERYTHING) {
            setIcon(iconActive1);
            setRolloverIcon(iconTransition2);
            setPressedIcon(iconActive2);
            setDisabledIcon(iconInactive1);
        }
    }
}
