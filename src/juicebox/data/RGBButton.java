/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2017 Broad Institute, Aiden Lab
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

package juicebox.data;

import juicebox.windowui.layers.LayerPanelButtons;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.IOException;

public class RGBButton extends JButton {

    private static final long serialVersionUID = 123123123L;
    private final ImageIcon icon1, icon2, icon3, icon4,
            iconTrans1, iconTrans2, iconTrans3, iconTrans4;
    private final String filename;
    private Channel channel = Channel.RED;

    public RGBButton(String filename, Channel initChannel, final DatasetReaderV2 reader) throws IOException {
        super();

        this.filename = filename;
        setToolTipText("Click to toggle color channel");

        // full
        String url1 = "/images/rgbtoggle/red.png";
        String url2 = "/images/rgbtoggle/green.png";
        String url3 = "/images/rgbtoggle/blue.png";
        String url4 = "/images/rgbtoggle/none.png";

        BufferedImage image1 = ImageIO.read(getClass().getResource(url1));
        BufferedImage image2 = ImageIO.read(getClass().getResource(url2));
        BufferedImage image3 = ImageIO.read(getClass().getResource(url3));
        BufferedImage image4 = ImageIO.read(getClass().getResource(url4));

        icon1 = new ImageIcon(image1);
        icon2 = new ImageIcon(image2);
        icon3 = new ImageIcon(image3);
        icon4 = new ImageIcon(image4);

        iconTrans1 = new ImageIcon(LayerPanelButtons.translucentImage(image1, 0.5f));
        iconTrans2 = new ImageIcon(LayerPanelButtons.translucentImage(image2, 0.5f));
        iconTrans3 = new ImageIcon(LayerPanelButtons.translucentImage(image3, 0.5f));
        iconTrans4 = new ImageIcon(LayerPanelButtons.translucentImage(image4, 0.5f));

        setCurrentState(initChannel);
        setPreferredSize(new Dimension(10, 10));

        addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                processClick();
                reader.setActiveChannel(channel);
            }
        });
    }

    public static int toChannelIndex(Channel channel) {
        switch (channel) {
            case RED:
                return 0;
            case GREEN:
                return 1;
            case BLUE:
                return 2;
        }
        return -1;
    }

    private void processClick() {
        switch (channel) {
            case RED:
                setCurrentState(Channel.GREEN);
                break;
            case GREEN:
                setCurrentState(Channel.BLUE);
                break;
            case BLUE:
                setCurrentState(Channel.NONE);
                break;
            case NONE:
                setCurrentState(Channel.RED);
                break;
        }
    }

    public String getFilename() {
        return filename;
    }

    public void setCurrentState(Channel state) {
        channel = state;
        switch (state) {
            case RED:
                setIcon(icon1);
                setRolloverIcon(iconTrans2);
                setPressedIcon(iconTrans2);
                break;
            case GREEN:
                setIcon(icon2);
                setRolloverIcon(iconTrans3);
                setPressedIcon(iconTrans3);
                break;
            case BLUE:
                setIcon(icon3);
                setRolloverIcon(iconTrans4);
                setPressedIcon(iconTrans4);
                break;
            case NONE:
                setIcon(icon4);
                setRolloverIcon(iconTrans1);
                setPressedIcon(iconTrans1);
                break;
        }
    }

    public enum Channel {RED, GREEN, BLUE, NONE}
}

