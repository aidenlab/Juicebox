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

package juicebox.windowui.layers;

import juicebox.gui.SuperAdapter;
import juicebox.track.feature.AnnotationLayerHandler;
import juicebox.windowui.DisabledGlassPane;
import org.broad.igv.ui.color.ColorChooserPanel;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;


/**
 * Created by ranganmostofa on 8/16/17.
 */
public class MiniAnnotationsLayerPanel extends JPanel {

    public static final DisabledGlassPane disabledGlassPane = new DisabledGlassPane(Cursor.WAIT_CURSOR);
    private static final long serialVersionUID = 126735123117L;
    private final int miniButtonSize = 22;
    private final int maximumVisibleLayers = 5;
    private final int dynamicHeight;
    private final int width = 210;


    public MiniAnnotationsLayerPanel(SuperAdapter superAdapter) {
        //getRootPane().setGlassPane(disabledGlassPane);
        dynamicHeight = Math.min(superAdapter.getAllLayers().size(), maximumVisibleLayers) * 40;
        setPreferredSize(new Dimension(width, dynamicHeight));
        setLayout(new GridLayout(0, 1));

        for (AnnotationLayerHandler handler : superAdapter.getAllLayers()) {
            try {
                JPanel panel = createMiniLayerPanel(handler, superAdapter);
                add(panel, 0);
            } catch (IOException e) {
                System.err.println("Unable to generate layer panel " + handler);
            }
        }
    }


    private JPanel createMiniLayerPanel(final AnnotationLayerHandler handler, final SuperAdapter superAdapter) throws IOException {
        final JPanel parentPanel = new JPanel();
        parentPanel.setLayout(new GridLayout(1, 0));
        parentPanel.setSize(new Dimension(width, 10));

        /* layer name */
        JLabel nameField = new JLabel(handler.getLayerName());
        handler.setMiniNameLabelField(nameField);
        nameField.setToolTipText(handler.getLayerName());
        parentPanel.add(nameField);

        JToggleButton writeButton = LayerPanelButtons.createWritingButton(this, superAdapter, handler);
        writeButton.setMaximumSize(new Dimension(miniButtonSize, miniButtonSize));
        parentPanel.add(writeButton);

        /* show/hide annotations for this layer */
        JToggleButton toggleVisibleButton = LayerPanelButtons.createVisibleButton(this, superAdapter, handler);
        toggleVisibleButton.setMaximumSize(new Dimension(miniButtonSize, miniButtonSize));
        parentPanel.add(toggleVisibleButton);

        ColorChooserPanel colorChooserPanel = LayerPanelButtons.createColorChooserButton(superAdapter, handler);
        colorChooserPanel.setMaximumSize(new Dimension(miniButtonSize, miniButtonSize));
        parentPanel.add(colorChooserPanel);

        JButton togglePlottingStyleButton = LayerPanelButtons.createTogglePlottingStyleButton(superAdapter, handler);
        togglePlottingStyleButton.setMaximumSize(new Dimension(miniButtonSize, miniButtonSize));
        parentPanel.add(togglePlottingStyleButton);

        return parentPanel;
    }

    public int getDynamicHeight() {
        return this.dynamicHeight;
    }
}
