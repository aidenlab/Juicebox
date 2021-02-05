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

import juicebox.gui.SuperAdapter;
import juicebox.track.feature.AnnotationLayerHandler;
import org.broad.igv.ui.color.ColorChooserPanel;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;


/**
 * Created by ranganmostofa on 8/16/17.
 */
public class MiniAnnotationsLayerPanel extends JPanel {

    private static final long serialVersionUID = 9000049;
    private static final int MAX_NUM_LETTERS = 8;
    private final int miniButtonSize = 22;
    private final int maximumVisibleLayers = 5;
    private final int indivRowSize;
    private final int width, maxHeight;
    private int dynamicHeight;


    public MiniAnnotationsLayerPanel(SuperAdapter superAdapter, int width, int height) {
        this.width = width;
        this.maxHeight = height;
        indivRowSize = height / maximumVisibleLayers;

        //getRootPane().setGlassPane(disabledGlassPane);
        setMaximumSize(new Dimension(width, maxHeight));
        setLayout(new GridLayout(0, 1));
        setRows(superAdapter);

    }

    /**
     * Return a string with a maximum length.
     * If there are more characters, then string ends with an ellipsis ("...").
     *
     * @param text
     * @return shortened text
     */
    public static String shortenedName(final String text) {
        // The letters [iIl1] are slim enough to only count as half a character.
        double length = MAX_NUM_LETTERS + Math.ceil(text.replaceAll("[^iIl]", "").length() / 2.0d);

        if (text.length() > length) {
            return text.substring(0, MAX_NUM_LETTERS - 3) + "...";
        }

        return text;
    }

    private void setRows(SuperAdapter superAdapter) {

        dynamicHeight = Math.min(superAdapter.getAllLayers().size(), maximumVisibleLayers) * indivRowSize;
        setMaximumSize(new Dimension(width, dynamicHeight));
        setPreferredSize(new Dimension(width, dynamicHeight));
        setMinimumSize(new Dimension(width, dynamicHeight));

        JPanel jj = new JPanel();
        jj.setMaximumSize(new Dimension(width, maxHeight));
        jj.setLayout(new GridLayout(0, 1));

        for (AnnotationLayerHandler handler : superAdapter.getAllLayers()) {
            try {
                JPanel panel = createMiniLayerPanel(handler, superAdapter);
                jj.add(panel, 0);
            } catch (IOException e) {
                System.err.println("Unable to generate layer panel " + handler);
            }
        }
        JScrollPane scrollPane = new JScrollPane(jj,
                JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED,
                JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);
        scrollPane.setMaximumSize(new Dimension(width, dynamicHeight));
        scrollPane.setPreferredSize(new Dimension(width, dynamicHeight));
        scrollPane.setMinimumSize(new Dimension(width, dynamicHeight));
        add(scrollPane);
    }

    private JPanel createMiniLayerPanel(final AnnotationLayerHandler handler, final SuperAdapter superAdapter) throws IOException {
        final JPanel parentPanel = new JPanel();
        parentPanel.setLayout(new GridLayout(1, 0));
        parentPanel.setSize(new Dimension(width, 10));

        /* layer name */
        JLabel nameField = new JLabel(shortenedName(handler.getLayerName()));
        handler.setMiniNameLabelField(nameField);
        nameField.setToolTipText(handler.getLayerName());
        parentPanel.add(nameField);

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

        JToggleButton writeButton = LayerPanelButtons.createWritingButton(this, superAdapter, handler);
        writeButton.setMaximumSize(new Dimension(miniButtonSize, miniButtonSize));
        parentPanel.add(writeButton);

        return parentPanel;
    }

    public int getDynamicHeight() {
        return this.dynamicHeight;
    }

    public void updateRows(SuperAdapter superAdapter) {
        removeAll();
        setRows(superAdapter);
    }
}
