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

package juicebox.windowui;

import juicebox.gui.SuperAdapter;
import juicebox.track.feature.AnnotationLayerHandler;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.List;


/**
 * Created by ranganmostofa on 8/16/17.
 */
public class MiniAnnotationsLayerPanel extends JPanel {

    private static final long serialVersionUID = 126735123117L;

    private int miniButtonSize = 22;

    private int horizontalBorderSize = 5;
    private int verticalBorderSize = 5;

    private int maximumVisibleLayers = 5;

    private int dynamicHeight;

    private Color backgroundColor = Color.gray;

    public MiniAnnotationsLayerPanel(SuperAdapter superAdapter) {
        List<AnnotationLayerHandler> annotationLayerHandlers = superAdapter.getAllLayers();

        dynamicHeight = this.horizontalBorderSize + Math.min(annotationLayerHandlers.size(), maximumVisibleLayers) * 40
            + this.horizontalBorderSize;

        setBackground(this.backgroundColor);
        setPreferredSize(new Dimension(210, dynamicHeight));

        for (int i = annotationLayerHandlers.size() - 1; i >= 0; i--) {
            try {
                AnnotationLayerHandler handler = annotationLayerHandlers.get(i);
                JPanel panel = createLayerPanel(handler, superAdapter, this);
                add(panel);
            } catch (IOException e) {
                // 1 - i points to the same layer in the main annotations layer since the list
                // is being iterated backwards for the mini annotations layer
                System.err.println("Unable to generate mini-layer panel " + (1 - i));
            }
        }
    }

    public JPanel createLayerPanel(final AnnotationLayerHandler handler, final SuperAdapter superAdapter,
                                   final JPanel miniAnnotationsLayerPanel) throws IOException {
        final JPanel parentPanel = new JPanel();
        parentPanel.setLayout(new FlowLayout());

        /* layer name */
        final JTextField nameField = new JTextField(handler.getLayerName(), 10);
        nameField.getDocument().addDocumentListener(anyTextChangeListener(superAdapter, handler, nameField));
        nameField.setToolTipText("Change the name for this layer: " + nameField.getText());
        nameField.setMaximumSize(new Dimension(20, 20));
        handler.setNameTextField(nameField);

        /* show/hide annotations for this layer */
        final JToggleButton toggleVisibleButton = createToggleIconButton("/images/layer/eye_clicked_green.png",
                "/images/layer/eye_clicked.png", handler.getLayerVisibility());
        toggleVisibleButton.setSelected(handler.getLayerVisibility());
        toggleVisibleButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                handler.setLayerVisibility(toggleVisibleButton.isSelected());
                updateLayers2DPanel(superAdapter);
                superAdapter.repaint();
            }
        });
        toggleVisibleButton.setToolTipText("Toggle visibility of this layer");

        JButton upButton = createIconButton("/images/layer/up.png");
        upButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                miniAnnotationsLayerPanel.remove(parentPanel);
                int index = superAdapter.moveUpIndex(handler);
                miniAnnotationsLayerPanel.add(parentPanel, index);
                miniAnnotationsLayerPanel.revalidate();
                updateLayers2DPanel(superAdapter);
                superAdapter.repaint();
            }
        });
        upButton.setToolTipText("Move this layer up (drawing order)");

        JButton downButton = createIconButton("/images/layer/down.png");
        downButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                miniAnnotationsLayerPanel.remove(parentPanel);
                int index = superAdapter.moveDownIndex(handler);
                miniAnnotationsLayerPanel.add(parentPanel, index);
                miniAnnotationsLayerPanel.revalidate();
                updateLayers2DPanel(superAdapter);
                superAdapter.repaint();
            }
        });
        downButton.setToolTipText("Move this layer down (drawing order)");

        parentPanel.add(nameField);
        Component[] allComponents = new Component[]{toggleVisibleButton, upButton, downButton};
        for (Component component : allComponents) {
            if (component instanceof AbstractButton) {
                component.setMaximumSize(new Dimension(miniButtonSize, miniButtonSize));
            }
            parentPanel.add(component);
        }
        parentPanel.setSize(new Dimension(210, 10));
        return parentPanel;
    }

    private DocumentListener anyTextChangeListener(final SuperAdapter superAdapter,
                                                   final AnnotationLayerHandler handler, final JTextField nameField) {
        return new DocumentListener() {
            @Override
            public void insertUpdate(DocumentEvent e) {
                handler.setLayerName(nameField.getText());
                nameField.setToolTipText("Change the name for this layer: " + nameField.getText());

//                updateLayers2DPanel(superAdapter);
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
                handler.setLayerName(nameField.getText());
                nameField.setToolTipText("Change the name for this layer: " + nameField.getText());
            }

            @Override
            public void changedUpdate(DocumentEvent e) {
                handler.setLayerName(nameField.getText());
                nameField.setToolTipText("Change the name for this layer: " + nameField.getText());

                System.out.print(handler.getLayerName());
                updateLayers2DPanel(superAdapter);
                superAdapter.repaint();
            }
        };
    }

    private JToggleButton createToggleIconButton(String url1, String url2, boolean activatedStatus) throws IOException {

        // image when button is active/selected (is the darkest shade/color)
        //BufferedImage imageActive = ImageIO.read(getClass().getResource(url1));
        ImageIcon iconActive = new ImageIcon(ImageIO.read(getClass().getResource(url1)));

        // image when button is inactive/transitioning (lighter shade/color)
        ImageIcon iconTransitionDown = new ImageIcon(translucentImage(ImageIO.read(getClass().getResource(url2)), 0.6f));
        ImageIcon iconTransitionUp = new ImageIcon(translucentImage(ImageIO.read(getClass().getResource(url1)), 0.6f));
        ImageIcon iconInactive = new ImageIcon(translucentImage(ImageIO.read(getClass().getResource(url2)), 0.2f));
        ImageIcon iconDisabled = new ImageIcon(translucentImage(ImageIO.read(getClass().getResource(url2)), 0.1f));

        JToggleButton toggleButton = new JToggleButton(iconInactive);
        toggleButton.setRolloverIcon(iconTransitionDown);
        toggleButton.setPressedIcon(iconDisabled);
        toggleButton.setSelectedIcon(iconActive);
        toggleButton.setRolloverSelectedIcon(iconTransitionUp);
        toggleButton.setDisabledIcon(iconDisabled);
        toggleButton.setDisabledSelectedIcon(iconDisabled);

        toggleButton.setBorderPainted(false);
        toggleButton.setSelected(activatedStatus);
        toggleButton.setPreferredSize(new Dimension(miniButtonSize, miniButtonSize));

        return toggleButton;
    }

    private Image translucentImage(BufferedImage originalImage, float alpha) {

        int width = originalImage.getWidth(), height = originalImage.getHeight();

        BufferedImage newImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = newImage.createGraphics();
        g.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, alpha));
        g.drawImage(originalImage, 0, 0, width, height, null);
        g.dispose();

        return newImage;
    }

    private JButton createIconButton(String url) throws IOException {
        BufferedImage imageActive = ImageIO.read(getClass().getResource(url));
        ImageIcon iconActive = new ImageIcon(imageActive);

        // image when button is inactive/transitioning (lighter shade/color)
        ImageIcon iconTransition = new ImageIcon(translucentImage(imageActive, 0.6f));
        ImageIcon iconInactive = new ImageIcon(translucentImage(imageActive, 0.2f));

        JButton button = new JButton(iconActive);
        button.setRolloverIcon(iconTransition);
        button.setPressedIcon(iconInactive);
        button.setBorderPainted(false);
        button.setPreferredSize(new Dimension(miniButtonSize, miniButtonSize));
        return button;
    }

    private void updateLayers2DPanel(SuperAdapter superAdapter) {
        superAdapter.getLayersPanel().updateLayers2DPanel(superAdapter);
    }

    public int getDynamicHeight() {
        return this.dynamicHeight;
    }

    private int getHorizontalBorderSize() {
        return this.horizontalBorderSize;
    }

    private int getVerticalBorderSize() {
        return this.verticalBorderSize;
    }

    private int getMiniButtonSize() {
        return this.miniButtonSize;
    }

    private int getMaximumVisibleLayers() {
        return this.maximumVisibleLayers;
    }

    private int getCurrentHeight() {
        return this.dynamicHeight;
    }

    private Color getBackgroundColor() {
        return this.backgroundColor;
    }
}
