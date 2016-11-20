/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2016 Broad Institute, Aiden Lab
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

package juicebox.tools.dev;

import juicebox.gui.MainMenuBar;
import juicebox.gui.SuperAdapter;
import juicebox.track.feature.CustomAnnotationHandler;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.border.Border;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import java.awt.*;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by muhammadsaadshamim on 8/4/16.
 */
public class LayersPanel extends JPanel {

    private static final long serialVersionUID = 81248921738L;
    private final List<ActionListener> actionListeners = new ArrayList<ActionListener>();

    public LayersPanel(SuperAdapter superAdapter) {
        super(new BorderLayout());

        Border padding = BorderFactory.createEmptyBorder(20, 20, 5, 20);

        JPanel layersPanel = generateLayerSelectionPanel(superAdapter);
        if (layersPanel != null) {
            layersPanel.setBorder(padding);
        }

        add(layersPanel, BorderLayout.CENTER);
    }

    /**
     * @param superAdapter
     */
    public static void launchLayersGUI(SuperAdapter superAdapter) {
        JFrame frame = new JFrame("Layer Panel");
        LayersPanel newContentPane = new LayersPanel(superAdapter);
        newContentPane.setOpaque(true);
        frame.setContentPane(newContentPane);
        frame.pack();
        frame.setVisible(true);
    }

    /**
     *
     */
    private JPanel generateLayerSelectionPanel(final SuperAdapter superAdapter) {
        final JButton showItButton = new JButton("Update View");

        List<JPanel> layerPanel = new ArrayList<JPanel>();

        int i = 0;
        for (CustomAnnotationHandler handler : MainMenuBar.customAnnotationHandlers) {
            try {
                JPanel panel = createLayerPanel(handler);
                layerPanel.add(panel);
            } catch (IOException e) {
                System.err.println("Unable to generate layer panel " + (i - 1));
                //e.printStackTrace();
            }
        }

        return createPane(layerPanel, showItButton);
    }

    private JPanel createLayerPanel(final CustomAnnotationHandler handler) throws IOException {
        JPanel panel = new JPanel();
        panel.setLayout(new GridLayout(1, 0));

        final JTextField nameField = new JTextField(handler.getLayerName());
        nameField.getDocument().addDocumentListener(new DocumentListener() {
            @Override
            public void insertUpdate(DocumentEvent e) {
                handler.setLayerName(nameField.getText());
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
                //handler.setLayerName(nameField.getText());

            }

            @Override
            public void changedUpdate(DocumentEvent e) {
                //handler.setLayerName(nameField.getText());
            }
        });

        JToggleButton toggleVisibleButton = createToggleIconButton("/images/layer/eye_clicked.png");
        JToggleButton toggleTransparentButton = createToggleIconButton("/images/layer/trans_clicked.png");
        JToggleButton toggleEnlargeButton = createToggleIconButton("/images/layer/enlarge_clicked.png");
        JToggleButton toggleLLButton = createToggleIconButton("/images/layer/ll_clicked.png");
        JToggleButton toggleURButton = createToggleIconButton("/images/layer/ur_clicked.png");

        JButton addAnnotationsButton = createIconButton("/images/layer/add_icon.png");
        JButton upButton = createIconButton("/images/layer/up.png");
        JButton downButton = createIconButton("/images/layer/down.png");
        JButton exportLayerButton = createIconButton("/images/layer/export_icon.png");
        JButton deleteButton = createIconButton("/images/layer/trash.png");

        panel.add(nameField);
        panel.add(toggleVisibleButton);
        panel.add(toggleTransparentButton);
        panel.add(toggleEnlargeButton);
        panel.add(toggleLLButton);
        panel.add(toggleURButton);
        panel.add(addAnnotationsButton);
        panel.add(upButton);
        panel.add(downButton);
        panel.add(exportLayerButton);
        panel.add(deleteButton);

        return panel;
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
        return button;
    }

    /**
     * @param url
     * @return toggle button which changes icon transparency when clicked
     * @throws IOException
     */
    private JToggleButton createToggleIconButton(String url) throws IOException {

        // image when button is active/selected (is the darkest shade/color)
        BufferedImage imageActive = ImageIO.read(getClass().getResource(url));
        ImageIcon iconActive = new ImageIcon(imageActive);

        // image when button is inactive/transitioning (lighter shade/color)
        ImageIcon iconTransition = new ImageIcon(translucentImage(imageActive, 0.6f));
        ImageIcon iconInactive = new ImageIcon(translucentImage(imageActive, 0.2f));
        ImageIcon iconDisabled = new ImageIcon(translucentImage(imageActive, 0.1f));

        JToggleButton toggleButton = new JToggleButton(iconInactive);
        toggleButton.setRolloverIcon(iconTransition);
        toggleButton.setPressedIcon(iconDisabled);
        toggleButton.setSelectedIcon(iconActive);
        toggleButton.setRolloverSelectedIcon(iconTransition);
        toggleButton.setDisabledIcon(iconDisabled);
        toggleButton.setDisabledSelectedIcon(iconDisabled);

        toggleButton.setBorderPainted(false);

        return toggleButton;
    }

    /**
     * @param originalImage
     * @param alpha
     * @return original image with transparency alpha
     */
    public Image translucentImage(BufferedImage originalImage, float alpha) {

        int width = originalImage.getWidth(), height = originalImage.getHeight();

        BufferedImage newImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = newImage.createGraphics();
        g.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, alpha));
        g.drawImage(originalImage, 0, 0, width, height, null);
        g.dispose();

        return newImage;
    }

    /**
     * @return
     */
    private JPanel createPane(List<JPanel> panels, JButton showButton) {

        JPanel box = new JPanel();
        box.setLayout(new BoxLayout(box, BoxLayout.PAGE_AXIS));
        for (JPanel panel : panels) {
            box.add(panel);
        }
        JScrollPane scrollPane = new JScrollPane(box);

        JPanel pane = new JPanel(new BorderLayout());
        pane.add(scrollPane, BorderLayout.CENTER);
        pane.add(showButton, BorderLayout.PAGE_END);

        return pane;
    }
}
