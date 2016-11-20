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
import juicebox.mapcolorui.FeatureRenderer;
import juicebox.track.LoadAction;
import juicebox.track.feature.CustomAnnotationHandler;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.border.Border;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import java.awt.*;
import java.awt.event.ActionEvent;
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
        JButton refreshButton = new JButton("Refresh View");
        refreshButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                superAdapter.refresh();
            }
        });

        JButton newLayerButton = new JButton("Add New Layer");

        JButton mergeButton = new JButton("Merge Visible Layers");

        List<JPanel> layerPanel = new ArrayList<JPanel>();

        int i = 0;
        for (CustomAnnotationHandler handler : MainMenuBar.customAnnotationHandlers) {
            try {
                JPanel panel = createLayerPanel(handler, superAdapter);
                layerPanel.add(panel);
            } catch (IOException e) {
                System.err.println("Unable to generate layer panel " + (i - 1));
                //e.printStackTrace();
            }
        }

        JPanel buttonPanel = new JPanel();
        buttonPanel.setLayout(new GridLayout(1, 0));
        buttonPanel.add(refreshButton);
        buttonPanel.add(newLayerButton);
        buttonPanel.add(mergeButton);

        return createPane(layerPanel, buttonPanel);
    }

    private JPanel createLayerPanel(final CustomAnnotationHandler handler, final SuperAdapter superAdapter) throws IOException {
        JPanel panel = new JPanel();
        panel.setLayout(new GridLayout(1, 0));

        /* layer name */
        final JTextField nameField = new JTextField(handler.getLayerName());
        nameField.getDocument().addDocumentListener(new DocumentListener() {
            @Override
            public void insertUpdate(DocumentEvent e) {
                handler.setLayerName(nameField.getText());
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
            }

            @Override
            public void changedUpdate(DocumentEvent e) {
            }
        });
        nameField.setToolTipText("Change the name for this layer");

        /* show/hide annotations for this layer */
        final JToggleButton toggleVisibleButton = createToggleIconButton("/images/layer/eye_clicked.png", handler.getLayerVisibility());
        toggleVisibleButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                handler.setLayerVisibility(toggleVisibleButton.isSelected());
                superAdapter.repaint();
            }
        });
        toggleVisibleButton.setToolTipText("Toggle visibility of this layer");

        /* toggle transparency for this layer */
        final JToggleButton toggleTransparentButton = createToggleIconButton("/images/layer/trans_clicked.png", handler.getIsTransparent());
        toggleTransparentButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                handler.setIsTransparent(toggleTransparentButton.isSelected());
                superAdapter.repaint();
            }
        });
        toggleTransparentButton.setToolTipText("Toggle transparency of this layer");

        /* toggle whether the features will be enlarged for this layer */
        final JToggleButton toggleEnlargeButton = createToggleIconButton("/images/layer/enlarge_clicked.png", handler.getIsEnlarged());
        toggleEnlargeButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                handler.setIsEnlarged(toggleEnlargeButton.isSelected());
                superAdapter.repaint();
            }
        });
        toggleEnlargeButton.setToolTipText("Enlarge features in this layer");

        /* toggle plotting styles; setup and action done in helper function */
        JButton togglePlottingStyle = createTogglePlottingStyleIconButton(handler, superAdapter);
        togglePlottingStyle.setToolTipText("Change partial plotting style in this layer");

        /* export annotations in layer to new file */
        final JButton exportLayerButton = createIconButton("/images/layer/export_icon.png");
        exportLayerButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                handler.exportAnnotations();
            }
        });
        handler.setExportButton(exportLayerButton);
        exportLayerButton.setEnabled(handler.getExportCapability());
        exportLayerButton.setToolTipText("Export annotations from this layer");

        /* undo last annotation in layer */
        final JButton undoButton = createIconButton("/images/layer/undo.png");
        undoButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                handler.undo(undoButton);
                superAdapter.repaint();
            }
        });
        handler.setUndoButton(undoButton);
        undoButton.setEnabled(handler.getUndoCapability());
        undoButton.setToolTipText("Undo last new feature in this layer");

        /* clear annoations in this layer */
        JButton clearButton = createIconButton("/images/layer/erase.png");
        clearButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (superAdapter.clearCustomAnnotationDialog() == JOptionPane.YES_OPTION) {
                    //TODO: do something with the saving... just update temp?
                    handler.clearAnnotations();
                    handler.setExportAbility(false);
                    superAdapter.repaint();
                    handler.setExportAbility(false);
                    handler.setUndoAbility(false);
                }
            }
        });
        clearButton.setToolTipText("Clear all annotations in this layer");

        JButton importAnnotationsButton = createIconButton("/images/layer/import_icon.png");
        importAnnotationsButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                LoadAction loadAction = new LoadAction("Import 2D Annotations...", handler, superAdapter);
                loadAction.actionPerformed(e);
            }
        });
        handler.setImportAnnotationButton(importAnnotationsButton);
        importAnnotationsButton.setEnabled(handler.getImportAnnotationsEnabled());
        importAnnotationsButton.setToolTipText("Import annotations into this layer");

        JButton writeButton = createIconButton("/images/layer/write.png");
        JButton upButton = createIconButton("/images/layer/up.png");
        JButton downButton = createIconButton("/images/layer/down.png");

        JButton copyButton = createIconButton("/images/layer/copy.png");
        JButton deleteButton = createIconButton("/images/layer/trash.png");

        panel.add(writeButton);
        panel.add(nameField);
        panel.add(toggleVisibleButton);
        panel.add(toggleTransparentButton);
        panel.add(toggleEnlargeButton);
        panel.add(togglePlottingStyle);
        panel.add(undoButton);
        panel.add(clearButton);
        panel.add(importAnnotationsButton);
        panel.add(exportLayerButton);
        panel.add(copyButton);
        panel.add(deleteButton);
        panel.add(upButton);
        panel.add(downButton);

        return panel;
    }

    private JButton createTogglePlottingStyleIconButton(final CustomAnnotationHandler handler,
                                                        final SuperAdapter superAdapter) throws IOException {

        // triple state toggle button
        String url1 = "/images/layer/full_clicked.png";
        String url2 = "/images/layer/ll_clicked.png";
        String url3 = "/images/layer/ur_clicked.png";

        // full
        BufferedImage imageActive1 = ImageIO.read(getClass().getResource(url1));
        final ImageIcon iconActive1 = new ImageIcon(imageActive1);
        final ImageIcon iconTransition1 = new ImageIcon(translucentImage(imageActive1, 0.6f));
        final ImageIcon iconInactive1 = new ImageIcon(translucentImage(imageActive1, 0.2f));

        // ll
        BufferedImage imageActive2 = ImageIO.read(getClass().getResource(url2));
        final ImageIcon iconActive2 = new ImageIcon(imageActive2);
        final ImageIcon iconTransition2 = new ImageIcon(translucentImage(imageActive2, 0.6f));
        final ImageIcon iconInactive2 = new ImageIcon(translucentImage(imageActive2, 0.2f));

        // ur
        BufferedImage imageActive3 = ImageIO.read(getClass().getResource(url3));
        final ImageIcon iconActive3 = new ImageIcon(imageActive3);
        final ImageIcon iconTransition3 = new ImageIcon(translucentImage(imageActive3, 0.6f));
        final ImageIcon iconInactive3 = new ImageIcon(translucentImage(imageActive3, 0.2f));

        final JButton triStateButton = new JButton();
        triStateButton.setBorderPainted(false);
        triStateButton.addActionListener(new ActionListener() {

            private FeatureRenderer.PlottingOption currentState = handler.getPlottingStyle();

            @Override
            public void actionPerformed(ActionEvent e) {
                currentState = getNextState(currentState);
                setStateIcons(triStateButton, currentState, iconActive1, iconTransition1, iconInactive1,
                        iconActive2, iconTransition2, iconInactive2, iconActive3, iconTransition3, iconInactive3);
                handler.setPlottingStyle(currentState);
                superAdapter.repaint();
            }

        });

        setStateIcons(triStateButton, handler.getPlottingStyle(), iconActive1, iconTransition1, iconInactive1,
                iconActive2, iconTransition2, iconInactive2, iconActive3, iconTransition3, iconInactive3);

        return triStateButton;
    }

    private FeatureRenderer.PlottingOption getNextState(FeatureRenderer.PlottingOption state) {
        switch (state) {
            case ONLY_LOWER_LEFT:
                return FeatureRenderer.PlottingOption.ONLY_UPPER_RIGHT;
            case ONLY_UPPER_RIGHT:
                return FeatureRenderer.PlottingOption.EVERYTHING;
            case EVERYTHING:
                return FeatureRenderer.PlottingOption.ONLY_LOWER_LEFT;
        }
        return FeatureRenderer.PlottingOption.EVERYTHING;
    }

    private void setStateIcons(JButton triStateButton, FeatureRenderer.PlottingOption state,
                               ImageIcon iconActive1, ImageIcon iconTransition1, ImageIcon iconInactive1,
                               ImageIcon iconActive2, ImageIcon iconTransition2, ImageIcon iconInactive2,
                               ImageIcon iconActive3, ImageIcon iconTransition3, ImageIcon iconInactive3) {
        switch (state) {
            case ONLY_LOWER_LEFT:
                triStateButton.setIcon(iconActive2);
                triStateButton.setRolloverIcon(iconTransition3);
                triStateButton.setPressedIcon(iconActive3);
                triStateButton.setDisabledIcon(iconInactive2);
                break;
            case ONLY_UPPER_RIGHT:
                triStateButton.setIcon(iconActive3);
                triStateButton.setRolloverIcon(iconTransition1);
                triStateButton.setPressedIcon(iconActive1);
                triStateButton.setDisabledIcon(iconInactive3);
                break;
            case EVERYTHING:
                triStateButton.setIcon(iconActive1);
                triStateButton.setRolloverIcon(iconTransition2);
                triStateButton.setPressedIcon(iconActive2);
                triStateButton.setDisabledIcon(iconInactive1);
                break;
        }
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
    private JToggleButton createToggleIconButton(String url, boolean activatedStatus) throws IOException {

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
        toggleButton.setSelected(activatedStatus);

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
    private JPanel createPane(List<JPanel> panels, JPanel buttons) {

        JPanel box = new JPanel();
        box.setLayout(new BoxLayout(box, BoxLayout.PAGE_AXIS));
        for (JPanel panel : panels) {
            box.add(panel);
        }
        JScrollPane scrollPane = new JScrollPane(box);

        JPanel pane = new JPanel(new BorderLayout());
        pane.add(scrollPane, BorderLayout.CENTER);
        pane.add(buttons, BorderLayout.PAGE_END);

        return pane;
    }
}
