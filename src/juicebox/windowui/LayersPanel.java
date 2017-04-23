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
import juicebox.mapcolorui.FeatureRenderer;
import juicebox.track.HiCTrack;
import juicebox.track.LoadAction;
import juicebox.track.LoadEncodeAction;
import juicebox.track.TrackConfigPanel;
import juicebox.track.feature.AnnotationLayerHandler;
import org.broad.igv.ui.color.ColorChooserPanel;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.border.Border;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


/**
 * Created by muhammadsaadshamim on 8/4/16.
 */
public class LayersPanel extends JDialog {

    public static final DisabledGlassPane disabledGlassPane = new DisabledGlassPane(Cursor.WAIT_CURSOR);
    private static final long serialVersionUID = 8124112892178L;
    private static final int miniButtonSize = 30;
    private static LoadAction trackLoadAction;
    private static LoadEncodeAction encodeAction;

    public LayersPanel(final SuperAdapter superAdapter) {
        super(superAdapter.getMainWindow(), "Annotations Layer Panel");
        rootPane.setGlassPane(disabledGlassPane);

        Border padding = BorderFactory.createEmptyBorder(20, 20, 5, 20);

        JPanel annotations1DPanel = generate1DAnnotationsLayerSelectionPanel(superAdapter);
        if (annotations1DPanel != null) annotations1DPanel.setBorder(padding);

        JPanel layers2DPanel = generate2DAnnotationsLayerSelectionPanel(superAdapter);
        if (layers2DPanel != null) layers2DPanel.setBorder(padding);

        JTabbedPane tabbedPane = new JTabbedPane();
        tabbedPane.addTab("1D Annotations", null, annotations1DPanel,
                "Manage 1D Annotations");
        //tabbedPane.setMnemonicAt(1, KeyEvent.VK_2);

        tabbedPane.addTab("2D Annotations", null, layers2DPanel,
                "Manage 2D Annotations");
        //tabbedPane.setMnemonicAt(0, KeyEvent.VK_1);

        setSize(800, 600);
        add(tabbedPane);
        setVisible(true);

        this.addWindowListener(new WindowListener() {
            public void windowActivated(WindowEvent e) {
            }

            public void windowClosed(WindowEvent e) {
            }

            public void windowClosing(WindowEvent e) {
                superAdapter.setLayersPanelGUIControllersSelected(false);
            }

            public void windowDeactivated(WindowEvent e) {
            }

            public void windowDeiconified(WindowEvent e) {
            }

            public void windowIconified(WindowEvent e) {
            }

            public void windowOpened(WindowEvent e) {
            }
        });
    }

    public LoadEncodeAction getEncodeAction() {
        return encodeAction;
    }

    private JPanel generate1DAnnotationsLayerSelectionPanel(final SuperAdapter superAdapter) {
        final JPanel layerBoxGUI = new JPanel();
        layerBoxGUI.setLayout(new GridLayout(0, 1));
        JScrollPane scrollPane = new JScrollPane(layerBoxGUI);

        JPanel buttonPanel = new JPanel(new GridLayout(1, 0));
        JButton loadBasicButton = new JButton("Load Basic Annotations...");
        buttonPanel.add(loadBasicButton);
        JButton loadEncodeButton = new JButton("Load ENCODE Tracks...");
        buttonPanel.add(loadEncodeButton);
        JButton loadFromURLButton = new JButton("Load from URL...");
        buttonPanel.add(loadFromURLButton);
        JButton refreshButton = new JButton("Refresh View");
        buttonPanel.add(refreshButton);

        final JPanel pane = new JPanel(new BorderLayout());
        pane.add(scrollPane, BorderLayout.CENTER);
        pane.add(buttonPanel, BorderLayout.PAGE_END);

        Dimension dim = pane.getPreferredSize();
        dim.setSize(dim.getWidth(), dim.getHeight() * 4);
        pane.setPreferredSize(dim);

        final Runnable repaint1DLayersPanel = new Runnable() {
            @Override
            public void run() {
                redraw1DLayerPanels(superAdapter, layerBoxGUI, pane);
            }
        };
        repaint1DLayersPanel.run();

        trackLoadAction = new LoadAction("Load Basic Annotations...", superAdapter.getMainWindow(),
                superAdapter.getHiC(), repaint1DLayersPanel);
        loadBasicButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                trackLoadAction.actionPerformed(e);
            }
        });

        encodeAction = new LoadEncodeAction("Load ENCODE Tracks...",
                superAdapter.getMainWindow(), superAdapter.getHiC(), repaint1DLayersPanel);

        loadEncodeButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                encodeAction.actionPerformed(e);
            }
        });

        loadFromURLButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                superAdapter.safeLoadFromURLActionPerformed(repaint1DLayersPanel);
            }
        });

        refreshButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                superAdapter.refresh();
                redraw1DLayerPanels(superAdapter, layerBoxGUI, pane);
            }
        });
        return pane;
    }

    private void redraw1DLayerPanels(SuperAdapter superAdapter, JPanel layerBoxGUI, JPanel pane) {
        layerBoxGUI.removeAll();
        for (HiCTrack track : superAdapter.getHiC().getLoadedTracks()) {
            if (track != null) {
                layerBoxGUI.add(new TrackConfigPanel(superAdapter, track));
            }
        }
        layerBoxGUI.revalidate();
        layerBoxGUI.repaint();
        pane.revalidate();
        pane.repaint();
    }

    /**
     *
     * @param superAdapter
     * @return
     */
    private JPanel generate2DAnnotationsLayerSelectionPanel(final SuperAdapter superAdapter) {
        final JPanel layerBoxGUI = new JPanel();
        //layerBoxGUI.setLayout(new BoxLayout(layerBoxGUI, BoxLayout.PAGE_AXIS));
        layerBoxGUI.setLayout(new GridLayout(0, 1));

        int i = 0;
        for (AnnotationLayerHandler handler : superAdapter.getAllLayers()) {
            try {
                JPanel panel = createLayerPanel(handler, superAdapter, layerBoxGUI);
                //layerPanels.add(panel);
                layerBoxGUI.add(panel, 0);
            } catch (IOException e) {
                System.err.println("Unable to generate layer panel " + (i - 1));
                //e.printStackTrace();
            }
        }
        final JScrollPane scrollPane = new JScrollPane(layerBoxGUI);

        JButton refreshButton = new JButton("Refresh View");
        refreshButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                superAdapter.refresh();
            }
        });

        JButton importButton = new JButton("Load Loops/Domains...");
        JButton newLayerButton = new JButton("Add New Layer");
        JButton mergeButton = new JButton("Merge Visible Layers");

        JPanel buttonPanel = new JPanel(new GridLayout(1, 0));
        buttonPanel.add(importButton);
        buttonPanel.add(newLayerButton);
        buttonPanel.add(mergeButton);
        buttonPanel.add(refreshButton);

        final JPanel pane = new JPanel(new BorderLayout());
        pane.add(scrollPane, BorderLayout.CENTER);
        pane.add(buttonPanel, BorderLayout.PAGE_END);

        Dimension dim = pane.getPreferredSize();
        dim.setSize(dim.getWidth(), dim.getHeight() * 4);
        pane.setPreferredSize(dim);

        /* import 2d annotations into layer */
        importButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                Load2DAnnotationsDialog dialog = new Load2DAnnotationsDialog(LayersPanel.this, superAdapter, layerBoxGUI);
                dialog.setVisible(true);
            }
        });
        importButton.setToolTipText("Import annotations into new layer");

        newLayerButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                new2DAnnotationsLayerAction(superAdapter, layerBoxGUI, null);
            }
        });

        mergeButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                merge2DAnnotationsAction(superAdapter, layerBoxGUI);
            }
        });

        superAdapter.updateLayerDeleteStatus();
        return pane;
    }

    public AnnotationLayerHandler new2DAnnotationsLayerAction(SuperAdapter superAdapter, JPanel layerBoxGUI,
                                                              AnnotationLayerHandler sourceHandler) {
        AnnotationLayerHandler handler = superAdapter.createNewLayer();
        if (sourceHandler != null) handler.duplicateDetailsFrom(sourceHandler);
        try {
            JPanel panel = createLayerPanel(handler, superAdapter, layerBoxGUI);
            layerBoxGUI.add(panel, 0);
            layerBoxGUI.revalidate();
            layerBoxGUI.repaint();
            superAdapter.setActiveLayer(handler);
            superAdapter.updateLayerDeleteStatus();
        } catch (Exception ee) {
            System.err.println("Unable to add new layer to GUI");
        }
        return handler;
    }

    private void merge2DAnnotationsAction(SuperAdapter superAdapter, JPanel layerBoxGUI) {
        List<AnnotationLayerHandler> visibleLayers = new ArrayList<>();
        for (AnnotationLayerHandler handler : superAdapter.getAllLayers()) {
            if (handler.getLayerVisibility()) {
                visibleLayers.add(handler);
            }
        }

        AnnotationLayerHandler mergedHandler = superAdapter.createNewLayer();
        mergedHandler.mergeDetailsFrom(visibleLayers);
        try {
            JPanel panel = createLayerPanel(mergedHandler, superAdapter, layerBoxGUI);
            layerBoxGUI.add(panel, 0);

            for (AnnotationLayerHandler handler : visibleLayers) {
                int index = superAdapter.removeLayer(handler);
                if (index > -1) {
                    layerBoxGUI.remove(index);
                }
            }

            layerBoxGUI.revalidate();
            layerBoxGUI.repaint();
            superAdapter.setActiveLayer(mergedHandler);
            superAdapter.updateLayerDeleteStatus();
        } catch (Exception ee) {
            System.err.println("Unable to add merged layer to GUI");
            ee.printStackTrace();
        }
    }

    /**
     * @param handler
     * @param superAdapter
     * @param layerBoxGUI
     * @return
     * @throws IOException
     */
    private JPanel createLayerPanel(final AnnotationLayerHandler handler, final SuperAdapter superAdapter,
                                    final JPanel layerBoxGUI) throws IOException {
        final JPanel parentPanel = new JPanel();
        parentPanel.setLayout(new FlowLayout());

        /* layer name */
        final JTextField nameField = new JTextField(handler.getLayerName(), 10);
        nameField.getDocument().addDocumentListener(anyTextChangeListener(handler, nameField));
        nameField.setToolTipText("Change the name for this layer: " + nameField.getText());
        nameField.setMaximumSize(new Dimension(100, 30));
        handler.setNameTextField(nameField);

        /* show/hide annotations for this layer */
        final JToggleButton toggleVisibleButton = createToggleIconButton("/images/layer/eye_clicked_green.png",
                "/images/layer/eye_clicked.png", handler.getLayerVisibility());
        toggleVisibleButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                handler.setLayerVisibility(toggleVisibleButton.isSelected());
                superAdapter.repaint();
            }
        });
        toggleVisibleButton.setToolTipText("Toggle visibility of this layer");

        /* toggle transparency for this layer */
        final JToggleButton toggleTransparentButton = createToggleIconButton("/images/layer/trans_clicked_green.png",
                "/images/layer/trans_clicked.png", handler.getIsTransparent());
        toggleTransparentButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                handler.setIsTransparent(toggleTransparentButton.isSelected());
                superAdapter.repaint();
            }
        });
        toggleTransparentButton.setToolTipText("Toggle transparency of this layer");


        /*  Sparse (/subset) plotting for 2d annotations  */
        final JToggleButton toggleSparseButton = createToggleIconButton("/images/layer/sparse.png", handler.getIsSparse());
        toggleSparseButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                handler.setIsSparse(toggleSparseButton.isSelected());
                superAdapter.repaint();
            }
        });
        toggleSparseButton.setToolTipText("Plot a limited number of 2D annotations in this layer at a time " +
                "(speed up plotting when there are many annotations).");

        /* toggle whether the features will be enlarged for this layer */
        final JToggleButton toggleEnlargeButton = createToggleIconButton("/images/layer/enlarge_clicked_down.png",
                "/images/layer/enlarge_clicked_up.png", handler.getIsEnlarged());
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
        handler.setPlottingStyleButton(togglePlottingStyle);

        /* export annotations in layer to new file */
        final JButton exportLayerButton = createIconButton("/images/layer/export_icon_green.png");
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

        final ColorChooserPanel colorChooserPanel = new ColorChooserPanel();
        colorChooserPanel.setSelectedColor(handler.getDefaultColor());
        colorChooserPanel.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                Color c = colorChooserPanel.getSelectedColor();
                if (c != null) {
                    handler.setColorOfAllAnnotations(c);
                    superAdapter.repaint();
                }
            }
        });
        colorChooserPanel.setToolTipText("Re-color all annotations in this layer");
        handler.setColorChooserPanel(colorChooserPanel);

        /* clear annotations in this layer */
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

        final JToggleButton writeButton = createToggleIconButton("/images/layer/pencil.png", "/images/layer/pencil_gray.png", handler.isActiveLayer(superAdapter));
        handler.setActiveLayerButton(writeButton);
        writeButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                superAdapter.setActiveLayer(handler);
            }
        });
        writeButton.setToolTipText("Enable drawing of annotations to this layer; Hold down shift key, then click and drag on map");

        JButton deleteButton = createIconButton("/images/layer/trash.png");
        deleteButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (superAdapter.deleteCustomAnnotationDialog(handler.getLayerName()) == JOptionPane.YES_OPTION) {
                    layerBoxGUI.remove(parentPanel);
                    superAdapter.removeLayer(handler);
                    layerBoxGUI.revalidate();
                    layerBoxGUI.repaint();
                    superAdapter.repaint();
                }
            }
        });
        handler.setDeleteLayerButton(deleteButton);
        deleteButton.setToolTipText("Delete this layer");

        JButton upButton = createIconButton("/images/layer/up.png");
        upButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                layerBoxGUI.remove(parentPanel);
                int index = superAdapter.moveUpIndex(handler);
                layerBoxGUI.add(parentPanel, index);
                layerBoxGUI.revalidate();
                layerBoxGUI.repaint();
                superAdapter.repaint();
            }
        });
        upButton.setToolTipText("Move this layer up (drawing order)");

        JButton downButton = createIconButton("/images/layer/down.png");
        downButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                layerBoxGUI.remove(parentPanel);
                int index = superAdapter.moveDownIndex(handler);
                layerBoxGUI.add(parentPanel, index);
                layerBoxGUI.revalidate();
                layerBoxGUI.repaint();
                superAdapter.repaint();
            }
        });
        downButton.setToolTipText("Move this layer down (drawing order)");

        JButton copyButton = createIconButton("/images/layer/copy.png");
        copyButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                AnnotationLayerHandler handlerDup = new2DAnnotationsLayerAction(superAdapter, layerBoxGUI, handler);
            }
        });
        copyButton.setToolTipText("Duplicate this layer");

        parentPanel.add(nameField);
        Component[] allComponents = new Component[]{writeButton, toggleVisibleButton,
                colorChooserPanel, toggleTransparentButton, toggleEnlargeButton, togglePlottingStyle, toggleSparseButton,
                undoButton, clearButton, exportLayerButton, copyButton, upButton, downButton, deleteButton};
        for (Component component : allComponents) {
            if (component instanceof AbstractButton) {
                component.setMaximumSize(new Dimension(miniButtonSize, miniButtonSize));
            }
            parentPanel.add(component);
        }

        return parentPanel;
    }

    private DocumentListener anyTextChangeListener(final AnnotationLayerHandler handler,
                                                   final JTextField nameField) {
        return new DocumentListener() {
            @Override
            public void insertUpdate(DocumentEvent e) {
                handler.setLayerName(nameField.getText());
                nameField.setToolTipText("Change the name for this layer: " + nameField.getText());
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
            }
        };
    }

    private JButton createTogglePlottingStyleIconButton(final AnnotationLayerHandler handler,
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
        triStateButton.setPreferredSize(new Dimension(miniButtonSize, miniButtonSize));
        triStateButton.setBorderPainted(false);
        triStateButton.addActionListener(new ActionListener() {

            private FeatureRenderer.PlottingOption currentState = handler.getPlottingStyle();

            @Override
            public void actionPerformed(ActionEvent e) {
                currentState = FeatureRenderer.getNextState(currentState);
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

    // BufferedImage imageActive = new BufferedImage(25, 25, BufferedImage.TYPE_INT_ARGB);
    // = ImageIO.read(getClass().getResource(url));

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
        toggleButton.setPreferredSize(new Dimension(miniButtonSize, miniButtonSize));

        return toggleButton;
    }

    /**
     * @return toggle button which changes icon transparency when clicked
     * @throws IOException
     */
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

    /**
     * @param originalImage
     * @param alpha
     * @return original image with transparency alpha
     */
    private Image translucentImage(BufferedImage originalImage, float alpha) {

        int width = originalImage.getWidth(), height = originalImage.getHeight();

        BufferedImage newImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = newImage.createGraphics();
        g.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, alpha));
        g.drawImage(originalImage, 0, 0, width, height, null);
        g.dispose();

        return newImage;
    }

    public LoadAction getTrackLoadAction() {
        return trackLoadAction;
    }
}
