/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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
import juicebox.mapcolorui.FeatureRenderer;
import juicebox.track.feature.AnnotationLayerHandler;
import org.broad.igv.ui.color.ColorChooserPanel;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.IOException;

class LayerPanelButtons {

    public static final int miniButtonSize = 30;

    public static ColorChooserPanel createColorChooserButton(final SuperAdapter superAdapter, final AnnotationLayerHandler handler) {
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
        return colorChooserPanel;
    }

    public static JButton createCensorButton(final LayersPanel lp, final SuperAdapter superAdapter,
                                             final AnnotationLayerHandler handler) throws IOException {
        final JButton censorButton = createIconButton(lp, "/images/layer/hiclogo.png");
        censorButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                superAdapter.createCustomChromosomeMap(handler.getAnnotationLayer().getFeatureList(), handler.getLayerName());
            }
        });
        handler.setCensorButton(censorButton);
        censorButton.setEnabled(handler.getExportCapability());
        censorButton.setToolTipText("Create sub-map (custom chromosome) from this layer");
        return censorButton;
    }

    public static JToggleButton createVisibleButton(final Object lp, final SuperAdapter superAdapter,
                                                    final AnnotationLayerHandler handler) throws IOException {
        final JToggleButton toggleVisibleButton = createToggleIconButton(lp, "/images/layer/eye_clicked_green.png",
                "/images/layer/eye_clicked.png", handler.getLayerVisibility());
        toggleVisibleButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                handler.setLayerVisibility(toggleVisibleButton.isSelected());
                if (lp instanceof LayersPanel) {
                    superAdapter.updateMiniAnnotationsLayerPanel();
                } else if (lp instanceof MiniAnnotationsLayerPanel) {
                    superAdapter.updateMainLayersPanel();
                }
                superAdapter.repaint();
            }
        });
        toggleVisibleButton.setToolTipText("Toggle visibility of this layer");
        return toggleVisibleButton;
    }

    /**
     * toggle transparency for this layer
     */
    public static JToggleButton createTransparencyButton(final LayersPanel lp, final SuperAdapter superAdapter,
                                                         final AnnotationLayerHandler handler) throws IOException {
        final JToggleButton toggleTransparentButton = createToggleIconButton(lp, "/images/layer/trans_clicked_green.png",
                "/images/layer/trans_clicked.png", handler.getIsTransparent());
        toggleTransparentButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                handler.setIsTransparent(toggleTransparentButton.isSelected());
                superAdapter.repaint();
            }
        });
        toggleTransparentButton.setToolTipText("Toggle transparency of this layer");
        return toggleTransparentButton;
    }

    /**
     * Sparse (/subset) plotting for 2d annotations
     */
    public static JToggleButton createToggleSparseButton(final LayersPanel lp, final SuperAdapter superAdapter,
                                                         final AnnotationLayerHandler handler) throws IOException {
        final JToggleButton toggleSparseButton = createToggleIconButton(lp, "/images/layer/sparse.png",
                "/images/layer/sparse.png", handler.getIsSparse());
        toggleSparseButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                handler.setIsSparse(toggleSparseButton.isSelected());
                superAdapter.repaint();
            }
        });
        toggleSparseButton.setToolTipText("Plot a limited number of 2D annotations in this layer at a time " +
                "(speed up plotting when there are many annotations).");
        return toggleSparseButton;
    }

    public static JButton createCopyButton(final LayersPanel lp, final SuperAdapter superAdapter,
                                           final AnnotationLayerHandler handler) throws IOException {
        JButton copyButton = createIconButton(lp, "/images/layer/copy.png");
        copyButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                //AnnotationLayerHandler handlerDup =
                lp.createNewLayerAndAddItToPanels(superAdapter, handler);
            }
        });
        copyButton.setToolTipText("Duplicate this layer");
        return copyButton;
    }

    public static JButton createMoveDownButton(final LayersPanel layersPanel, final SuperAdapter superAdapter,
                                               final JPanel layerBoxGUI, final JPanel parentPanel,
                                               final AnnotationLayerHandler handler) throws IOException {
        JButton downButton = createIconButton(layersPanel, "/images/layer/down.png");
        downButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                layerBoxGUI.remove(parentPanel);
                int index = superAdapter.moveDownIndex(handler);
                layerBoxGUI.add(parentPanel, index);
                layerBoxGUI.revalidate();
                layerBoxGUI.repaint();
                superAdapter.updateMiniAnnotationsLayerPanel();
                superAdapter.repaint();
            }
        });
        downButton.setToolTipText("Move this layer down (drawing order)");
        return downButton;
    }

    public static JButton createMoveUpButton(final LayersPanel layersPanel, final SuperAdapter superAdapter,
                                             final JPanel layerBoxGUI, final JPanel parentPanel,
                                             final AnnotationLayerHandler handler) throws IOException {
        JButton upButton = createIconButton(layersPanel, "/images/layer/up.png");
        upButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                layerBoxGUI.remove(parentPanel);
                int index = superAdapter.moveUpIndex(handler);
                layerBoxGUI.add(parentPanel, index);
                layerBoxGUI.revalidate();
                layerBoxGUI.repaint();
                superAdapter.updateMiniAnnotationsLayerPanel();
                superAdapter.repaint();
            }
        });
        upButton.setToolTipText("Move this layer up (drawing order)");
        return upButton;
    }

    public static JButton createDeleteButton(final LayersPanel layersPanel, final SuperAdapter superAdapter,
                                             final JPanel layerBoxGUI, final JPanel parentPanel,
                                             final AnnotationLayerHandler handler) throws IOException {

        JButton deleteButton = createIconButton(layersPanel, "/images/layer/trash.png");
        deleteButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (superAdapter.deleteCustomAnnotationDialog(handler.getLayerName()) == JOptionPane.YES_OPTION) {
                    layerBoxGUI.remove(parentPanel);
                    superAdapter.removeLayer(handler);
                    layerBoxGUI.revalidate();
                    layerBoxGUI.repaint();
                    superAdapter.updateMiniAnnotationsLayerPanel();
                    superAdapter.repaint();
                }
            }
        });
        handler.setDeleteLayerButton(deleteButton);
        deleteButton.setToolTipText("Delete this layer");
        return deleteButton;
    }

    /**
     * toggle whether the features will be enlarged for this layer
     */
    public static JToggleButton createToggleEnlargeButton(final LayersPanel lp, final SuperAdapter superAdapter,
                                                          final AnnotationLayerHandler handler) throws IOException {

        final JToggleButton toggleEnlargeButton = createToggleIconButton(lp, "/images/layer/enlarge_clicked_down.png",
                "/images/layer/enlarge_clicked_up.png", handler.getIsEnlarged());
        toggleEnlargeButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                handler.setIsEnlarged(toggleEnlargeButton.isSelected());
                superAdapter.repaint();
            }
        });
        toggleEnlargeButton.setToolTipText("Enlarge features in this layer");
        return toggleEnlargeButton;
    }

    /**
     * toggle plotting styles; setup and action done in helper function
     */
    public static JButton createTogglePlottingStyleButton(final SuperAdapter superAdapter, final AnnotationLayerHandler handler) throws IOException {
        PlottingStyleButton togglePlottingStyleButton = createTogglePlottingStyleIconButton(handler, superAdapter);
        togglePlottingStyleButton.setToolTipText("Change partial plotting style in this layer");
        handler.setPlottingStyleButton(togglePlottingStyleButton);
        return togglePlottingStyleButton;
    }

    /**
     * export annotations in layer to new file
     */
    public static JButton createExportButton(final LayersPanel layersPanel, final AnnotationLayerHandler handler) throws IOException {
        final JButton exportLayerButton = createIconButton(layersPanel, "/images/layer/export_icon_green.png");
        exportLayerButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                handler.exportAnnotations();
            }
        });
        handler.setExportButton(exportLayerButton);
        exportLayerButton.setEnabled(handler.getExportCapability());
        exportLayerButton.setToolTipText("Export annotations from this layer");
        return exportLayerButton;
    }

    /**
     * undo last annotation in layer
     */
    public static JButton createUndoButton(final LayersPanel lp, final SuperAdapter superAdapter,
                                           final AnnotationLayerHandler handler) throws IOException {
        final JButton undoButton = createIconButton(lp, "/images/layer/undo.png");
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
        return undoButton;
    }

    /**
     * clear annotations in this layer
     */
    public static JButton createEraseButton(final LayersPanel lp, final SuperAdapter superAdapter,
                                            final AnnotationLayerHandler handler) throws IOException {
        JButton clearButton = createIconButton(lp, "/images/layer/erase.png");
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
        return clearButton;
    }

    /**
     * button to set active layer
     */
    public static JToggleButton createWritingButton(final Object lp, final SuperAdapter superAdapter,
                                                    final AnnotationLayerHandler handler) throws IOException {
        final JToggleButton writeButton = createToggleIconButton(lp, "/images/layer/pencil.png", "/images/layer/pencil_gray.png", handler.isActiveLayer(superAdapter));
        handler.setActiveLayerButton(writeButton);
        writeButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                superAdapter.setActiveLayerHandler(handler);
                superAdapter.updateMiniAnnotationsLayerPanel();
                superAdapter.updateMainLayersPanel();
                superAdapter.repaint();
            }
        });
        writeButton.setToolTipText("Enable drawing of annotations to this layer; Hold down shift key, then click and drag on map");
        return writeButton;
    }

    /**
     * triple state toggle button for plotting styles
     */
    private static PlottingStyleButton createTogglePlottingStyleIconButton(final AnnotationLayerHandler handler,
                                                                           final SuperAdapter superAdapter) throws IOException {

        final PlottingStyleButton triStateButton = new PlottingStyleButton();
        triStateButton.setPreferredSize(new Dimension(miniButtonSize, miniButtonSize));
        triStateButton.setBorderPainted(false);
        triStateButton.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent e) {
                FeatureRenderer.PlottingOption currentState = FeatureRenderer.getNextState(handler.getPlottingStyle());
                handler.setPlottingStyle(currentState);
                superAdapter.repaint();
            }

        });

        triStateButton.setCurrentState(handler.getPlottingStyle());

        return triStateButton;
    }

    private static JButton createIconButton(Object ls, String url) throws IOException {
        BufferedImage imageActive = ImageIO.read(ls.getClass().getResource(url));
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
     * @return toggle button which changes icon transparency when clicked
     * @throws IOException
     */
    private static JToggleButton createToggleIconButton(Object lp, String url1, String url2, boolean activatedStatus) throws IOException {

        // image when button is active/selected (is the darkest shade/color)
        //BufferedImage imageActive = ImageIO.read(getClass().getResource(url1));
        ImageIcon iconActive = new ImageIcon(ImageIO.read(lp.getClass().getResource(url1)));

        // image when button is inactive/transitioning (lighter shade/color)
        ImageIcon iconTransitionDown = new ImageIcon(translucentImage(ImageIO.read(lp.getClass().getResource(url2)), 0.6f));
        ImageIcon iconTransitionUp = new ImageIcon(translucentImage(ImageIO.read(lp.getClass().getResource(url1)), 0.6f));
        ImageIcon iconInactive = new ImageIcon(translucentImage(ImageIO.read(lp.getClass().getResource(url2)), 0.2f));
        ImageIcon iconDisabled = new ImageIcon(translucentImage(ImageIO.read(lp.getClass().getResource(url2)), 0.1f));

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
    public static Image translucentImage(BufferedImage originalImage, float alpha) {

        int width = originalImage.getWidth(), height = originalImage.getHeight();

        BufferedImage newImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = newImage.createGraphics();
        g.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, alpha));
        g.drawImage(originalImage, 0, 0, width, height, null);
        g.dispose();

        return newImage;
    }
}
