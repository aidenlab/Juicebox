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

import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.gui.SuperAdapter;
import juicebox.track.*;
import juicebox.track.feature.AnnotationLayerHandler;
import juicebox.windowui.DisabledGlassPane;
import org.broad.igv.ui.color.ColorChooserPanel;

import javax.swing.*;
import javax.swing.border.Border;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


/**
 * Created by muhammadsaadshamim on 8/4/16.
 */
public class LayersPanel extends JDialog {

    private static final long serialVersionUID = 9000047;
    public static final DisabledGlassPane disabledGlassPane = new DisabledGlassPane(Cursor.WAIT_CURSOR);
    private static LoadAction trackLoadAction;
    private static LoadEncodeAction encodeAction;
    private static Load2DAnnotationsDialog load2DAnnotationsDialog;
    private final JPanel layers2DPanel;
    //    private JPanel assemblyAnnotationsPanel;
    private final JPanel layerBoxGUI2DAnnotations = new JPanel(new GridLayout(0, 1));
    private final JTabbedPane tabbedPane;
    final JPanel layerBox1DGUI = new JPanel();
    final JPanel annotations1DPanel;

    public LayersPanel(final SuperAdapter superAdapter) {
        super(superAdapter.getMainWindow(), "Annotations Layer Panel");
        rootPane.setGlassPane(disabledGlassPane);

        Border padding = BorderFactory.createEmptyBorder(20, 20, 5, 20);

        annotations1DPanel = generate1DAnnotationsLayerSelectionPanel(superAdapter);
        if (annotations1DPanel != null) annotations1DPanel.setBorder(padding);
        layers2DPanel = generate2DAnnotationsLayerSelectionPanel(superAdapter);
        if (layers2DPanel != null) layers2DPanel.setBorder(padding);

//        assemblyAnnotationsPanel = generateAssemblyAnnotationsPanel(superAdapter);
//        if (assemblyAnnotationsPanel != null) assemblyAnnotationsPanel.setBorder(padding);

        tabbedPane = new JTabbedPane();
        tabbedPane.addTab("1D Annotations", null, annotations1DPanel,
                "Manage 1D Annotations");
        //tabbedPane.setMnemonicAt(1, KeyEvent.VK_2);

        tabbedPane.addTab("2D Annotations", null, layers2DPanel,
                "Manage 2D Annotations");
        //tabbedPane.setMnemonicAt(0, KeyEvent.VK_1);

//        tabbedPane.addTab("Assembly Annotations", null, assemblyAnnotationsPanel,
//                "Manage Assembly Annotations");

        setSize(1000, 700);
        add(tabbedPane);
        //setVisible(true);

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

        layerBox1DGUI.setLayout(new GridLayout(0, 1));
        JScrollPane scrollPane = new JScrollPane(layerBox1DGUI, JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED,
                JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);

        JPanel buttonPanel = new JPanel(new GridLayout(1, 0));
        JButton loadBasicButton = new JButton("Load Basic Annotations...");
        buttonPanel.add(loadBasicButton);
        JButton addLocalButton = new JButton("Add Local...");
        buttonPanel.add(addLocalButton);
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
                redraw1DLayerPanels(superAdapter);
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

        addLocalButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                HiC hiC = superAdapter.getHiC();
                if (hiC.getResourceTree() == null) {
                    ResourceTree resourceTree = new ResourceTree(superAdapter.getHiC(), null);
                    hiC.setResourceTree(resourceTree);
                }
                boolean loadSuccessful = superAdapter.getHiC().getResourceTree().addLocalButtonActionPerformed(superAdapter);
                if (loadSuccessful) {
                    trackLoadAction.actionPerformed(e);
                }
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
                redraw1DLayerPanels(superAdapter);
            }
        });
        return pane;
    }

    public void redraw1DLayerPanels(SuperAdapter superAdapter) {
        layerBox1DGUI.removeAll();
        for (HiCTrack track : superAdapter.getHiC().getLoadedTracks()) {
            if (track != null) {
                layerBox1DGUI.add(new TrackConfigPanel(superAdapter, track));
            }
        }
        layerBox1DGUI.revalidate();
        layerBox1DGUI.repaint();
        if (annotations1DPanel != null) {
            annotations1DPanel.revalidate();
            annotations1DPanel.repaint();
        }
    }

    /**
     * @param superAdapter
     * @return
     */
    private JPanel generate2DAnnotationsLayerSelectionPanel(final SuperAdapter superAdapter) {

        int i = 0;
        for (AnnotationLayerHandler handler : superAdapter.getAllLayers()) {
            try {
                JPanel panel = createLayerPanel(handler, superAdapter, layerBoxGUI2DAnnotations);
                //layerPanels.add(panel);
                layerBoxGUI2DAnnotations.add(panel, 0);
            } catch (IOException e) {
                System.err.println("Unable to generate layer panel " + (i - 1));
                //e.printStackTrace();
            }
        }
        final JScrollPane scrollPane = new JScrollPane(layerBoxGUI2DAnnotations, JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED,
                JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);

        JButton refreshButton = new JButton("Refresh View");
        refreshButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                superAdapter.updateMainLayersPanel();
                superAdapter.updateMiniAnnotationsLayerPanel();
                superAdapter.refresh();
            }
        });

        JButton importButton = new JButton("Load Loops/Domains...");
        JButton addLocalButton = new JButton("Add Local...");
        JButton newLayerButton = new JButton("Add New Layer");
        JButton mergeButton = new JButton("Merge Visible Layers");

        JPanel buttonPanel = new JPanel(new GridLayout(1, 0));
        buttonPanel.add(importButton);
        buttonPanel.add(addLocalButton);
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
                if (load2DAnnotationsDialog == null) {
                    load2DAnnotationsDialog = new Load2DAnnotationsDialog(LayersPanel.this, superAdapter);
                }
                load2DAnnotationsDialog.setVisible(true);
            }
        });
        importButton.setToolTipText("Import annotations into new layer");

        addLocalButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (load2DAnnotationsDialog == null) {
                    load2DAnnotationsDialog = new Load2DAnnotationsDialog(LayersPanel.this, superAdapter);
                }
                load2DAnnotationsDialog.addLocalButtonActionPerformed(LayersPanel.this);
            }
        });

        newLayerButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                createNewLayerAndAddItToPanels(superAdapter, null);
            }
        });

        mergeButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                merge2DAnnotationsAction(superAdapter);
            }
        });

        superAdapter.updateLayerDeleteStatus();
        return pane;
    }

    private JScrollPane generateLayers2DScrollPane(SuperAdapter superAdapter) {
        final JPanel layerBoxGUI = new JPanel();
        //layerBoxGUI.setLayout(new BoxLayout(layerBoxGUI, BoxLayout.PAGE_AXIS));
        layerBoxGUI.setLayout(new GridLayout(0, 1));
        //initialize here

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
        return new JScrollPane(layerBoxGUI, JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED,
                JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);
    }

    public AnnotationLayerHandler createNewLayerAndAddItToPanels(SuperAdapter superAdapter, AnnotationLayerHandler sourceHandler) {
        AnnotationLayerHandler handler = superAdapter.createNewLayer(null);
        if (sourceHandler != null) handler.duplicateDetailsFrom(sourceHandler);
        try {
            JPanel panel = createLayerPanel(handler, superAdapter, layerBoxGUI2DAnnotations);
            layerBoxGUI2DAnnotations.add(panel, 0);
            layerBoxGUI2DAnnotations.revalidate();
            layerBoxGUI2DAnnotations.repaint();
            superAdapter.setActiveLayerHandler(handler);
            superAdapter.updateLayerDeleteStatus();
            superAdapter.updateMiniAnnotationsLayerPanel();
            superAdapter.updateMainLayersPanel();
        } catch (Exception ee) {
            System.err.println("Unable to add new layer to GUI");
        }
        return handler;
    }

    private void merge2DAnnotationsAction(SuperAdapter superAdapter) {
        List<AnnotationLayerHandler> visibleLayers = new ArrayList<>();
        for (AnnotationLayerHandler handler : superAdapter.getAllLayers()) {
            if (handler.getLayerVisibility()) {
                visibleLayers.add(handler);
            }
        }

        AnnotationLayerHandler mergedHandler = superAdapter.createNewLayer(null);
        mergedHandler.mergeDetailsFrom(visibleLayers);
        try {
            JPanel panel = createLayerPanel(mergedHandler, superAdapter, layerBoxGUI2DAnnotations);
            layerBoxGUI2DAnnotations.add(panel, 0);

            for (AnnotationLayerHandler handler : visibleLayers) {
                int index = superAdapter.removeLayer(handler);
                if (index > -1) {
                    layerBoxGUI2DAnnotations.remove(index);
                }
            }

            layerBoxGUI2DAnnotations.revalidate();
            layerBoxGUI2DAnnotations.repaint();
            superAdapter.setActiveLayerHandler(mergedHandler);
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
        JToggleButton toggleVisibleButton = LayerPanelButtons.createVisibleButton(this, superAdapter, handler);
        JToggleButton toggleTransparentButton = LayerPanelButtons.createTransparencyButton(this, superAdapter, handler);
        JToggleButton toggleSparseButton = LayerPanelButtons.createToggleSparseButton(this, superAdapter, handler);
        JToggleButton toggleEnlargeButton = LayerPanelButtons.createToggleEnlargeButton(this, superAdapter, handler);
        JButton togglePlottingStyleButton = LayerPanelButtons.createTogglePlottingStyleButton(superAdapter, handler);
        JButton exportLayerButton = LayerPanelButtons.createExportButton(this, handler);
        JButton undoButton = LayerPanelButtons.createUndoButton(this, superAdapter, handler);
        ColorChooserPanel colorChooserPanel = LayerPanelButtons.createColorChooserButton(superAdapter, handler);
        JButton clearButton = LayerPanelButtons.createEraseButton(this, superAdapter, handler);
        JToggleButton writeButton = LayerPanelButtons.createWritingButton(this, superAdapter, handler);
        JButton deleteButton = LayerPanelButtons.createDeleteButton(this, superAdapter, layerBoxGUI, parentPanel, handler);
        JButton upButton = LayerPanelButtons.createMoveUpButton(this, superAdapter, layerBoxGUI, parentPanel, handler);
        JButton downButton = LayerPanelButtons.createMoveDownButton(this, superAdapter, layerBoxGUI, parentPanel, handler);
        JButton copyButton = LayerPanelButtons.createCopyButton(this, superAdapter, handler);

        parentPanel.add(nameField);
        Component[] allComponents = new Component[]{writeButton, toggleVisibleButton,
                colorChooserPanel, toggleTransparentButton, toggleEnlargeButton, togglePlottingStyleButton, toggleSparseButton,
                undoButton, clearButton, exportLayerButton, copyButton, upButton, downButton, deleteButton};

        if (HiCGlobals.isDevCustomChromosomesAllowedPublic) {
            JButton censorButton = LayerPanelButtons.createCensorButton(this, superAdapter, handler);
            allComponents = new Component[]{writeButton, toggleVisibleButton,
                    colorChooserPanel, toggleTransparentButton, toggleEnlargeButton, togglePlottingStyleButton, toggleSparseButton,
                    undoButton, clearButton, censorButton, exportLayerButton, copyButton, upButton, downButton, deleteButton};
        }
        for (Component component : allComponents) {
            if (component instanceof AbstractButton) {
                component.setMaximumSize(new Dimension(LayerPanelButtons.miniButtonSize, LayerPanelButtons.miniButtonSize));
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
                handler.setLayerNameAndOtherField(nameField.getText());
                nameField.setToolTipText("Change the name for this layer: " + nameField.getText());
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
                handler.setLayerNameAndOtherField(nameField.getText());
                nameField.setToolTipText("Change the name for this layer: " + nameField.getText());
            }

            @Override
            public void changedUpdate(DocumentEvent e) {
                handler.setLayerNameAndOtherField(nameField.getText());
                nameField.setToolTipText("Change the name for this layer: " + nameField.getText());
            }
        };
    }

    public void updateLayers2DPanel(SuperAdapter superAdapter) {
        layers2DPanel.remove(0);
        layers2DPanel.add(generateLayers2DScrollPane(superAdapter), BorderLayout.CENTER, 0);
        tabbedPane.updateUI();
        tabbedPane.repaint();
        tabbedPane.revalidate();
    }

    public void updateBothLayersPanels(SuperAdapter superAdapter) {
        superAdapter.updateMiniAnnotationsLayerPanel();
        superAdapter.updateMainLayersPanel();
    }

    public LoadAction getTrackLoadAction() {
        return trackLoadAction;
    }
}
