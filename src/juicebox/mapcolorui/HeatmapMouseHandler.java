/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2022 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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

package juicebox.mapcolorui;

import com.jidesoft.swing.JidePopupMenu;
import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.MainWindow;
import juicebox.assembly.AssemblyOperationExecutor;
import juicebox.assembly.AssemblyScaffoldHandler;
import juicebox.assembly.Scaffold;
import juicebox.data.ChromosomeHandler;
import juicebox.data.ExpectedValueFunction;
import juicebox.data.Matrix;
import juicebox.data.MatrixZoomData;
import juicebox.data.basics.Chromosome;
import juicebox.gui.SuperAdapter;
import juicebox.track.HiCFragmentAxis;
import juicebox.track.HiCGridAxis;
import juicebox.track.feature.AnnotationLayer;
import juicebox.track.feature.AnnotationLayerHandler;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DGuiContainer;
import juicebox.windowui.EditFeatureAttributesDialog;
import juicebox.windowui.MatrixType;
import org.broad.igv.util.Pair;

import javax.swing.*;
import java.awt.*;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.StringSelection;
import java.awt.event.*;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static java.awt.Toolkit.getDefaultToolkit;

public class HeatmapMouseHandler extends MouseAdapter {

    public static final int clickDelay = 500;
    private static final int clickLong = 400;
    private final List<Feature2D> highlightedFeatures = new ArrayList<>();
    private final List<Integer> selectedSuperscaffolds = new ArrayList<>();
    private final NumberFormat formatter = NumberFormat.getInstance();
    private final int RESIZE_SNAP = 5;
    private final HiC hic;
    private final SuperAdapter superAdapter;
    private final HeatmapPanel parent;
    private final List<Feature2D> selectedFeatures = new ArrayList<>();
    private final List<Feature2DGuiContainer> allFeaturePairs = new ArrayList<>();
    private final List<Feature2DGuiContainer> allMainFeaturePairs = new ArrayList<>();
    private final List<Feature2DGuiContainer> allEditFeaturePairs = new ArrayList<>();

    DragMode dragMode = DragMode.NONE;
    double startTime, endTime;
    private Robot heatmapMouseBot;
    private PromptedAssemblyAction currentPromptedAssemblyAction = PromptedAssemblyAction.NONE;
    private PromptedAssemblyAction promptedAssemblyActionOnClick = PromptedAssemblyAction.NONE;
    private Pair<Pair<Integer, Integer>, Feature2D> preAdjustLoop = null;
    private boolean featureOptionMenuEnabled = false;
    private boolean firstAnnotation;
    private AdjustAnnotation adjustAnnotation = AdjustAnnotation.NONE;
    private Point lastMousePoint;
    private Point lastPressedMousePoint;
    private boolean straightEdgeEnabled = false, diagonalEdgeEnabled = false;
    private int debrisFeatureSize = RESIZE_SNAP;
    private Rectangle zoomRectangle;
    private Rectangle annotateRectangle;
    private Feature2DGuiContainer currentFeature = null;
    private boolean changedSize = false;
    private Feature2DGuiContainer currentUpstreamFeature = null;
    private Feature2DGuiContainer currentDownstreamFeature = null;
    private boolean showFeatureHighlight = true;
    private boolean activelyEditingAssembly = false;
    private Feature2D debrisFeature = null;
    private Feature2D tempSelectedGroup = null;

    public HeatmapMouseHandler(HiC hic, SuperAdapter superAdapter, HeatmapPanel parent) {
        this.hic = hic;
        this.superAdapter = superAdapter;
        this.parent = parent;
        this.firstAnnotation = true;
        try {
            heatmapMouseBot = new Robot();
        } catch (AWTException ignored) {
        }
    }

    public List<Feature2D> getHighlightedFeature() {
        return highlightedFeatures;
    }

    public void eraseHighlightedFeatures() {
        highlightedFeatures.clear();
        hic.setHighlightedFeatures(new ArrayList<>());
    }

    public void clearSelectedFeatures() {
        selectedSuperscaffolds.clear();
        updateSelectedFeatures(false);
        selectedFeatures.clear();
    }

    public boolean getIsActivelyEditingAssembly() {
        return activelyEditingAssembly;
    }

    public void clearFeaturePairs() {
        allFeaturePairs.clear();
        if (activelyEditingAssembly) {
            allMainFeaturePairs.clear();
            allEditFeaturePairs.clear();
        }
    }

    public boolean getShouldShowHighlight() {
        return showFeatureHighlight;
    }

    public void setActivelyEditingAssembly(boolean bool) {
        activelyEditingAssembly = bool;
    }

    private JidePopupMenu getPopupMenu(final int xMousePos, final int yMousePos) {

        JidePopupMenu menu = new JidePopupMenu();

        if (SuperAdapter.assemblyModeCurrentlyActive) {
            getAssemblyPopupMenu(xMousePos, yMousePos, menu);
            menu.addSeparator();
        }


        final JMenuItem miUndoZoom = new JMenuItem("Undo Zoom");
        miUndoZoom.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                hic.setCursorPoint(new Point(xMousePos, yMousePos));
                hic.undoZoomAction();
            }
        });
        miUndoZoom.setEnabled(hic.getZoomActionTracker().validateUndoZoom());
        menu.add(miUndoZoom);

        final JMenuItem miRedoZoom = new JMenuItem("Redo Zoom");
        miRedoZoom.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                hic.setCursorPoint(new Point(xMousePos, yMousePos));
                hic.redoZoomAction();
            }
        });
        miRedoZoom.setEnabled(hic.getZoomActionTracker().validateRedoZoom());
        menu.add(miRedoZoom);

        // add Jump to Diagonal menu items
        addJumpToDiagonalMenuItems(menu, xMousePos, yMousePos);

        final JCheckBoxMenuItem mi = new JCheckBoxMenuItem("Enable straight edge");
        mi.setSelected(straightEdgeEnabled);
        mi.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (mi.isSelected()) {
                    straightEdgeEnabled = true;
                    diagonalEdgeEnabled = false;
                    parent.setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));
                } else {
                    straightEdgeEnabled = false;
                    hic.setCursorPoint(null);
                    parent.setCursor(Cursor.getDefaultCursor());
                    parent.repaint();
                    superAdapter.repaintTrackPanels();
                }
            }
        });
        menu.add(mi);

        final JCheckBoxMenuItem miv2 = new JCheckBoxMenuItem("Enable diagonal edge");
        miv2.setSelected(diagonalEdgeEnabled);
        miv2.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (miv2.isSelected()) {
                    straightEdgeEnabled = false;
                    diagonalEdgeEnabled = true;
                    parent.setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));
                } else {
                    diagonalEdgeEnabled = false;
                    hic.setDiagonalCursorPoint(null);
                    parent.setCursor(Cursor.getDefaultCursor());
                    parent.repaint();
                    superAdapter.repaintTrackPanels();
                }

            }
        });
        menu.add(miv2);

        // internally, single sync = what we previously called sync
        final JMenuItem mi3 = new JMenuItem("Broadcast Single Sync");
        mi3.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                hic.broadcastLocation();
            }
        });

        // internally, continuous sync = what we used to call linked
        final JCheckBoxMenuItem mi4 = new JCheckBoxMenuItem("Broadcast Continuous Sync");
        mi4.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                final boolean isLinked = mi4.isSelected();
                if (isLinked) {
                    HiCGlobals.wasLinkedBeforeMousePress = false;
                    hic.broadcastLocation();
                }
                hic.setLinkedMode(isLinked);
            }
        });

        final JCheckBoxMenuItem mi5 = new JCheckBoxMenuItem("Freeze hover text");
        mi5.setSelected(!superAdapter.isTooltipAllowedToUpdated());
        mi5.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                superAdapter.toggleToolTipUpdates(!superAdapter.isTooltipAllowedToUpdated());
            }
        });

        final JMenuItem mi6 = new JMenuItem("Copy hover text to clipboard");
        mi6.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                StringSelection stringSelection = new StringSelection(superAdapter.getToolTip());
                Clipboard clpbrd = getDefaultToolkit().getSystemClipboard();
                clpbrd.setContents(stringSelection, null);
            }
        });

        final JMenuItem mi7 = new JMenuItem("Copy top position to clipboard");
        mi7.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                StringSelection stringSelection = new StringSelection(hic.getXPosition());
                superAdapter.setPositionChrTop(hic.getXPosition().concat(":").concat(String.valueOf(hic.getXContext().getZoom().getBinSize())));
                superAdapter.setPositionChrLeft(hic.getYPosition().concat(":").concat(String.valueOf(hic.getYContext().getZoom().getBinSize())));
                Clipboard clpbrd = getDefaultToolkit().getSystemClipboard();
                clpbrd.setContents(stringSelection, null);
            }
        });

        // TODO - can we remove this second option and just have a copy position to clipboard? Is this used?
        final JMenuItem mi8 = new JMenuItem("Copy left position to clipboard");
        mi8.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                StringSelection stringSelection = new StringSelection(hic.getYPosition());
                superAdapter.setPositionChrTop(hic.getXPosition().concat(":").concat(String.valueOf(hic.getXContext().getZoom().getBinSize())));
                superAdapter.setPositionChrLeft(hic.getYPosition().concat(":").concat(String.valueOf(hic.getYContext().getZoom().getBinSize())));
                Clipboard clpbrd = getDefaultToolkit().getSystemClipboard();
                clpbrd.setContents(stringSelection, null);
            }
        });

        final JCheckBoxMenuItem mi85Highlight = new JCheckBoxMenuItem("Highlight");
        mi85Highlight.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                //highlightedFeatures.clear();
                addHighlightedFeature(currentFeature.getFeature2D());
            }
        });

        final JCheckBoxMenuItem mi86Toggle = new JCheckBoxMenuItem("Toggle Highlight Visibility");
        mi86Toggle.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                featureOptionMenuEnabled = false;
                showFeatureHighlight = !showFeatureHighlight;
                hic.setShowFeatureHighlight(showFeatureHighlight);
                parent.repaint();
            }
        });

        final JMenuItem mi87Remove = new JMenuItem("Remove Highlight");
        mi87Remove.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                parent.removeHighlightedFeature();
            }
        });

        final JMenuItem mi9_c = new JMenuItem("Export data centered on pixel");
        mi9_c.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                try {
                    hic.exportDataCenteredAboutRegion(xMousePos, yMousePos);
                } catch (Exception ee) {
                    ee.printStackTrace();
                }
            }
        });

        final JCheckBoxMenuItem mi9_h = new JCheckBoxMenuItem("Generate Horizontal 1D Track");
        mi9_h.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                hic.generateTrackFromLocation(yMousePos, true);
            }
        });

        final JCheckBoxMenuItem mi9_v = new JCheckBoxMenuItem("Generate Vertical 1D Track");
        mi9_v.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                hic.generateTrackFromLocation(xMousePos, false);
            }
        });


        final JMenuItem mi10_1 = new JMenuItem("Change Color");
        mi10_1.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                featureOptionMenuEnabled = false;
                Pair<Rectangle, Feature2D> featureCopy =
                        new Pair<>(currentFeature.getRectangle(), currentFeature.getFeature2D());
                parent.launchColorSelectionMenu(featureCopy);
            }
        });

        final JMenuItem mi10_2 = new JMenuItem("Change Attributes");
        mi10_2.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                featureOptionMenuEnabled = false;
                new EditFeatureAttributesDialog(parent.getMainWindow(), currentFeature.getFeature2D(),
                        superAdapter.getActiveLayerHandler().getAnnotationLayer());
            }
        });

        final JMenuItem mi10_3 = new JMenuItem("Delete");
        mi10_3.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                featureOptionMenuEnabled = false;
                Feature2D feature = currentFeature.getFeature2D();
                int chr1Idx = hic.getXContext().getChromosome().getIndex();
                int chr2Idx = hic.getYContext().getChromosome().getIndex();
                try {
                    superAdapter.getActiveLayerHandler().removeFromList(hic.getZd(), chr1Idx, chr2Idx, 0, 0,
                            Feature2DHandler.numberOfLoopsToFind, hic.getXContext().getBinOrigin(),
                            hic.getYContext().getBinOrigin(), hic.getScaleFactor(), feature);
                } catch (Exception ee) {
                    System.err.println("Could not remove custom annotation");
                }
                superAdapter.refresh();
            }
        });


        final JMenu configureFeatureMenu = new JMenu("Configure feature");
        configureFeatureMenu.add(mi10_1);
        configureFeatureMenu.add(mi10_2);
        configureFeatureMenu.add(mi10_3);

        if (hic != null) {
            //    menu.add(mi2);
            menu.add(mi3);
            mi4.setSelected(hic.isLinkedMode());
            menu.add(mi4);
            menu.add(mi5);
            menu.add(mi6);
            menu.add(mi7);
            menu.add(mi8);
            if (!ChromosomeHandler.isAllByAll(hic.getXContext().getChromosome())
                    && MatrixType.isObservedOrControl(hic.getDisplayOption())) {
                menu.addSeparator();
                menu.add(mi9_h);
                menu.add(mi9_v);
                menu.add(mi9_c);
            }

            boolean menuSeparatorNotAdded = true;

            if (highlightedFeatures.size() > 0) {
                menu.addSeparator();
                menuSeparatorNotAdded = false;
                mi86Toggle.setSelected(showFeatureHighlight);
                menu.add(mi86Toggle);
            }

            if (currentFeature != null) {//mouseIsOverFeature
                featureOptionMenuEnabled = true;
                if (menuSeparatorNotAdded) {
                    menu.addSeparator();
                }

                if (highlightedFeatures.size() > 0) {
                    if (!highlightedFeatures.contains(currentFeature.getFeature2D())) {
                        configureFeatureMenu.add(mi85Highlight);
                        menu.add(mi87Remove);
                    } else {
                        configureFeatureMenu.add(mi87Remove);
                    }
                } else {
                    configureFeatureMenu.add(mi85Highlight);
                }

                menu.add(configureFeatureMenu);
            } else if (highlightedFeatures.size() > 0) {
                menu.add(mi87Remove);
            }
        }


        return menu;

    }

    public Feature2D getDebrisFeature() {
        return debrisFeature;
    }

    public Feature2D getTempSelectedGroup() {
        return tempSelectedGroup;
    }

    public void setTempSelectedGroup(Feature2D feature2D) {
        tempSelectedGroup = feature2D;
    }

    public List<Feature2D> getSelectedFeatures() {
        return selectedFeatures;
    }

    private void addHighlightedFeatures(List<Feature2D> feature2DList) {
        highlightedFeatures.addAll(feature2DList);
        featureOptionMenuEnabled = false;
        showFeatureHighlight = true;
        hic.setShowFeatureHighlight(showFeatureHighlight);
        hic.setHighlightedFeatures(highlightedFeatures);
        superAdapter.repaintTrackPanels();
        parent.repaint();
    }

    private void updateSelectedFeatures(boolean status) {
        for (Feature2D feature2D : selectedFeatures) {
            feature2D.setSetIsSelectedColorUpdate(status);
        }
    }

    public void resetCurrentPromptedAssemblyAction() {
        currentPromptedAssemblyAction = HeatmapMouseHandler.PromptedAssemblyAction.NONE;
    }

    private String getFloatString(float value) {
        String valueString;
        if (Float.isNaN(value)) {
            valueString = "NaN";
        } else if (value < 0.001) {
            valueString = String.valueOf(value);
        } else {
            valueString = formatter.format(value);
        }
        return valueString;
    }

    private double getExpectedValue(int c1, int c2, int binX, int binY, MatrixZoomData zd,
                                    ExpectedValueFunction df) {
        double ev = 0;
        if (c1 == c2) {
            if (df != null) {
                int distance = Math.abs(binX - binY);
                ev = df.getExpectedValue(c1, distance);
            }
        } else {
            ev = zd.getAverageCount();
        }
        return ev;
    }

    private void executeSplitMenuAction() {

        AssemblyOperationExecutor.splitContig(selectedFeatures.get(0), debrisFeature, superAdapter, hic, true);

        HiCGlobals.splitModeEnabled = false;
        Chromosome chrX = superAdapter.getHiC().getXContext().getChromosome();
        Chromosome chrY = superAdapter.getHiC().getYContext().getChromosome();
        superAdapter.getEditLayer().filterTempSelectedGroup(chrX.getIndex(), chrY.getIndex());
        superAdapter.getEditLayer().clearAnnotations();
        superAdapter.setActiveLayerHandler(superAdapter.getMainLayer());
        debrisFeature = null;
        //moveDebrisToEnd();
        parent.removeSelection();
        reset();
    }

    /*
    public Feature2D generateDebrisFeature(int xMousePos, int yMousePos) {
        final double scaleFactor = hic.getScaleFactor();
        double binOriginX = hic.getXContext().getBinOrigin();
        double binOriginY = hic.getYContext().getBinOrigin();
        Rectangle annotateRectangle = new Rectangle(xMousePos, (int) (yMousePos + (binOriginX - binOriginY) * scaleFactor), RESIZE_SNAP, RESIZE_SNAP);
        superAdapter.getEditLayer().updateSelectionRegion(annotateRectangle);
        return superAdapter.getEditLayer().generateFeature(hic);
    }

    public void toggleActivelyEditingAssembly() {
        this.activelyEditingAssembly = !this.activelyEditingAssembly;
    }

    //private enum AdjustAnnotation {LEFT, RIGHT, NONE}
    */

    public void setFeatureOptionMenuEnabled(boolean bool) {
        featureOptionMenuEnabled = bool;
    }

    public PromptedAssemblyAction getCurrentPromptedAssemblyAction() {
        return this.currentPromptedAssemblyAction;
    }

    public PromptedAssemblyAction getPromptedAssemblyActionOnClick() {
        return this.promptedAssemblyActionOnClick;
    }

    public void setPromptedAssemblyActionOnClick(PromptedAssemblyAction promptedAssemblyAction) {
        this.promptedAssemblyActionOnClick = promptedAssemblyAction;
    }



    @Override
    public void mouseEntered(MouseEvent e) {
        setProperCursor();
    }

    public void reset() {
        debrisFeatureSize = RESIZE_SNAP;
    }

    private void setProperCursor() {
        if (straightEdgeEnabled || diagonalEdgeEnabled) {
            parent.setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));
        } else {
            parent.setCursor(Cursor.getDefaultCursor());
        }
    }

    private void addHighlightedFeature(Feature2D feature2D) {
        highlightedFeatures.add(feature2D);
        featureOptionMenuEnabled = false;
        showFeatureHighlight = true;
        hic.setShowFeatureHighlight(showFeatureHighlight);
        hic.setHighlightedFeatures(highlightedFeatures);
        superAdapter.repaintTrackPanels();
        parent.repaint();
    }

    public void renderMouseAnnotations(Graphics2D g2) {
        if (zoomRectangle != null) {
            g2.draw(zoomRectangle);
        }

        if (annotateRectangle != null) {
            g2.draw(annotateRectangle);
        }
    }

    public void addAllFeatures(AnnotationLayerHandler handler, List<Feature2D> loops, MatrixZoomData zd,
                               double binOriginX, double binOriginY, double scaleFactor,
                               boolean activelyEditingAssembly) {
        allFeaturePairs.addAll(handler.convertToFeaturePairs(handler, loops, zd, binOriginX, binOriginY, scaleFactor));

        if (activelyEditingAssembly) {
            if (handler == superAdapter.getMainLayer()) {
                allMainFeaturePairs.addAll(superAdapter.getMainLayer().convertToFeaturePairs(handler, loops, zd, binOriginX, binOriginY, scaleFactor));
            } else if (handler == superAdapter.getEditLayer() && !selectedFeatures.isEmpty()) {
                allEditFeaturePairs.addAll(superAdapter.getEditLayer().convertToFeaturePairs(handler, loops, zd, binOriginX, binOriginY, scaleFactor));
            }
        }
    }

    @Override
    public void mouseExited(MouseEvent e) {
        hic.setCursorPoint(null);
        if (straightEdgeEnabled || diagonalEdgeEnabled) {
            superAdapter.repaintTrackPanels();
        }
    }

    @Override
    public void mousePressed(final MouseEvent e) {
        startTime = System.nanoTime();
        featureOptionMenuEnabled = false;
        if (hic.isWholeGenome()) {
            if (e.isPopupTrigger()) {
                getPopupMenu(e.getX(), e.getY()).show(parent, e.getX(), e.getY());
            }
            return;
        }
        // Priority is right click
        if (e.isPopupTrigger()) {
            getPopupMenu(e.getX(), e.getY()).show(parent, e.getX(), e.getY());
        } else {

            // turn off continuous sync for dragging
            if (hic.isLinkedMode()) {
                HiCGlobals.wasLinkedBeforeMousePress = true;
                hic.setLinkedMode(false);
            }

            // Alt down for zoom
            if (e.isAltDown()) {
                dragMode = DragMode.ZOOM;
                // Shift down for custom annotations
            } else if (e.isShiftDown() && (activelyEditingAssembly || superAdapter.getActiveLayerHandler().getAnnotationLayerType() != AnnotationLayer.LayerType.SCAFFOLD)) {

                if (!activelyEditingAssembly) {
                    if (superAdapter.unsavedEditsExist() && firstAnnotation) {
                        firstAnnotation = false;
                        String text = "There are unsaved hand annotations from your previous session! \n" +
                                "Go to 'Annotations > Hand Annotations > Load Last' to restore.";
                        System.err.println(text);
                        JOptionPane.showMessageDialog(superAdapter.getMainWindow(), text);
                    }

                    //superAdapter.getActiveLayerHandler().updateSelectionPoint(e.getX(), e.getY());
                    superAdapter.getActiveLayerHandler().doPeak();
                }

                dragMode = DragMode.ANNOTATE;
                //superAdapter.getActiveLayer().updateSelectionPoint(e.getX(), e.getY());
                superAdapter.getActiveLayerHandler().doPeak();
                parent.setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));
                // Corners for resize annotation

                try {
                    List<Feature2D> newSelectedFeatures = superAdapter.getMainLayer().getSelectedFeatures(hic, e.getX(), e.getY());
                    if (!selectedFeatures.get(0).equals(newSelectedFeatures.get(0))) {

                        HiCGlobals.splitModeEnabled = false;
                        superAdapter.setActiveLayerHandler(superAdapter.getMainLayer());
                        superAdapter.getLayersPanel().updateBothLayersPanels(superAdapter);
                        superAdapter.getEditLayer().clearAnnotations();
                    }
                    if (selectedFeatures.size() == 1 && selectedFeatures.get(0).equals(newSelectedFeatures.get(0))) {
                        HiCGlobals.splitModeEnabled = true;
                    }
                } catch (Exception ignored) {
                }
            } else if (adjustAnnotation != AdjustAnnotation.NONE) {
                dragMode = DragMode.RESIZE;
                Feature2D loop;
                if (activelyEditingAssembly && currentPromptedAssemblyAction == PromptedAssemblyAction.ADJUST) {
                    loop = superAdapter.getEditLayer().getFeatureHandler().getFeatureList().get(1, 1).get(0);
                } else {
                    loop = currentFeature.getFeature2D();
                }
                // Resizing upper left corner, keep end points stationary
                if (adjustAnnotation == AdjustAnnotation.LEFT) {
                    superAdapter.getActiveLayerHandler().setStationaryEnd(loop.getEnd1(), loop.getEnd2());
                    // Resizing lower right corner, keep start points stationary
                } else {
                    superAdapter.getActiveLayerHandler().setStationaryStart(loop.getStart1(), loop.getStart2());
                }


                try {
                    HiCGridAxis xAxis = hic.getZd().getXGridAxis();
                    HiCGridAxis yAxis = hic.getZd().getYGridAxis();
                    final double scaleFactor = hic.getScaleFactor();
                    double binOriginX = hic.getXContext().getBinOrigin();
                    double binOriginY = hic.getYContext().getBinOrigin();

                    // hic.getFeature2DHandler()
                    annotateRectangle = superAdapter.getActiveLayerHandler().getFeatureHandler().getRectangleFromFeature(
                            xAxis, yAxis, loop, binOriginX, binOriginY, scaleFactor);
                    int chr1Idx = hic.getXContext().getChromosome().getIndex();
                    int chr2Idx = hic.getYContext().getChromosome().getIndex();
                    preAdjustLoop = new Pair<>(new Pair<>(chr1Idx, chr2Idx), loop);

                } catch (Exception ex) {
                    ex.printStackTrace();
                }

            } else if (!e.isShiftDown() && currentPromptedAssemblyAction == PromptedAssemblyAction.CUT) {
                Feature2D debrisFeature = generateDebrisFeature(e, debrisFeatureSize);
                setDebrisFeauture(debrisFeature);
                int chr1Idx = hic.getXContext().getChromosome().getIndex();
                int chr2Idx = hic.getYContext().getChromosome().getIndex();
                if (debrisFeature != null) {
                    superAdapter.getEditLayer().getAnnotationLayer().getFeatureHandler().getFeatureList().checkAndRemoveFeature(chr1Idx, chr2Idx, debrisFeature);
                }
                superAdapter.getEditLayer().getAnnotationLayer().add(chr1Idx, chr2Idx, debrisFeature);
                HiCGlobals.splitModeEnabled = true;
                superAdapter.setActiveLayerHandler(superAdapter.getEditLayer());
                restoreDefaultVariables();
                parent.repaint();
            } else {
                dragMode = DragMode.PAN;
                parent.setCursor(MainWindow.fistCursor);
            }
            lastMousePoint = e.getPoint();
            lastPressedMousePoint = e.getPoint();
        }
    }

    @Override
    public void mouseReleased(final MouseEvent e) {
        endTime = System.nanoTime();
        if (e.isPopupTrigger()) {
            getPopupMenu(e.getX(), e.getY()).show(parent, e.getX(), e.getY());
            dragMode = DragMode.NONE;
            lastMousePoint = null;
            zoomRectangle = null;
            annotateRectangle = null;
            setProperCursor();
            // After popup, priority is assembly mode, highlighting those features.

        } /*else if (HiCGlobals.splitModeEnabled && activelyEditingAssembly) {
                if (dragMode == DragMode.ANNOTATE) {
                    Feature2D feature2D = superAdapter.getActiveLayerHandler().generateFeature(hic); //TODO can modify split to wait for user to accept split
                    if (feature2D == null) {
                        int x = (int) lastMousePoint.getX();
                        int y = (int) lastMousePoint.getY();
                        Rectangle annotateRectangle = new Rectangle(x, y, 1, 1);
                        superAdapter.getActiveLayerHandler().updateSelectionRegion(annotateRectangle);
                        feature2D = superAdapter.getActiveLayerHandler().generateFeature(hic); //TODO can modify split to wait for user to accept split
                    }
                    AnnotationLayerHandler editLayerHandler = superAdapter.getEditLayer();
                    debrisFeature = feature2D;
//                    editLayerHandler.getAnnotationLayer().add(Hic.);
                    int chr1Idx = hic.getXContext().getChromosomeFromName().getIndex();
                    int chr2Idx = hic.getYContext().getChromosomeFromName().getIndex();
//                    executeSplitMenuAction(selectedFeatures.get(0),debrisFeature);
                    editLayerHandler.getAnnotationLayer().add(chr1Idx, chr2Idx, debrisFeature);
                    restoreDefaultVariables();

                }

            }*/ else {
            // turn on continuous sync after dragging
            if (HiCGlobals.wasLinkedBeforeMousePress) {
                HiCGlobals.wasLinkedBeforeMousePress = false;
                hic.setLinkedMode(true);

                if (lastPressedMousePoint != null) {
                    double deltaX = e.getX() - lastPressedMousePoint.getX();
                    double deltaY = e.getY() - lastPressedMousePoint.getY();
                    if (Math.abs(deltaX) > 0 && Math.abs(deltaY) > 0) {
                        hic.broadcastLocation();
                    }
                } else {
                    hic.broadcastLocation();
                }
            }

            if (e.isShiftDown() && HiCGlobals.phasing) {

                //superscaffold selection handling in the phasing case

                Feature2DGuiContainer newSelectedSuperscaffold = getMouseHoverSuperscaffold(e.getX(), e.getY());
                if (newSelectedSuperscaffold == null) {
                    parent.removeSelection();
                } else {
                    int tentativeSuperscaffoldId = Integer.parseInt(newSelectedSuperscaffold.getFeature2D().getAttribute("Superscaffold #")) - 1;
                    int altTentativeSuperscaffoldId;
                    if (tentativeSuperscaffoldId % 2 == 0) {
                        altTentativeSuperscaffoldId = tentativeSuperscaffoldId + 1;
                    } else {
                        altTentativeSuperscaffoldId = tentativeSuperscaffoldId - 1;
                    }

                    if (selectedSuperscaffolds.contains(tentativeSuperscaffoldId)) {
                        selectedSuperscaffolds.remove(Integer.valueOf(tentativeSuperscaffoldId));
                        highlightedFeatures.remove(newSelectedSuperscaffold.getFeature2D());
                    } else if (selectedSuperscaffolds.contains(altTentativeSuperscaffoldId)) {
                        return;
                    } else {
                        selectedSuperscaffolds.add(tentativeSuperscaffoldId);
                        highlightedFeatures.add(newSelectedSuperscaffold.getFeature2D());
                    }

                    addHighlightedFeatures(highlightedFeatures);

                    superAdapter.getMainViewPanel().toggleToolTipUpdates(Boolean.TRUE);
                    superAdapter.updateMainViewPanelToolTipText(toolTipText(e.getX(), e.getY()));
                    superAdapter.getMainViewPanel().toggleToolTipUpdates(highlightedFeatures.isEmpty());

                    currentPromptedAssemblyAction = PromptedAssemblyAction.NONE;

                    restoreDefaultVariables();
                }
                return;
            }

            if (activelyEditingAssembly && HiCGlobals.splitModeEnabled && currentPromptedAssemblyAction == PromptedAssemblyAction.CUT) {
                // disable long click: it seems that no one is using it anyway. But let's keep it commented around for now..
//                    holdTime = (endTime - startTime) / Math.pow(10, 6);
                //Short click: execute split, long click: expert mode leave annotation be for editing purposes
//                    if (holdTime <= clickDelay) {
                debrisFeature = generateDebrisFeature(e, debrisFeatureSize);
                executeSplitMenuAction();
//                    }
                currentPromptedAssemblyAction = PromptedAssemblyAction.NONE;
            }
            if (activelyEditingAssembly && (dragMode == DragMode.ANNOTATE || currentPromptedAssemblyAction == PromptedAssemblyAction.ADJUST)) {
                // New annotation is added (not single click) and new feature from custom annotation

                updateSelectedFeatures(false);
                highlightedFeatures.clear();
                List<Feature2D> newSelectedFeatures = superAdapter.getMainLayer().getSelectedFeatures(hic, e.getX(), e.getY());

                // selects superscaffold
                if ((newSelectedFeatures == null || newSelectedFeatures.size() == 0) && selectedFeatures.size() == 0) {
                    Feature2DGuiContainer newSelectedSuperscaffold = getMouseHoverSuperscaffold(e.getX(), e.getY());

                    if (newSelectedSuperscaffold != null) {
                        final List<Integer> curScaffolds = superAdapter.getAssemblyStateTracker().getAssemblyHandler().getListOfSuperscaffolds().get(
                                Integer.parseInt(newSelectedSuperscaffold.getFeature2D().getAttribute("Superscaffold #")) - 1);

                        newSelectedFeatures.clear();
                        for (int scaffold : curScaffolds) {
                            Feature2D curScaffold = superAdapter.getAssemblyStateTracker().getAssemblyHandler().getListOfScaffolds().get(Math.abs(scaffold) - 1).getCurrentFeature2D();
                            newSelectedFeatures.add(curScaffold);
                        }
                    }
                }

                Collections.sort(newSelectedFeatures);

                // Damage rectangle is not precise, adjust boundaries...
                try {
                    if (currentPromptedAssemblyAction == PromptedAssemblyAction.ADJUST) {
                        if (adjustAnnotation == AdjustAnnotation.LEFT) {
                            while (!selectedFeatures.contains(newSelectedFeatures.get(newSelectedFeatures.size() - 1)) && !newSelectedFeatures.isEmpty()) {
                                newSelectedFeatures.remove(newSelectedFeatures.size() - 1);
                            }
                        } else {
                            while (!selectedFeatures.contains(newSelectedFeatures.get(0)) && !newSelectedFeatures.isEmpty()) {
                                newSelectedFeatures.remove(0);
                            }
                        }
                    }
                } catch (Exception e1) {
                    parent.removeSelection();
                }

                if (HiCGlobals.translationInProgress) {
                    translationInProgressMouseReleased(newSelectedFeatures);
                } else {
                    if (selectedFeatures.equals(newSelectedFeatures) && currentPromptedAssemblyAction != PromptedAssemblyAction.ADJUST) {
                        parent.removeSelection();
                    } else {
                        selectedFeatures.clear();
                        selectedFeatures.addAll(newSelectedFeatures);
                    }
                }
                updateSelectedFeatures(true);

                Chromosome chrX = superAdapter.getHiC().getXContext().getChromosome();
                Chromosome chrY = superAdapter.getHiC().getYContext().getChromosome();
                superAdapter.getEditLayer().filterTempSelectedGroup(chrX.getIndex(), chrY.getIndex());
                parent.repaint();

                if (!selectedFeatures.isEmpty()) {
                    if (superAdapter.getMainLayer().getLayerVisibility()) {
                        tempSelectedGroup = superAdapter.getEditLayer().addTempSelectedGroup(selectedFeatures, hic);
                        addHighlightedFeature(tempSelectedGroup);
                    }
                } else {
                    parent.removeHighlightedFeature();
                }

                //getAssemblyPopupMenu(e.getX(), e.getY()).show(parent, e.getX(), e.getY());

                superAdapter.getMainViewPanel().toggleToolTipUpdates(true);
                superAdapter.updateMainViewPanelToolTipText(toolTipText(e.getX(), e.getY()));
                superAdapter.getMainViewPanel().toggleToolTipUpdates(selectedFeatures.isEmpty());

                currentPromptedAssemblyAction = PromptedAssemblyAction.NONE;

                restoreDefaultVariables();
            } else if ((dragMode == DragMode.ZOOM || dragMode == DragMode.SELECT) && zoomRectangle != null) {
                Runnable runnable = new Runnable() {
                    @Override
                    public void run() {
                        unsafeDragging();
                    }
                };
                superAdapter.executeLongRunningTask(runnable, "Mouse Drag");
            } else if (dragMode == DragMode.ANNOTATE) {
                // New annotation is added (not single click) and new feature from custom annotation
                superAdapter.getActiveLayerHandler().addFeature(hic);
                restoreDefaultVariables();
            } else if (dragMode == DragMode.RESIZE) {
                // New annotation is added (not single click) and new feature from custom annotation
                int idx1 = preAdjustLoop.getFirst().getFirst();
                int idx2 = preAdjustLoop.getFirst().getSecond();

                Feature2D secondLoop = preAdjustLoop.getSecond();
                // Add a new loop if it was resized (prevents deletion on single click)

                try {
                    final double scaleFactor = hic.getScaleFactor();
                    final int screenWidth = parent.getBounds().width;
                    final int screenHeight = parent.getBounds().height;
                    int centerX = (int) (screenWidth / scaleFactor) / 2;
                    int centerY = (int) (screenHeight / scaleFactor) / 2;

                    if (superAdapter.getActiveLayerHandler().hasLoop(hic.getZd(), idx1, idx2, centerX, centerY,
                            Feature2DHandler.numberOfLoopsToFind, hic.getXContext().getBinOrigin(),
                            hic.getYContext().getBinOrigin(), hic.getScaleFactor(), secondLoop) && changedSize) {
                        Feature2D oldFeature2D = secondLoop.deepCopy();

                        superAdapter.getActiveLayerHandler().removeFromList(hic.getZd(), idx1, idx2, centerX, centerY,
                                Feature2DHandler.numberOfLoopsToFind, hic.getXContext().getBinOrigin(),
                                hic.getYContext().getBinOrigin(), hic.getScaleFactor(), secondLoop);

                        Feature2D tempFeature2D = superAdapter.getActiveLayerHandler().addFeature(hic);
                        superAdapter.getActiveLayerHandler().setLastItem(idx1, idx2, secondLoop);
                        for (String newKey : oldFeature2D.getAttributeKeys()) {
                            tempFeature2D.setAttribute(newKey, oldFeature2D.getAttribute(newKey));
                        }

                        //remove preadjust loop from list
                        if (activelyEditingAssembly && HiCGlobals.splitModeEnabled) {
                            debrisFeature = tempFeature2D;
                        }
                    }
                } catch (Exception ee) {
                    System.err.println("Unable to remove pre-resized loop");
                }
                restoreDefaultVariables();
            } else {
                setProperCursor();
            }
        }
    }

    // works for only one selected contig
    private void translationInProgressMouseReleased(List<Feature2D> newSelectedFeatures) {
        if (!selectedFeatures.isEmpty()) {
            Feature2D featureDestination = newSelectedFeatures.get(0);
            AssemblyOperationExecutor.moveSelection(superAdapter, selectedFeatures, featureDestination);
            parent.repaint();
        }

        if (newSelectedFeatures != null) {
            selectedFeatures.addAll(newSelectedFeatures);
        }
        HiCGlobals.translationInProgress = false;
        parent.removeSelection(); //TODO fix this so that highlight moves with translated selection
    }

    Feature2D generateDebrisFeature(final MouseEvent eF, int debrisFeatureSize) {
        final double scaleFactor = hic.getScaleFactor();
        double binOriginX = hic.getXContext().getBinOrigin();
        double binOriginY = hic.getYContext().getBinOrigin();
        Point mousePoint = eF.getPoint();
        double x = mousePoint.getX();
        double y = mousePoint.getY();
        int rightCorner = (int) Math.max(x, y + (binOriginY - binOriginX) * scaleFactor);
        Rectangle annotateRectangle = new Rectangle(rightCorner - debrisFeatureSize,
                (int) (rightCorner - debrisFeatureSize - (binOriginY - binOriginX) * scaleFactor), debrisFeatureSize, debrisFeatureSize);
        superAdapter.getEditLayer().updateSelectionRegion(annotateRectangle);
        debrisFeature = superAdapter.getEditLayer().generateFeature(hic);
        return debrisFeature;
    }

    private void restoreDefaultVariables() {
        dragMode = DragMode.NONE;
        adjustAnnotation = AdjustAnnotation.NONE;
        annotateRectangle = null;
        lastMousePoint = null;
        zoomRectangle = null;
        preAdjustLoop = null;
        hic.setCursorPoint(null);
        changedSize = false;
        parent.setCursor(Cursor.getDefaultCursor());
        parent.repaint();
        superAdapter.repaintTrackPanels();
    }

    private void unsafeDragging() {
        final double scaleFactor1 = hic.getScaleFactor();
        double binX = hic.getXContext().getBinOrigin() + (zoomRectangle.x / scaleFactor1);
        double binY = hic.getYContext().getBinOrigin() + (zoomRectangle.y / scaleFactor1);
        double wBins = (int) (zoomRectangle.width / scaleFactor1);
        double hBins = (int) (zoomRectangle.height / scaleFactor1);

        try {
            final MatrixZoomData currentZD = hic.getZd();
            long xBP0 = currentZD.getXGridAxis().getGenomicStart(binX);

            long yBP0 = currentZD.getYGridAxis().getGenomicEnd(binY);

            double newXBinSize = wBins * currentZD.getBinSize() / parent.getWidth();
            double newYBinSize = hBins * currentZD.getBinSize() / parent.getHeight();
            double newBinSize = Math.max(newXBinSize, newYBinSize);

            hic.zoomToDrawnBox(xBP0, yBP0, newBinSize);
        } catch (Exception e) {
            e.printStackTrace();
        }

        dragMode = DragMode.NONE;
        lastMousePoint = null;
        zoomRectangle = null;
        setProperCursor();
    }

    @Override
    final public void mouseDragged(final MouseEvent e) {

        Rectangle lastRectangle, damageRect;
        int x, y;
        double x_d, y_d;

        try {
            hic.getZd();
        } catch (Exception ex) {
            return;
        }

        if (hic.isWholeGenome()) {
            return;
        }

        if (lastMousePoint == null) {
            lastMousePoint = e.getPoint();
            return;
        }

        int deltaX = e.getX() - lastMousePoint.x;
        int deltaY = e.getY() - lastMousePoint.y;
        double deltaX_d = e.getX() - lastMousePoint.x;
        double deltaY_d = e.getY() - lastMousePoint.y;

        if (dragMode == DragMode.ZOOM) {
            lastRectangle = zoomRectangle;

            if (deltaX == 0 || deltaY == 0) {
                return;
            }

            // Constrain aspect ratio of zoom rectangle to that of panel
            double aspectRatio = (double) parent.getWidth() / parent.getHeight();
            if (deltaX * aspectRatio > deltaY) {
                deltaY = (int) (deltaX / aspectRatio);
            } else {
                deltaX = (int) (deltaY * aspectRatio);
            }

            x = deltaX > 0 ? lastMousePoint.x : lastMousePoint.x + deltaX;
            y = deltaY > 0 ? lastMousePoint.y : lastMousePoint.y + deltaY;
            zoomRectangle = new Rectangle(x, y, Math.abs(deltaX), Math.abs(deltaY));

            damageRect = lastRectangle == null ? zoomRectangle : zoomRectangle.union(lastRectangle);
            damageRect.x--;
            damageRect.y--;
            damageRect.width += 2;
            damageRect.height += 2;
            parent.paintImmediately(damageRect);
        } else if (dragMode == DragMode.ANNOTATE) {
            lastRectangle = annotateRectangle;

            if (deltaX_d == 0 || deltaY_d == 0) {
                return;
            }

            x = deltaX > 0 ? lastMousePoint.x : lastMousePoint.x + deltaX;
            y = deltaY > 0 ? lastMousePoint.y : lastMousePoint.y + deltaY;
            annotateRectangle = new Rectangle(x, y, Math.abs(deltaX), Math.abs(deltaY));

            damageRect = lastRectangle == null ? annotateRectangle : annotateRectangle.union(lastRectangle);
            superAdapter.getActiveLayerHandler().updateSelectionRegion(damageRect);
            damageRect.x--;
            damageRect.y--;
            damageRect.width += 2;
            damageRect.height += 2;
            parent.paintImmediately(damageRect);
        } else if (dragMode == DragMode.RESIZE) {
            if (deltaX_d == 0 || deltaY_d == 0) {
                return;
            }

            lastRectangle = annotateRectangle;
            double rectX;
            double rectY;

            // Resizing upper left corner
            if (adjustAnnotation == AdjustAnnotation.LEFT) {
                rectX = annotateRectangle.getX() + annotateRectangle.getWidth();
                rectY = annotateRectangle.getY() + annotateRectangle.getHeight();
                // Resizing lower right corner
            } else {
                rectX = annotateRectangle.getX();
                rectY = annotateRectangle.getY();
            }
            deltaX_d = e.getX() - rectX;
            deltaY_d = e.getY() - rectY;

            x_d = deltaX_d > 0 ? rectX : rectX + deltaX_d;
            y_d = deltaY_d > 0 ? rectY : rectY + deltaY_d;

            annotateRectangle = new Rectangle((int) x_d, (int) y_d, (int) Math.abs(deltaX_d), (int) Math.abs(deltaY_d));
            damageRect = lastRectangle == null ? annotateRectangle : annotateRectangle.union(lastRectangle);
            damageRect.width += 1;
            damageRect.height += 1;
            parent.paintImmediately(damageRect);
            superAdapter.getActiveLayerHandler().updateSelectionRegion(damageRect);
            changedSize = true;
        } else {
            lastMousePoint = e.getPoint();    // Always save the last Point

            double deltaXBins = -deltaX / hic.getScaleFactor();
            double deltaYBins = -deltaY / hic.getScaleFactor();
            hic.moveBy(deltaXBins, deltaYBins);
        }
    }

    @Override
    public void mouseMoved(MouseEvent e) {
        try {
            hic.getZd();
        } catch (Exception ex) {
            return;
        }
        if (hic.getXContext() != null) {
            adjustAnnotation = AdjustAnnotation.NONE;
            currentPromptedAssemblyAction = PromptedAssemblyAction.NONE;
            // Update tool tip text
            if (!featureOptionMenuEnabled) {
                superAdapter.updateMainViewPanelToolTipText(toolTipText(e.getX(), e.getY()));
            }
            // Set check if hovering over feature corner

            // Following was commented out since it was causing flickering of the cursor on windows machines, don't know if was necessary
//        parent.setCursor(Cursor.getDefaultCursor());
            int minDist = 20;
            if (currentFeature != null) {

                boolean resizeable = (currentFeature.getAnnotationLayerHandler().getAnnotationLayerType() != AnnotationLayer.LayerType.SCAFFOLD) && (currentFeature.getAnnotationLayerHandler().getAnnotationLayerType() != AnnotationLayer.LayerType.SUPERSCAFFOLD);
//                    if (activelyEditingAssembly) {
//                        resizeable = (resizeable && HiCGlobals.splitModeEnabled);
//                    }
                if (resizeable) {
                    Rectangle loop = currentFeature.getRectangle();
                    Point mousePoint = e.getPoint();
                    // Mouse near top left corner
                    if ((Math.abs(loop.getMinX() - mousePoint.getX()) <= minDist &&
                            Math.abs(loop.getMinY() - mousePoint.getY()) <= minDist)) {
                        adjustAnnotation = AdjustAnnotation.LEFT;
                        parent.setCursor(Cursor.getPredefinedCursor(Cursor.NW_RESIZE_CURSOR));
                        // Mouse near bottom right corner
                    } else if (Math.abs(loop.getMaxX() - mousePoint.getX()) <= minDist &&
                            Math.abs(loop.getMaxY() - mousePoint.getY()) <= minDist) {
                        adjustAnnotation = AdjustAnnotation.RIGHT;
                        parent.setCursor(Cursor.getPredefinedCursor(Cursor.SE_RESIZE_CURSOR));
                    }
                }

            }
            if (activelyEditingAssembly && !allMainFeaturePairs.isEmpty() && !e.isShiftDown()) {

                final double scaleFactor = hic.getScaleFactor();
                double binOriginX = hic.getXContext().getBinOrigin();
                double binOriginY = hic.getYContext().getBinOrigin();

                Point mousePoint = e.getPoint();
                double x = mousePoint.getX();
                double y = mousePoint.getY();

                // this is a good place to handle inserts to top and bottom as it should be done even if individual
                // features at the beginning of the assembly are not visible
                // find the x and y in relation to the displayed screen
                int topLeftCornerX = (int) ((0 - binOriginX) * scaleFactor);
                int topLeftCornerY = (int) ((0 - binOriginY) * scaleFactor);

                List<Scaffold> listOfScaffolds =
                        superAdapter.getAssemblyStateTracker().getAssemblyHandler().getListOfAggregateScaffolds();
                long lastGenomicBin = 0;
                try {
                    lastGenomicBin = listOfScaffolds.get(listOfScaffolds.size() - 1).getCurrentFeature2D().getEnd2() /
                            hic.getZd().getBinSize();
                } catch (NullPointerException e1) {
                    e1.printStackTrace();
                }
                int bottomRightCornerX = (int) ((lastGenomicBin - binOriginX) * scaleFactor);
                int bottomRightCornerY = (int) ((lastGenomicBin - binOriginY) * scaleFactor);

                if (!selectedFeatures.isEmpty() && !HiCGlobals.phasing) {
                    if (mousePoint.getX() - topLeftCornerX >= 0 &&
                            mousePoint.getX() - topLeftCornerX <= minDist &&
                            mousePoint.getY() - topLeftCornerY >= 0 &&
                            mousePoint.getY() - topLeftCornerY <= minDist) {
                        parent.setCursor(MainWindow.pasteNWCursor);
                        currentPromptedAssemblyAction = PromptedAssemblyAction.PASTETOP;
                    }
                    if (bottomRightCornerX - mousePoint.getX() >= 0 &&
                            bottomRightCornerX - mousePoint.getX() <= minDist &&
                            bottomRightCornerY - mousePoint.getY() >= 0 &&
                            bottomRightCornerY - mousePoint.getY() <= minDist) {
                        parent.setCursor(MainWindow.pasteSECursor);
                        currentPromptedAssemblyAction = PromptedAssemblyAction.PASTEBOTTOM;
                    }
                }

                currentUpstreamFeature = null;
                currentDownstreamFeature = null;

                for (Feature2DGuiContainer asmFragment : allMainFeaturePairs) {
                    if (asmFragment.getRectangle().contains(x, x + (binOriginX - binOriginY) * scaleFactor)) {
                        currentUpstreamFeature = asmFragment;
                    }
                    if (asmFragment.getRectangle().contains(y + (binOriginY - binOriginX) * scaleFactor, y)) {
                        currentDownstreamFeature = asmFragment;
                    }
                }

                if (currentUpstreamFeature != null && currentDownstreamFeature != null) {
                    if (currentUpstreamFeature.getFeature2D().getStart1() > currentDownstreamFeature.getFeature2D().getStart1()) {
                        Feature2DGuiContainer temp = currentUpstreamFeature;
                        currentUpstreamFeature = currentDownstreamFeature;
                        currentDownstreamFeature = temp;
                    }

                    // inserting from highlight: keeping for future development
//            if (!selectedFeatures.isEmpty()) {
//
//              // upstream feature is the same
//              if (currentUpstreamFeature.getFeature2D().getStart1() >= selectedFeatures.get(0).getStart1() &&
//                  currentUpstreamFeature.getFeature2D().getEnd1() <=
//                      selectedFeatures.get(selectedFeatures.size() - 1).getEnd1()) {
//
//                int topYright = currentUpstreamFeature.getRectangle().y;
//                int bottomYright =
//                    currentUpstreamFeature.getRectangle().y + (int) currentUpstreamFeature.getRectangle().getHeight();
//                int leftXbottom = currentUpstreamFeature.getRectangle().x;
//                int rightXbottom =
//                    currentUpstreamFeature.getRectangle().x + (int) currentUpstreamFeature.getRectangle().getWidth();
//
//                if (mousePoint.getY() >= topYright && mousePoint.getY() <= bottomYright) {
//
//                  if ((mousePoint.getX() >= currentDownstreamFeature.getRectangle().getMinX() &&
//                      mousePoint.getX() <= currentDownstreamFeature.getRectangle().getMinX() + minDist)) {
//
//                    // if the start doesn't match the end of the previous one, there's a gap, do not insert
//                    if (currentDownstreamFeature.getFeature2D().getStart1() ==
//                        allMainFeaturePairs.get(idxUp - 1).getFeature2D().getEnd1()) {
//                      parent.setCursor(MainWindow.pasteSWCursor);
//                      currentUpstreamFeature = allMainFeaturePairs.get(idxUp - 1);
//                      currentPromptedAssemblyAction = PromptedAssemblyAction.PASTE;
//                    }
//                  }
//                }
//
//                if (mousePoint.getX() >= leftXbottom && mousePoint.getX() <= rightXbottom) {
//                  // -y axis
//                  if ((mousePoint.getY() >= currentDownstreamFeature.getRectangle().getMinY() &&
//                      mousePoint.getY() <= currentDownstreamFeature.getRectangle().getMinY() + minDist)) {
//                    if (currentDownstreamFeature.getFeature2D().getStart1() ==
//                        allMainFeaturePairs.get(idxDown - 1).getFeature2D().getEnd1()) {
//                      parent.setCursor(MainWindow.pasteNECursor);
//                      currentUpstreamFeature = allMainFeaturePairs.get(idxDown - 1);
//                      currentPromptedAssemblyAction = PromptedAssemblyAction.PASTE;
//                    }
//                  }
//                }
//              }
//
//              // downstream feature the same
//              if ((currentDownstreamFeature.getFeature2D().getStart1() >= selectedFeatures.get(0).getStart1() &&
//                  currentDownstreamFeature.getFeature2D().getEnd1() <=
//                      selectedFeatures.get(selectedFeatures.size() - 1).getEnd1())
//                  ) {
//                int topYleft = currentDownstreamFeature.getRectangle().y;
//                int bottomYleft =
//                    currentDownstreamFeature.getRectangle().y +
//                        (int) currentDownstreamFeature.getRectangle().getHeight();
//                int leftXtop = currentDownstreamFeature.getRectangle().x;
//                int rightXtop =
//                    currentDownstreamFeature.getRectangle().x +
//                        (int) currentDownstreamFeature.getRectangle().getWidth();
//
//                // y axis
//                if (mousePoint.getX() >= leftXtop && mousePoint.getX() <= rightXtop) {
//                  if ((mousePoint.getY() >= currentUpstreamFeature.getRectangle().getMaxY() - minDist &&
//                      mousePoint.getY() <= currentUpstreamFeature.getRectangle().getMaxY())) {
//                    parent.setCursor(MainWindow.pasteSWCursor);
//                    currentPromptedAssemblyAction = PromptedAssemblyAction.PASTE;
//                  }
//                }
//                // -x axis
//                else if (mousePoint.getY() >= topYleft && mousePoint.getY() <= bottomYleft) {
//                  if ((mousePoint.getX() >= currentUpstreamFeature.getRectangle().getMaxX() - minDist &&
//                      mousePoint.getX() <= (currentUpstreamFeature.getRectangle().getMaxX()))) {
//                    parent.setCursor(MainWindow.pasteNECursor);
//                    currentPromptedAssemblyAction = PromptedAssemblyAction.PASTE;
//                  }
//                }
//              }
//            }

                    if (!HiCGlobals.splitModeEnabled && (currentUpstreamFeature.getFeature2D().getEnd1() == currentDownstreamFeature.getFeature2D().getStart1()) || (currentDownstreamFeature == null && currentUpstreamFeature == null)) {

                        if ((mousePoint.getX() - currentUpstreamFeature.getRectangle().getMaxX() >= 0) &&
                                (mousePoint.getX() - currentUpstreamFeature.getRectangle().getMaxX() <= minDist) &&
                                (currentUpstreamFeature.getRectangle().getMaxY() - mousePoint.getY() >= 0) &&
                                (currentUpstreamFeature.getRectangle().getMaxY() - mousePoint.getY() <= minDist)) {
                            if (selectedFeatures == null || selectedFeatures.isEmpty()) {
                                parent.setCursor(MainWindow.groupSWCursor);
                                currentPromptedAssemblyAction = PromptedAssemblyAction.REGROUP;
                            } else if (!(currentUpstreamFeature.getFeature2D().getEnd1() >= selectedFeatures.get(0).getStart1() &&
                                    currentUpstreamFeature.getFeature2D().getEnd1() <=
                                            selectedFeatures.get(selectedFeatures.size() - 1).getEnd1())) {
                                parent.setCursor(MainWindow.pasteSWCursor);
                                currentPromptedAssemblyAction = PromptedAssemblyAction.PASTE;
                            }
                        } else if ((currentUpstreamFeature.getRectangle().getMaxX() - mousePoint.getX() >= 0) &&
                                (currentUpstreamFeature.getRectangle().getMaxX() - mousePoint.getX() <= minDist) &&
                                (mousePoint.getY() - currentUpstreamFeature.getRectangle().getMaxY() >= 0) &&
                                (mousePoint.getY() - currentUpstreamFeature.getRectangle().getMaxY() <= minDist)) {
                            if (selectedFeatures.isEmpty()) {
                                parent.setCursor(MainWindow.groupNECursor);
                                currentPromptedAssemblyAction = PromptedAssemblyAction.REGROUP;
                            } else if (!(currentUpstreamFeature.getFeature2D().getEnd1() >= selectedFeatures.get(0).getStart1() &&
                                    currentUpstreamFeature.getFeature2D().getEnd1() <=
                                            selectedFeatures.get(selectedFeatures.size() - 1).getEnd1())) {
                                parent.setCursor(MainWindow.pasteNECursor);
                                currentPromptedAssemblyAction = PromptedAssemblyAction.PASTE;
                            }
                        }
                    }
                }

                if (!HiCGlobals.splitModeEnabled && !selectedFeatures.isEmpty()) {

                    for (Feature2DGuiContainer asmFragment : allEditFeaturePairs) {
                        if (asmFragment.getFeature2D().equals(tempSelectedGroup) && !asmFragment.getFeature2D().equals(debrisFeature)) {
                            if (Math.abs(asmFragment.getRectangle().getMaxX() - mousePoint.getX()) < minDist &&
                                    Math.abs(asmFragment.getRectangle().getMinY() - mousePoint.getY()) < minDist) {
                                parent.setCursor(MainWindow.invertSWCursor);
                                if (debrisFeature != null) {
                                    int chr1Idx = hic.getXContext().getChromosome().getIndex();
                                    int chr2Idx = hic.getYContext().getChromosome().getIndex();
                                    superAdapter.getEditLayer().getAnnotationLayer().getFeatureHandler().getFeatureList().checkAndRemoveFeature(chr1Idx, chr2Idx, debrisFeature);
                                }
                                currentPromptedAssemblyAction = PromptedAssemblyAction.INVERT;
                            } else if (Math.abs(asmFragment.getRectangle().getMinX() - mousePoint.getX()) < minDist &&
                                    Math.abs(asmFragment.getRectangle().getMaxY() - mousePoint.getY()) < minDist) {
                                parent.setCursor(MainWindow.invertNECursor);
                                if (debrisFeature != null) {
                                    int chr1Idx = hic.getXContext().getChromosome().getIndex();
                                    int chr2Idx = hic.getYContext().getChromosome().getIndex();
                                    superAdapter.getEditLayer().getAnnotationLayer().getFeatureHandler().getFeatureList().checkAndRemoveFeature(chr1Idx, chr2Idx, debrisFeature);
                                }
                                currentPromptedAssemblyAction = PromptedAssemblyAction.INVERT;
                            } else if (selectedFeatures.size() == 1 && Math.abs(x - (y + (binOriginY - binOriginX) * scaleFactor)) < minDist &&
                                    Math.abs(y - (x + (binOriginX - binOriginY) * scaleFactor)) < minDist &&
                                    x - asmFragment.getRectangle().getMinX() > debrisFeatureSize + RESIZE_SNAP + scaleFactor &&
                                    asmFragment.getRectangle().getMaxX() - x > RESIZE_SNAP + scaleFactor &&
                                    y - asmFragment.getRectangle().getMinY() > debrisFeatureSize + RESIZE_SNAP + scaleFactor &&
                                    asmFragment.getRectangle().getMaxY() - y > RESIZE_SNAP + scaleFactor) {
                                parent.setCursor(MainWindow.scissorCursor);
                                currentPromptedAssemblyAction = PromptedAssemblyAction.CUT;

                                int chr1Idx = hic.getXContext().getChromosome().getIndex();
                                int chr2Idx = hic.getYContext().getChromosome().getIndex();
                                if (debrisFeature != null) {
                                    superAdapter.getEditLayer().getAnnotationLayer().getFeatureHandler().getFeatureList().checkAndRemoveFeature(chr1Idx, chr2Idx, debrisFeature);
                                }
                                generateDebrisFeature(e, debrisFeatureSize);
                                superAdapter.getEditLayer().getAnnotationLayer().add(chr1Idx, chr2Idx, debrisFeature);
                            } else if (Math.abs(x - asmFragment.getRectangle().getMinX()) <= RESIZE_SNAP &&
                                    Math.abs(y - asmFragment.getRectangle().getMinY()) <= RESIZE_SNAP &&
                                    y + x < asmFragment.getRectangle().getMaxX() + asmFragment.getRectangle().getMinY()) {
                                parent.setCursor(Cursor.getPredefinedCursor(Cursor.NW_RESIZE_CURSOR));
                                currentPromptedAssemblyAction = PromptedAssemblyAction.ADJUST;
                                adjustAnnotation = AdjustAnnotation.LEFT;
                            } else if (Math.abs(asmFragment.getRectangle().getMaxX() - x) <= RESIZE_SNAP &&
                                    Math.abs(asmFragment.getRectangle().getMaxY() - y) <= RESIZE_SNAP &&
                                    y + x > asmFragment.getRectangle().getMaxX() + asmFragment.getRectangle().getMinY()) {
                                parent.setCursor(Cursor.getPredefinedCursor(Cursor.SE_RESIZE_CURSOR));
                                currentPromptedAssemblyAction = PromptedAssemblyAction.ADJUST;
                                adjustAnnotation = AdjustAnnotation.RIGHT;
                            } else if (debrisFeature != null) {
                                int chr1Idx = hic.getXContext().getChromosome().getIndex();
                                int chr2Idx = hic.getYContext().getChromosome().getIndex();
                                superAdapter.getEditLayer().getAnnotationLayer().getFeatureHandler().getFeatureList().checkAndRemoveFeature(chr1Idx, chr2Idx, debrisFeature);
                            }
                        }
                    }
                }
            }

            if (hic.isWholeGenome()) {
                synchronized (this) {
                    hic.setGWCursorPoint(e.getPoint());
                    superAdapter.repaintGridRulerPanels();
                }
            } else {
                hic.setGWCursorPoint(null);
            }

            if (straightEdgeEnabled || e.isShiftDown()) {
                synchronized (this) {
                    hic.setCursorPoint(e.getPoint());
                    superAdapter.repaintTrackPanels();
                }
            } else if (diagonalEdgeEnabled) {
                synchronized (this) {
                    hic.setDiagonalCursorPoint(e.getPoint());
                    superAdapter.repaintTrackPanels();
                }
            } else if (adjustAnnotation == AdjustAnnotation.NONE && currentPromptedAssemblyAction == PromptedAssemblyAction.NONE) {
                hic.setCursorPoint(null);
                parent.setCursor(Cursor.getDefaultCursor());
            }
            parent.repaint();
        }

    }

    @Override
    public void mouseWheelMoved(MouseWheelEvent e) {
        try {
            if (currentPromptedAssemblyAction == PromptedAssemblyAction.CUT) {

                final double scaleFactor = hic.getScaleFactor();
                //double binOriginX = hic.getXContext().getBinOrigin();
                //double binOriginY = hic.getYContext().getBinOrigin();
                Point mousePoint = e.getPoint();
                double x = mousePoint.getX();
                double y = mousePoint.getY();
                int rightCorner = (int) Math.max(x, y);

                debrisFeatureSize = debrisFeatureSize - e.getUnitsToScroll();
                if (rightCorner - debrisFeatureSize < currentFeature.getRectangle().getMinX() + RESIZE_SNAP) {
                    debrisFeatureSize = rightCorner - (int) currentFeature.getRectangle().getMinX() - RESIZE_SNAP - 1;
                }
                if (debrisFeatureSize <= scaleFactor) {
                    debrisFeatureSize = (int) Math.max(scaleFactor, 1);
                }

                int chr1Idx = hic.getXContext().getChromosome().getIndex();
                int chr2Idx = hic.getYContext().getChromosome().getIndex();
                if (debrisFeature != null) {
                    superAdapter.getEditLayer().getAnnotationLayer().getFeatureHandler().getFeatureList().checkAndRemoveFeature(chr1Idx, chr2Idx, debrisFeature);
                }
                generateDebrisFeature(e, debrisFeatureSize);
                superAdapter.getEditLayer().getAnnotationLayer().add(chr1Idx, chr2Idx, debrisFeature);
                parent.repaint();
                return;
            }
            int scroll = (int) Math.round(e.getPreciseWheelRotation());

            hic.moveBy(scroll, scroll);
            superAdapter.updateMainViewPanelToolTipText(toolTipText(e.getX(), e.getY()));
        } catch (Exception e2) {
            parent.repaint();
        }
    }

    private Point calculateSelectionPoint(int unscaledX, int unscaledY) {
        final MatrixZoomData zd;
        try {
            zd = hic.getZd();
        } catch (Exception err) {
            return null;
        }

        final HiCGridAxis xAxis = zd.getXGridAxis();
        final HiCGridAxis yAxis = zd.getYGridAxis();
        final double binOriginX = hic.getXContext().getBinOrigin();
        final double binOriginY = hic.getYContext().getBinOrigin();
        final double scale = hic.getScaleFactor();

        float x = (float) (((unscaledX / scale) + binOriginX) * xAxis.getBinSize());
        float y = (float) (((unscaledY / scale) + binOriginY) * yAxis.getBinSize());
        return new Point((int) x, (int) y);
    }

    private void setDebrisFeauture(Feature2D debrisFeature) {
        this.debrisFeature = debrisFeature;
    }

    private Feature2DGuiContainer getMouseHoverSuperscaffold(int x, int y) {
        final Point mousePoint = calculateSelectionPoint(x, y);

        if (activelyEditingAssembly) {
            for (Feature2DGuiContainer loop : allFeaturePairs) {
                if (loop.getFeature2D().getFeatureType() == Feature2D.FeatureType.SUPERSCAFFOLD) {
                    if (loop.getFeature2D().containsPoint(mousePoint)) {
                        return loop;
                    }
                }
            }
        }

        return null;
    }

    private String toolTipText(int x, int y) {
        // Update popup text
        final MatrixZoomData zd;
        HiCGridAxis xGridAxis, yGridAxis;
        try {
            zd = hic.getZd();
            xGridAxis = zd.getXGridAxis();
            yGridAxis = zd.getYGridAxis();
        } catch (Exception e) {
            return "";
        }

        int binX = (int) (hic.getXContext().getBinOrigin() + x / hic.getScaleFactor());
        int binY = (int) (hic.getYContext().getBinOrigin() + y / hic.getScaleFactor());

        long xGenomeStart = xGridAxis.getGenomicStart(binX) + 1; // Conversion from in internal "0" -> 1 base coordinates
        long yGenomeStart = yGridAxis.getGenomicStart(binY) + 1;
        long xGenomeEnd = xGridAxis.getGenomicEnd(binX);
        long yGenomeEnd = yGridAxis.getGenomicEnd(binY);

        if (hic.isWholeGenome()) {

            final long[] chromosomeBoundaries = parent.getChromosomeBoundaries();
            Chromosome xChrom = getChromFromBoundaries(chromosomeBoundaries, xGenomeStart);
            Chromosome yChrom = getChromFromBoundaries(chromosomeBoundaries, yGenomeStart);

            if (xChrom != null && yChrom != null) {

                long leftBoundaryX = xChrom.getIndex() == 1 ? 0 : chromosomeBoundaries[xChrom.getIndex() - 2];
                long leftBoundaryY = yChrom.getIndex() == 1 ? 0 : chromosomeBoundaries[yChrom.getIndex() - 2];

                long xChromPos = (xGenomeStart - leftBoundaryX) * 1000;
                long yChromPos = (yGenomeStart - leftBoundaryY) * 1000;

                String txt = "";
                txt += "<html><span style='color:" + HiCGlobals.topChromosomeColor + "; font-family: arial; font-size: 12pt;'>";
                txt += xChrom.getName();
                txt += ":";
                txt += String.valueOf(xChromPos);
                txt += "</span><br><span style='color:" + HiCGlobals.leftChromosomeColor + "; font-family: arial; font-size: 12pt;'>";
                txt += yChrom.getName();
                txt += ":";
                txt += String.valueOf(yChromPos);
                txt += "</span></html>";

                if (xChrom.getName().toLowerCase().contains("chr")) {
                    hic.setXPosition(xChrom.getName() + ":" + xChromPos);
                } else {
                    hic.setXPosition("chr" + xChrom.getName() + ":" + xChromPos);
                }
                if (yChrom.getName().toLowerCase().contains("chr")) {
                    hic.setYPosition(yChrom.getName() + ":" + yChromPos);
                } else {
                    hic.setYPosition("chr" + yChrom.getName() + ":" + yChromPos);
                }
                return txt;
            }

        } else {

            //Update Position in hic. Used for clipboard copy:
            if (hic.getXContext().getChromosome().getName().toLowerCase().contains("chr")) {
                hic.setXPosition(hic.getXContext().getChromosome().getName() + ":" + formatter.format(xGenomeStart) + "-" + formatter.format(xGenomeEnd));
            } else {
                hic.setXPosition("chr" + hic.getXContext().getChromosome().getName() + ":" + formatter.format(xGenomeStart) + "-" + formatter.format(xGenomeEnd));
            }
            if (hic.getYContext().getChromosome().getName().toLowerCase().contains("chr")) {
                hic.setYPosition(hic.getYContext().getChromosome().getName() + ":" + formatter.format(yGenomeStart) + "-" + formatter.format(yGenomeEnd));
            } else {
                hic.setYPosition("chr" + hic.getYContext().getChromosome().getName() + ":" + formatter.format(yGenomeStart) + "-" + formatter.format(yGenomeEnd));
            }

            //int binX = (int) ((mainWindow.xContext.getOrigin() + e.getX() * mainWindow.xContext.getScale()) / getBinWidth());
            //int binY = (int) ((mainWindow.yContext.getOrigin() + e.getY() * mainWindow.yContext.getScale()) / getBinWidth());
            StringBuilder txt = new StringBuilder();

            txt.append("<html><span style='color:" + HiCGlobals.topChromosomeColor + "; font-family: arial; font-size: 12pt; '>");
            txt.append(hic.getXContext().getChromosome().getName());
            txt.append(":");
            txt.append(formatter.format(Math.round((xGenomeStart - 1) * HiCGlobals.hicMapScale + 1)));
            txt.append("-");
            txt.append(formatter.format(Math.round(xGenomeEnd) * HiCGlobals.hicMapScale));

            if (xGridAxis instanceof HiCFragmentAxis) {
                String fragNumbers;
                int binSize = zd.getZoom().getBinSize();
                if (binSize == 1) {
                    fragNumbers = formatter.format(binX);
                } else {
                    int leftFragment = binX * binSize;
                    int rightFragment = ((binX + 1) * binSize) - 1;
                    fragNumbers = formatter.format(leftFragment) + "-" + formatter.format(rightFragment);
                }
                txt.append("  (");
                txt.append(fragNumbers);
                txt.append("  len=");
                txt.append(formatter.format(xGenomeEnd - xGenomeStart));
                txt.append(")");
            }

            txt.append("</span><br><span style='color:" + HiCGlobals.leftChromosomeColor + "; font-family: arial; font-size: 12pt; '>");
            txt.append(hic.getYContext().getChromosome().getName());
            txt.append(":");
            txt.append(formatter.format(Math.round((yGenomeStart - 1) * HiCGlobals.hicMapScale + 1)));
            txt.append("-");
            txt.append(formatter.format(Math.round(yGenomeEnd * HiCGlobals.hicMapScale)));

            if (yGridAxis instanceof HiCFragmentAxis) {
                String fragNumbers;
                int binSize = zd.getZoom().getBinSize();
                if (binSize == 1) {
                    fragNumbers = formatter.format(binY);
                } else {
                    int leftFragment = binY * binSize;
                    int rightFragment = ((binY + 1) * binSize) - 1;
                    fragNumbers = formatter.format(leftFragment) + "-" + formatter.format(rightFragment);
                }
                txt.append("  (");
                txt.append(fragNumbers);
                txt.append("  len=");
                txt.append(formatter.format(yGenomeEnd - yGenomeStart));
                txt.append(")");
            }
            txt.append("</span><span style='font-family: arial; font-size: 12pt;'>");

            if (hic.isInPearsonsMode()) {
                float value = zd.getPearsonValue(binX, binY, hic.getObsNormalizationType());
                if (!Float.isNaN(value)) {

                    txt.append("<br><span style='font-family: arial; font-size: 12pt;'>");
                    txt.append("value = ");
                    txt.append(value);
                    txt.append("</span>");

                }
            } else {
                float value = hic.getNormalizedObservedValue(binX, binY);
                if (!Float.isNaN(value)) {
                    txt.append("<br><span style='font-family: arial; font-size: 12pt;'>");
                    txt.append("observed value (O) = ");
                    txt.append(getFloatString(value));
                    txt.append("</span>");
                }

                int c1 = hic.getXContext().getChromosome().getIndex();
                int c2 = hic.getYContext().getChromosome().getIndex();

                double ev = getExpectedValue(c1, c2, binX, binY, zd, hic.getExpectedValues());
                String evString = ev < 0.001 || Double.isNaN(ev) ? String.valueOf(ev) : formatter.format(ev);
                txt.append("<br><span style='font-family: arial; font-size: 12pt;'>expected value (E) = ").append(evString).append("</span>");
                if (ev > 0 && !Float.isNaN(value)) {
                    txt.append("<br><span style='font-family: arial; font-size: 12pt;'>O/E            = ");
                    txt.append(formatter.format(value / ev)).append("</span>");
                } else {
                    txt.append("<br><span style='font-family: arial; font-size: 12pt;'>O/E            = NaN</span>");
                }

                MatrixZoomData controlZD = hic.getControlZd();
                if (controlZD != null) {
                    float controlValue = hic.getNormalizedControlValue(binX, binY);
                    txt.append("<br><br><span style='font-family: arial; font-size: 12pt;'>");
                    txt.append("control value (C) = ");
                    txt.append(getFloatString(controlValue));
                    txt.append("</span>");

                    double evCtrl = getExpectedValue(c1, c2, binX, binY, controlZD, hic.getExpectedControlValues());
                    String evStringCtrl = evCtrl < 0.001 || Double.isNaN(evCtrl) ? String.valueOf(evCtrl) : formatter.format(evCtrl);
                    txt.append("<br><span style='font-family: arial; font-size: 12pt;'>expected control value (EC) = ").append(evStringCtrl).append("</span>");
                    if (evCtrl > 0 && !Float.isNaN(controlValue)) {
                        txt.append("<br><span style='font-family: arial; font-size: 12pt;'>C/EC            = ");
                        txt.append(formatter.format(controlValue / evCtrl)).append("</span>");
                        txt.append("<br><br><span style='font-family: arial; font-size: 12pt;'>(O/E)/(C/EC)            = ");
                        txt.append(formatter.format((value / ev) / (controlValue / evCtrl))).append("</span>");
                    } else {
                        txt.append("<br><span style='font-family: arial; font-size: 12pt;'>C/EC            = NaN</span>");
                    }

                    double obsAvg = zd.getAverageCount();
                    double obsValue = (value / obsAvg);
                    txt.append("<br><br><span style='font-family: arial; font-size: 12pt;'>");
                    txt.append("average observed value (AVG) = ").append(getFloatString((float) obsAvg));
                    txt.append("<br>O' = O/AVG = ").append(getFloatString((float) obsValue));
                    txt.append("</span>");

                    double ctrlAvg = controlZD.getAverageCount();
                    double ctlValue = (float) (controlValue / ctrlAvg);
                    txt.append("<br><span style='font-family: arial; font-size: 12pt;'>");
                    txt.append("average control value (AVGC) = ").append(getFloatString((float) ctrlAvg));
                    txt.append("<br>C' = C/AVGC = ").append(getFloatString((float) ctlValue));
                    txt.append("</span>");

                    if (value > 0 && controlValue > 0) {
                        double ratio = obsValue / ctlValue;
                        txt.append("<br><span style='font-family: arial; font-size: 12pt;'>");
                        txt.append("O'/C' = ").append(getFloatString((float) ratio));
                        txt.append("</span>");

                        double diff = (obsValue - ctlValue) * (obsAvg / 2. + ctrlAvg / 2.);
                        txt.append("<br><span style='font-family: arial; font-size: 12pt;'>");
                        txt.append("(O'-C')*(AVG/2 + AVGC/2) = ");
                        txt.append(getFloatString((float) diff));
                        txt.append("</span>");
                    }
                }

                txt.append(superAdapter.getTrackPanelPrintouts(x, y));
            }

            Point currMouse = new Point(x, y);
            double minDistance = Double.POSITIVE_INFINITY;
            //mouseIsOverFeature = false;
            currentFeature = null;
            if (activelyEditingAssembly) {
                // current feature is populated only from all main feature pairs, contains does not work
                for (Feature2DGuiContainer loop : allMainFeaturePairs) {
                    if (loop.getRectangle().contains(x, y)) {
                        currentFeature = loop;
                    }
                }

                if (!selectedFeatures.isEmpty()) {
                    Collections.sort(selectedFeatures);
                    appendWithSpan(txt, selectedFeatures);
                } else {
                    StringBuilder txt0 = new StringBuilder();
                    for (Feature2DGuiContainer loop : allFeaturePairs) {
                        if (loop.getRectangle().contains(x, y)) {
                            // TODO - why is this code duplicated in this file?
                            txt0.append("<br><br><span style='font-family: arial; font-size: 12pt;'>");
                            txt0.append(loop.getFeature2D().tooltipText());
                            txt0.append("</span>");
                        }
                    }
                    txt.insert(0, txt0);
                }
            } else {
                int numLayers = superAdapter.getAllLayers().size();
                int globalPriority = numLayers;
                for (Feature2DGuiContainer loop : allFeaturePairs) {
                    if (loop.getRectangle().contains(x, y)) {
                        // TODO - why is this code duplicated in this file?
                        txt.append("<br><br><span style='font-family: arial; font-size: 12pt;'>");
                        txt.append(loop.getFeature2D().tooltipText());
                        txt.append("</span>");
                        int layerNum = superAdapter.getAllLayers().indexOf(loop.getAnnotationLayerHandler());
                        int loopPriority = numLayers - layerNum;
                        double distance = currMouse.distance(loop.getRectangle().getX(), loop.getRectangle().getY());
                        if (distance < minDistance && loopPriority <= globalPriority) {
                            minDistance = distance;
                            currentFeature = loop;
                            globalPriority = loopPriority;
                        }
                        //mouseIsOverFeature = true;
                    }
                }

            }

            txt.append("<br>");
            txt.append("</html>");
            return txt.toString();
        }

        return null;
    }

    private void appendWithSpan(StringBuilder txt, List<Feature2D> selectedFeatures) {
        int numFeatures = selectedFeatures.size();
        for (int i = 0; i < Math.min(numFeatures, 3); i++) {
            appendSectionWithSpan(txt, selectedFeatures.get(i).tooltipText());
        }
        if (numFeatures == 3) {
            appendSectionWithSpan(txt, selectedFeatures.get(2).tooltipText());
        } else if (numFeatures > 3) {
            appendSectionWithSpan(txt, "...");
            appendSectionWithSpan(txt, selectedFeatures.get(numFeatures - 1).tooltipText());
        }
    }

    private void appendSectionWithSpan(StringBuilder txt, String content) {
        txt.append("<br><br><span style='font-family: arial; font-size: 12pt;'>");
        txt.append(content);
        txt.append("</span>");
    }

    private Chromosome getChromFromBoundaries(long[] chromosomeBoundaries, long genomeStart) {
        Chromosome chrom = null;
        for (int i = 0; i < chromosomeBoundaries.length; i++) {
            if (chromosomeBoundaries[i] > genomeStart) {
                chrom = hic.getChromosomeHandler().getChromosomeFromIndex(i + 1);
                break;
            }
        }
        return chrom;
    }

    private void addJumpToDiagonalMenuItems(JidePopupMenu menu, int xMousePos, int yMousePos) {

        final double preJumpBinOriginX = hic.getXContext().getBinOrigin();
        final double preJumpBinOriginY = hic.getYContext().getBinOrigin();

        // xMousePos and yMousePos coordinates are relative to the heatmap panel and not the screen
        final int clickedBinX = (int) (preJumpBinOriginX + xMousePos / hic.getScaleFactor());
        final int clickedBinY = (int) (preJumpBinOriginY + yMousePos / hic.getScaleFactor());

        // these coordinates are relative to the screen and not the heatmap panel
        final int defaultPointerDestinationX = (int) (parent.getLocationOnScreen().getX() + xMousePos);
        final int defaultPointerDestinationY = (int) (parent.getLocationOnScreen().getY() + yMousePos);

        // get maximum number of bins on the X and Y axes
        Matrix matrix = hic.getMatrix();
        MatrixZoomData matrixZoomData = matrix.getZoomData(hic.getZoom());
        final long binCountX = matrixZoomData.getXGridAxis().getBinCount();
        final long binCountY = matrixZoomData.getYGridAxis().getBinCount();

        if (clickedBinX > clickedBinY) {

            final JMenuItem jumpToDiagonalLeft = new JMenuItem('\u25C0' + "  Jump To Diagonal");
            jumpToDiagonalLeft.setSelected(straightEdgeEnabled);
            jumpToDiagonalLeft.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    double postJumpBinOriginX = preJumpBinOriginX - (clickedBinX - clickedBinY);
                    hic.moveBy(clickedBinY - clickedBinX, 0);
                    if (postJumpBinOriginX < 0) {
                        heatmapMouseBot.mouseMove((int) (defaultPointerDestinationX + postJumpBinOriginX * hic.getScaleFactor()), defaultPointerDestinationY);
                        return;
                    }
                    heatmapMouseBot.mouseMove(defaultPointerDestinationX, defaultPointerDestinationY);
                }
            });
            menu.add(jumpToDiagonalLeft);

            final JMenuItem jumpToDiagonalDown = new JMenuItem('\u25BC' + "  Jump To Diagonal");
            jumpToDiagonalDown.setSelected(straightEdgeEnabled);
            jumpToDiagonalDown.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    double postJumpBinOriginY = preJumpBinOriginY + (clickedBinX - clickedBinY);
                    hic.moveBy(0, clickedBinX - clickedBinY);
                    if (postJumpBinOriginY + parent.getHeight() / hic.getScaleFactor() > binCountY) {
                        heatmapMouseBot.mouseMove(defaultPointerDestinationX, (int) (defaultPointerDestinationY + (postJumpBinOriginY + parent.getHeight() / hic.getScaleFactor() - binCountY)));
                        return;
                    }
                    heatmapMouseBot.mouseMove(defaultPointerDestinationX, defaultPointerDestinationY);
                }
            });
            menu.add(jumpToDiagonalDown);

        } else if (clickedBinX < clickedBinY) {

            final JMenuItem jumpToDiagonalUp = new JMenuItem('\u25B2' + "  Jump To Diagonal");
            jumpToDiagonalUp.setSelected(straightEdgeEnabled);
            jumpToDiagonalUp.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    double postJumpBinOriginY = preJumpBinOriginY - (clickedBinY - clickedBinX);
                    hic.moveBy(0, clickedBinX - clickedBinY);
                    if (postJumpBinOriginY < 0) {
                        heatmapMouseBot.mouseMove(defaultPointerDestinationX, (int) (defaultPointerDestinationY + postJumpBinOriginY * hic.getScaleFactor()));
                        return;
                    }
                    heatmapMouseBot.mouseMove(defaultPointerDestinationX, defaultPointerDestinationY);
                }
            });
            menu.add(jumpToDiagonalUp);

            final JMenuItem jumpToDiagonalRight = new JMenuItem('\u25B6' + "  Jump To Diagonal");
            jumpToDiagonalRight.setSelected(straightEdgeEnabled);
            jumpToDiagonalRight.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    double postJumpBinOriginX = preJumpBinOriginX + (clickedBinY - clickedBinX);
                    hic.moveBy(clickedBinY - clickedBinX, 0);
                    if (postJumpBinOriginX + parent.getWidth() / hic.getScaleFactor() > binCountX) {
                        heatmapMouseBot.mouseMove((int) (defaultPointerDestinationX + (postJumpBinOriginX + parent.getWidth() / hic.getScaleFactor() - binCountX)), defaultPointerDestinationY);
                        return;
                    }
                    heatmapMouseBot.mouseMove(defaultPointerDestinationX, defaultPointerDestinationY);
                }
            });
            menu.add(jumpToDiagonalRight);
        }
    }

    private JidePopupMenu getAssemblyPopupMenu(final int xMousePos, final int yMousePos, JidePopupMenu menu) {

        if (HiCGlobals.phasing) {
            final JMenuItem phaseMergeItems = new JMenuItem("Merge phased blocks");
            phaseMergeItems.setEnabled(selectedSuperscaffolds.size() > 1);
            phaseMergeItems.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    AssemblyOperationExecutor.phaseMerge(superAdapter, selectedSuperscaffolds);
                    // Cleanup
                    parent.removeSelection();
                }
            });
            menu.add(phaseMergeItems);
        } else {
            final JMenuItem miMoveToTop = new JMenuItem("Move to top");
            miMoveToTop.setEnabled(!selectedFeatures.isEmpty());
            miMoveToTop.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    AssemblyOperationExecutor.moveSelection(superAdapter,
                            selectedFeatures,
                            null);
                    parent.removeSelection();
                }
            });
            menu.add(miMoveToTop);

            final JMenuItem miMoveToDebris = new JMenuItem("Move to debris");
            miMoveToDebris.setEnabled(!selectedFeatures.isEmpty());
            miMoveToDebris.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    parent.moveSelectionToEnd();
                }
            });
            menu.add(miMoveToDebris);

            final JMenuItem miMoveToDebrisAndDisperse = new JMenuItem("Move to debris and add boundaries");
            miMoveToDebrisAndDisperse.setEnabled(selectedFeatures != null && !selectedFeatures.isEmpty());
            miMoveToDebrisAndDisperse.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    moveSelectionToEndAndDisperse();
                }
            });
            menu.add(miMoveToDebrisAndDisperse);

            final JMenuItem groupItems = new JMenuItem("Remove chr boundaries");
            groupItems.setEnabled(selectedFeatures.size() > 1);
            groupItems.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    AssemblyOperationExecutor.multiMerge(superAdapter, selectedFeatures);

                    // Cleanup
                    parent.removeSelection();
                }
            });
            menu.add(groupItems);


            final JMenuItem splitItems = new JMenuItem("Add chr boundaries");
            splitItems.setEnabled(!selectedFeatures.isEmpty());
            splitItems.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    AssemblyOperationExecutor.multiSplit(superAdapter, selectedFeatures);

                    // Cleanup
                    parent.removeSelection();
                }
            });
            menu.add(splitItems);
        }

        final JMenuItem miUndo = new JMenuItem("Undo");
        miUndo.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                superAdapter.getAssemblyStateTracker().undo();
                parent.removeSelection();
                superAdapter.refresh();
            }
        });
        miUndo.setEnabled(superAdapter.getAssemblyStateTracker().checkUndo());
        menu.add(miUndo);

        final JMenuItem miRedo = new JMenuItem("Redo");
        miRedo.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                superAdapter.getAssemblyStateTracker().redo();
                parent.removeSelection();
                superAdapter.refresh();
            }
        });
        miRedo.setEnabled(superAdapter.getAssemblyStateTracker().checkRedo());
        menu.add(miRedo);

        return menu;
    }

    void moveSelectionToEndAndDisperse() {
        AssemblyScaffoldHandler assemblyHandler = superAdapter.getAssemblyStateTracker().getAssemblyHandler();
        final List<Integer> lastLine = assemblyHandler.getListOfSuperscaffolds().get(assemblyHandler.getListOfSuperscaffolds().size() - 1);
        int lastId = Math.abs(lastLine.get(lastLine.size() - 1)) - 1;
        AssemblyOperationExecutor.moveAndDisperseSelection(superAdapter, selectedFeatures, assemblyHandler.getListOfScaffolds().get(lastId).getCurrentFeature2D());
        parent.removeSelection();
    }

    public Feature2DGuiContainer getCurrentUpstreamFeature() {
        return this.currentUpstreamFeature;
    }

    public Feature2DGuiContainer getCurrentDownstreamFeature() {
        return this.currentDownstreamFeature;
    }

    public enum PromptedAssemblyAction {REGROUP, PASTE, INVERT, CUT, ADJUST, NONE, PASTETOP, PASTEBOTTOM}

    private enum DragMode {ZOOM, ANNOTATE, RESIZE, PAN, SELECT, NONE}

    private enum AdjustAnnotation {LEFT, RIGHT, NONE}


    //    @Override
    //    public String getToolTipText(MouseEvent e) {
    //        return toolTipText(e.getX(), e.getY());
    //
    //    }
}
