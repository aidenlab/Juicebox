/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2015 Broad Institute, Aiden Lab
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

package juicebox.gui;

import juicebox.HiC;
import juicebox.MainWindow;
import juicebox.state.ImportFileDialog;
import juicebox.state.LoadStateFromXMLFile;
import juicebox.state.Slideshow;
import juicebox.state.XMLFileHandling;
import juicebox.track.LoadAction;
import juicebox.track.LoadEncodeAction;
import juicebox.track.feature.CustomAnnotation;
import juicebox.track.feature.CustomAnnotationHandler;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.Feature2DParser;
import juicebox.windowui.DumpDialog;
import juicebox.windowui.QCDialog;
import juicebox.windowui.SaveAnnotationsDialog;
import juicebox.windowui.SaveImageDialog;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.Point2D;
import java.io.File;
import java.util.List;

/**
 * Created by muhammadsaadshamim on 8/4/15.
 */
public class SuperAdapter {
    private MainWindow mainWindow;
    private HiC hic;
    private MainMenuBar mainMenuBar;
    private ToolBarPanel toolBarPanel;
    private MainViewPanel mainViewPanel;

    public SuperAdapter(MainWindow mainWindow, HiC hic, MainMenuBar mainMenuBar, ToolBarPanel toolBarPanel,
                        MainViewPanel mainViewPanel) {
        this.mainWindow = mainWindow;
        this.hic = hic;
        this.mainMenuBar = mainMenuBar;
        this.toolBarPanel = toolBarPanel;
        this.mainViewPanel = mainViewPanel;
    }

    public boolean unsavedEditsExist() {
        return mainMenuBar.unsavedEditsExist();
    }

    public void addRecentMapMenuEntry(String title, boolean status) {
        mainMenuBar.addRecentMapMenuEntry(title, status);
    }

    public void addRecentStateMenuEntry(String title, boolean status) {
        mainMenuBar.addRecentStateMenuEntry(title, status);
    }

    public void initializeCustomAnnotations() {
        mainMenuBar.initializeCustomAnnotations();
    }

    public JMenuBar createMenuBar() {
        return mainMenuBar.createMenuBar(this);
    }

    public void showDataSetMetrics() {
        if (hic.getDataset() == null) {
            JOptionPane.showMessageDialog(mainWindow, "File must be loaded to show info", "Error", JOptionPane.ERROR_MESSAGE);
        } else {
            new QCDialog(mainWindow, hic, mainWindow.getTitle() + " info");
        }
    }

    public void exportDataLauncher() {
        if (hic.getDataset() == null) {
            JOptionPane.showMessageDialog(mainWindow, "File must be loaded to show info", "Error", JOptionPane.ERROR_MESSAGE);
        } else {
            new DumpDialog(mainWindow, hic);
        }
    }

    public void setEnableForAllElements(boolean status) {
        mainMenuBar.setEnableForAllElements(status);
    }

    public void launchSlideShow() {
        Slideshow.viewShow(mainWindow, hic);
    }

    public void launchImportState(File fileForExport) {
        new ImportFileDialog(fileForExport, MainWindow.getInstance());
    }

    public void launchLoadStateFromXML(String mapPath) {
        LoadStateFromXMLFile.reloadSelectedState(mapPath, mainWindow, hic);
    }

    public void restoreLocation(String loc) {
        hic.restoreLocation(loc);
    }

    public LoadEncodeAction getEncodeAction() {
        return mainMenuBar.getEncodeAction();
    }

    public LoadAction getTrackLoadAction() {
        return mainMenuBar.getTrackLoadAction();
    }

    public void updatePrevStateNameFromImport(String path) {
        mainMenuBar.updatePrevStateNameFromImport(path);
    }

    public void loadFromListActionPerformed(boolean control) {
        mainWindow.loadFromListActionPerformed(control);
    }

    public void loadFromRecentActionPerformed(String url, String title, boolean control) {
        mainWindow.loadFromRecentActionPerformed(url, title, control);
    }

    public void launchExportImage() {
        new SaveImageDialog(null, hic, mainWindow.getHiCPanel());
    }

    public void exitActionPerformed() {
        mainWindow.exitActionPerformed();
    }

    public LoadAction createNewTrackLoadAction() {
        return new LoadAction("Load Basic Annotations...", mainWindow, hic);
    }

    public LoadEncodeAction createNewLoadEncodeAction() {
        return new LoadEncodeAction("Load ENCODE Tracks...", mainWindow, hic);
    }

    public void exportOverlapMIAction(CustomAnnotation customAnnotations) {
        List<Feature2DList> loops = hic.getAllVisibleLoopLists();
        if (loops.size() != 1)
            JOptionPane.showMessageDialog(mainWindow, "Please merge ONE loaded set of annotations at a time.", "Error", JOptionPane.ERROR_MESSAGE);
        else
            new SaveAnnotationsDialog(customAnnotations, loops.get(0));
    }

    public CustomAnnotation generateNewCustomAnnotation(File temp, String s) {
        return new CustomAnnotation(Feature2DParser.parseLoopFile(temp.getAbsolutePath(),
                hic.getChromosomes(), false, 0, 0, 0, true, null), s);
    }

    public int clearCustomAnnotationDialog() {
        return JOptionPane.showConfirmDialog(
                mainWindow,
                "Are you sure you want to clear all custom annotations?",
                "Confirm",
                JOptionPane.YES_NO_OPTION);
    }

    public void repaint() {
        mainWindow.repaint();
    }

    public void loadFromURLActionPerformed() {
        if (hic.getDataset() == null) {
            JOptionPane.showMessageDialog(mainWindow, "HiC file must be loaded to load tracks", "Error", JOptionPane.ERROR_MESSAGE);
            return;
        }

        String url = JOptionPane.showInputDialog("Enter URL: ");
        if (url != null) {
            hic.loadTrack(url);

        }
    }

    public String getLocationDescription() {
        return hic.getLocationDescription();
    }

    public String getStateDescription() {
        return JOptionPane.showInputDialog(mainWindow, "Enter description for saved location:",
                hic.getDefaultLocationDescription());
    }

    public void addNewStateToXML(String stateDescription) {
        XMLFileHandling.addNewStateToXML(stateDescription, hic, mainWindow);
    }

    public void setNormalizationDisplayState() {
        mainWindow.setNormalizationDisplayState();
    }

    public void setShowLoops(boolean showLoops) {
        hic.setShowLoops(showLoops);
    }

    public CustomAnnotation addVisibleLoops(CustomAnnotationHandler handler, CustomAnnotation customAnnotations) {
        return handler.addVisibleLoops(hic, customAnnotations);
    }

    public void centerMap(int xBP, int yBP) {
        hic.center(xBP, yBP);
    }

    public void moveMapBy(int dxBP, int dyBP) {
        hic.moveBy(dxBP, dyBP);
    }

    public boolean shouldVisibleWindowBeRendered() {
        return hic != null && hic.getXContext() != null;
    }

    public double getHiCScaleFactor() {
        return hic.getScaleFactor();
    }

    public Point2D.Double getHiCOrigin() {
        return new Point2D.Double(hic.getXContext().getBinOrigin(), hic.getYContext().getBinOrigin());
    }

    public Point2D.Double getHiCScale(int width, int height) {
        return new Point2D.Double((double) hic.getZd().getXGridAxis().getBinCount() / width,
                (double) hic.getZd().getYGridAxis().getBinCount() / height);
    }

    public Point getHeatMapPanelDimensions() {
        return new Point(mainWindow.getHeatmapPanel().getWidth(), mainWindow.getHeatmapPanel().getHeight());
    }
}
