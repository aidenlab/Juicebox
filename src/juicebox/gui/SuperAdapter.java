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

package juicebox.gui;

import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.MainWindow;
import juicebox.data.Dataset;
import juicebox.data.DatasetReader;
import juicebox.data.DatasetReaderFactory;
import juicebox.data.HiCFileLoader;
import juicebox.mapcolorui.HeatmapPanel;
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
import juicebox.windowui.*;
import org.apache.log4j.Logger;
import org.broad.igv.feature.Chromosome;

import javax.swing.*;
import javax.swing.border.LineBorder;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.geom.Point2D;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

/**
 * Created by muhammadsaadshamim on 8/4/15.
 */
public class SuperAdapter {
    private static final Logger log = Logger.getLogger(SuperAdapter.class);
    public static String currentlyLoadedMainFiles = "";
    public static String currentlyLoadedControlFiles = "";
    private static String datasetTitle = "";
    private static String controlTitle;
    private MainWindow mainWindow;
    private HiC hic;
    private MainMenuBar mainMenuBar;
    private MainViewPanel mainViewPanel;
    private HiCZoom initialZoom;

    public HiCZoom getInitialZoom() {
        return initialZoom;
    }

    public void setAdapters(MainWindow mainWindow, HiC hic, MainMenuBar mainMenuBar, MainViewPanel mainViewPanel) {
        this.mainWindow = mainWindow;
        this.hic = hic;
        this.mainMenuBar = mainMenuBar;
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
            try {
                new QCDialog(mainWindow, hic, mainWindow.getTitle() + " info");
            } catch (Exception e) {
                // TODO - test on hic file with no stats file specified
                e.printStackTrace();
                JOptionPane.showMessageDialog(mainWindow, "Unable to launch QC Statistics", "Error",
                        JOptionPane.ERROR_MESSAGE);
            }
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
        mainViewPanel.setEnableForAllElements(this, status);
        mainMenuBar.setEnableForAllElements(status);
    }

    public void resetControlMap() {
        hic.setControlDataset(null);
        MatrixType[] options = new MatrixType[]{MatrixType.OBSERVED, MatrixType.OE, MatrixType.PEARSON, MatrixType.EXPECTED};
        mainViewPanel.setSelectedDisplayOption(options, false);
        currentlyLoadedControlFiles = "";
        updateTitle();
    }

    public void launchSlideShow() {
        new Slideshow(mainWindow, this);
    }

//    public Slideshow getSlideshow() { return new Slideshow(mainWindow,this); }

    public void launchImportState(File fileForExport) {
        new ImportFileDialog(fileForExport, mainWindow);
    }

    public void launchLoadStateFromXML(String mapPath) {
        LoadStateFromXMLFile.reloadSelectedState(this, mapPath);
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
        new UnsavedAnnotationWarning(this);
        mainMenuBar.setShow2DAnnotations(true);
        HiCFileLoader.loadFromListActionPerformed(this, control);
    }

    public void loadFromRecentActionPerformed(String url, String title, boolean control) {
        new UnsavedAnnotationWarning(this);
        mainMenuBar.setShow2DAnnotations(true);
        HiCFileLoader.loadFromRecentActionPerformed(this, url, title, control);
    }

    public void launchExportImage() {
        new SaveImageDialog(null, hic, mainViewPanel.getHiCPanel());
    }

    public void exportAnnotations() {
        new SaveAnnotationsDialog(MainMenuBar.customAnnotations, getMapName());
    }

    public void exitActionPerformed() {
        new UnsavedAnnotationWarning(this);
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
        return new CustomAnnotation(Feature2DParser.loadFeatures(temp.getAbsolutePath(),
                hic.getChromosomes(), true, null, false), s);
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

    public String getDescription(String item) {
        return JOptionPane.showInputDialog(mainWindow, "Enter description for saved " + item + ":",
                hic.getDefaultLocationDescription());
    }

    public void addNewStateToXML(String stateDescription) {
        XMLFileHandling.addNewStateToXML(stateDescription, this);
    }

    public void setNormalizationDisplayState() {
        mainViewPanel.setNormalizationDisplayState(hic);
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
        // TODO - why does this sometimes return null?
        try {
            return new Point2D.Double((double) hic.getZd().getXGridAxis().getBinCount() / width,
                    (double) hic.getZd().getYGridAxis().getBinCount() / height);
        } catch (Exception e) {
            return null; // TODO is there a good default to return?
        }
    }

    public Point getHeatMapPanelDimensions() {
        return new Point(mainViewPanel.getHeatmapPanel().getWidth(), mainViewPanel.getHeatmapPanel().getHeight());
    }

    public void initializeMainView(Container contentPane, Dimension bigPanelDim, Dimension panelDim) {
        mainViewPanel.initializeMainView(this, contentPane, bigPanelDim, panelDim);
    }

    private void unsafeSetInitialZoom() {

        //For now, in case of Pearson - set initial to 500KB resolution.
        if ((hic.getDisplayOption() == MatrixType.PEARSON)) {
            initialZoom = hic.getMatrix().getFirstPearsonZoomData(HiC.Unit.BP).getZoom();
        } else if (hic.getXContext().getChromosome().getName().equals("All")) {
            mainViewPanel.getResolutionSlider().setEnabled(false);
            initialZoom = hic.getMatrix().getFirstZoomData(HiC.Unit.BP).getZoom();
        } else {
            mainViewPanel.getResolutionSlider().setEnabled(true);

            HiC.Unit currentUnit = hic.getZoom().getUnit();

            List<HiCZoom> zooms = (currentUnit == HiC.Unit.BP ? hic.getDataset().getBpZooms() :
                    hic.getDataset().getFragZooms());


//            Find right zoom level

            int pixels = mainViewPanel.getHeatmapPanel().getMinimumDimension();
            int len;
            if (currentUnit == HiC.Unit.BP) {
                len = (Math.max(hic.getXContext().getChrLength(), hic.getYContext().getChrLength()));
            } else {
                len = Math.max(hic.getDataset().getFragmentCounts().get(hic.getXContext().getChromosome().getName()),
                        hic.getDataset().getFragmentCounts().get(hic.getYContext().getChromosome().getName()));
            }

            int maxNBins = pixels / HiCGlobals.BIN_PIXEL_WIDTH;
            int bp_bin = len / maxNBins;
            initialZoom = zooms.get(zooms.size() - 1);
            for (int z = 1; z < zooms.size(); z++) {
                if (zooms.get(z).getBinSize() < bp_bin) {
                    initialZoom = zooms.get(z - 1);
                    break;
                }
            }

        }
        hic.unsafeActuallySetZoomAndLocation("", "", initialZoom, 0, 0, -1, true, HiC.ZoomCallType.INITIAL, true);
    }

    public void refresh() {
        mainViewPanel.getHeatmapPanel().clearTileCache();
        mainWindow.repaint();
        mainViewPanel.updateThumbnail(hic);
        //System.err.println(heatmapPanel.getSize());
    }

    private void refreshMainOnly() {
        mainViewPanel.getHeatmapPanel().clearTileCache();
        mainWindow.repaint();
    }

    private void unsafeLoad(final List<String> files, final boolean control, boolean restore) throws IOException {

        String newFilesToBeLoaded = "";
        boolean allFilesAreHiC = true;
        for (String file : files) {
            if (newFilesToBeLoaded.length() > 1) {
                newFilesToBeLoaded += "##";
            }
            newFilesToBeLoaded += file;
            allFilesAreHiC &= file.endsWith(".hic");
        }

        if ((!control) && newFilesToBeLoaded.equals(currentlyLoadedMainFiles)) {
            if (!restore) {
                JOptionPane.showMessageDialog(mainWindow, "File(s) already loaded");
            }
            return;
        } else if (control && newFilesToBeLoaded.equals(currentlyLoadedControlFiles)) {
            if (!restore) {
                JOptionPane.showMessageDialog(mainWindow, "File(s) already loaded");
            }
            return;
        }

        if (allFilesAreHiC) {
            mainViewPanel.setIgnoreUpdateThumbnail(true);
            //heatmapPanel.setBorder(LineBorder.createBlackLineBorder());
            //thumbnailPanel.setBorder(LineBorder.createBlackLineBorder());
            mainViewPanel.getMouseHoverTextPanel().setBorder(LineBorder.createGrayLineBorder());

            DatasetReader reader = DatasetReaderFactory.getReader(files);
            if (reader == null) return;
            Dataset dataset = reader.read();
            if (reader.getVersion() < HiCGlobals.minVersion) {
                JOptionPane.showMessageDialog(mainWindow, "This version of \"hic\" format is no longer supported");
                return;
            }

            MatrixType[] options;
            if (control) {
                hic.setControlDataset(dataset);
                options = HiCGlobals.enabledMatrixTypesWithControl;
            } else {
                hic.reset();
                hic.setDataset(dataset);
                hic.setChromosomes(dataset.getChromosomes());
                mainViewPanel.setChromosomes(hic.getChromosomes());

                String[] normalizationOptions;
                if (dataset.getVersion() < HiCGlobals.minVersion) {
                    normalizationOptions = new String[]{NormalizationType.NONE.getLabel()};
                } else {
                    ArrayList<String> tmp = new ArrayList<String>();
                    tmp.add(NormalizationType.NONE.getLabel());
                    for (NormalizationType t : hic.getDataset().getNormalizationTypes()) {
                        tmp.add(t.getLabel());
                    }

                    normalizationOptions = tmp.toArray(new String[tmp.size()]);
                }

                mainViewPanel.setEnabledForNormalization(normalizationOptions,
                        hic.getDataset().getVersion() >= HiCGlobals.minVersion);

                if (hic.isControlLoaded()) {
                    options = HiCGlobals.enabledMatrixTypesWithControl;
                } else {
                    options = HiCGlobals.enabledMatrixTypesNoControl;
                }

                hic.resetContexts();
                updateTrackPanel();
                mainMenuBar.getRecentLocationMenu().setEnabled(true);
                mainWindow.getContentPane().invalidate();
                mainWindow.repaint();
                mainViewPanel.resetResolutionSlider();
                mainViewPanel.unsafeRefreshChromosomes(SuperAdapter.this);

            }
            mainViewPanel.setSelectedDisplayOption(options, control);
            setEnableForAllElements(true);

            if (control) {
                currentlyLoadedControlFiles = newFilesToBeLoaded;
            } else {
                currentlyLoadedMainFiles = newFilesToBeLoaded;
            }

            mainMenuBar.setContolMapLoadableEnabled(true);
            mainViewPanel.setIgnoreUpdateThumbnail(false);
        } else {
            JOptionPane.showMessageDialog(mainWindow, "Please choose a .hic file to load");
        }
    }

    public void safeLoad(final List<String> files, final boolean control, final String title) {
        addRecentMapMenuEntry(title.trim() + "@@" + files.get(0), true);
        Runnable runnable = new Runnable() {
            public void run() {
                boolean isRestorenMode = false;
                unsafeLoadWithTitleFix(files, control, title, isRestorenMode);
            }
        };
        mainWindow.executeLongRunningTask(runnable, "MainWindow safe load");
    }

    public void unsafeLoadWithTitleFix(List<String> files, boolean control, String title, boolean restore) {
        String resetTitle = datasetTitle;
        if (control) resetTitle = controlTitle;

        ActionListener l = mainViewPanel.getDisplayOptionComboBox().getActionListeners()[0];
        try {
            mainViewPanel.getDisplayOptionComboBox().removeActionListener(l);
            unsafeLoad(files, control, restore);
            //mainViewPanel.updateThumbnail(hic);
            refresh();
            updateTitle(control, title);

        } catch (IOException e) {
            // TODO somehow still have trouble reloading the previous file
            log.error("Error loading hic file", e);
            JOptionPane.showMessageDialog(mainWindow, "Error loading .hic file", "Error", JOptionPane.ERROR_MESSAGE);
            if (!control) hic.reset();
            mainViewPanel.updateThumbnail(hic);
            updateTitle(control, resetTitle);
        }
        finally {
            mainViewPanel.getDisplayOptionComboBox().addActionListener(l);
        }
    }

    public KeyEventDispatcher getNewHiCKeyDispatcher() {
        return new HiCKeyDispatcher(hic, mainViewPanel.getDisplayOptionComboBox());
    }

    public LoadDialog launchLoadFileDialog(Properties properties) {
        return new LoadDialog(mainWindow, properties, this);
    }

    void safeRefreshButtonActionPerformed() {
        Runnable runnable = new Runnable() {
            @Override
            public void run() {
                mainViewPanel.unsafeRefreshChromosomes(SuperAdapter.this);
            }
        };
        mainWindow.executeLongRunningTask(runnable, "Refresh Button");
    }

    public boolean safeDisplayOptionComboBoxActionPerformed() {
        final boolean[] retVal = new boolean[1];
        Runnable runnable = new Runnable() {
            public void run() {
                retVal[0] = unsafeDisplayOptionComboBoxActionPerformed();
            }
        };
        mainWindow.executeLongRunningTask(runnable, "DisplayOptionsComboBox");
        return retVal[0];
    }


    void safeNormalizationComboBoxActionPerformed(final ActionEvent e) {
        Runnable runnable = new Runnable() {
            public void run() {
                unsafeNormalizationComboBoxActionPerformed(e);
            }
        };
        mainWindow.executeLongRunningTask(runnable, "Normalization ComboBox");
    }

    private boolean unsafeDisplayOptionComboBoxActionPerformed() {

        MatrixType option = (MatrixType) (mainViewPanel.getDisplayOptionComboBox().getSelectedItem());
        if (hic.isWholeGenome() && !MatrixType.isValidGenomeWideOption(option)) {
            JOptionPane.showMessageDialog(mainWindow, option + " matrix is not available for whole-genome view.");
            mainViewPanel.getDisplayOptionComboBox().setSelectedItem(hic.getDisplayOption());
            return false;
        }

        mainViewPanel.getColorRangePanel().handleNewFileLoading(option, MainViewPanel.preDefMapColor);

        if (option == MatrixType.VS) {
            if (!hic.getMatrix().isIntra()) {
                JOptionPane.showMessageDialog(mainWindow, "Observed VS Control is not available for inter-chr views.");
                mainViewPanel.getDisplayOptionComboBox().setSelectedItem(hic.getDisplayOption());
                return false;
            }
        }

        if (option == MatrixType.PEARSON) {
            if (!hic.getMatrix().isIntra()) {
                JOptionPane.showMessageDialog(mainWindow, "Pearson's matrix is not available for inter-chr views.");
                mainViewPanel.getDisplayOptionComboBox().setSelectedItem(hic.getDisplayOption());
                return false;

            } else {
                try {
                    if (hic.getZd().getPearsons(hic.getDataset().getExpectedValues(hic.getZd().getZoom(), hic.getNormalizationType())) == null) {
                        JOptionPane.showMessageDialog(mainWindow, "Pearson's matrix is not available at this resolution");
                        mainViewPanel.getDisplayOptionComboBox().setSelectedItem(hic.getDisplayOption());
                        return false;
                    }
                } catch (Exception ex) {
                    JOptionPane.showMessageDialog(mainWindow, "Pearson's matrix is not available at this region");
                    mainViewPanel.getDisplayOptionComboBox().setSelectedItem(hic.getDisplayOption());
                    return false;
                }
            }
        }

        hic.setDisplayOption(option);
        refresh(); // necessary to invalidate minimap when changing view
        return true;
    }

    /**
     * TODO deprecate
     *
     * @return hic
     */
    public HiC getHiC() {
        return hic;
    }

    /**
     * TODO deprecate
     *
     * @return mainwindow
     */
    public MainWindow getMainWindow() {
        return mainWindow;
    }

    public void revalidate() {
        mainWindow.revalidate();
    }

    public void updateToolTipText(String text) {
        mainViewPanel.updateToolTipText(text);
    }

    public void setPositionChrTop(String positionChrTop) {
        mainViewPanel.setPositionChrTop(positionChrTop);
    }

    public void setPositionChrLeft(String positionChrLeft) {
        mainViewPanel.setPositionChrLeft(positionChrLeft);
    }

    public String getToolTip() {
        return mainViewPanel.getToolTip();
    }

    public void repaintTrackPanels() {
        mainViewPanel.repaintTrackPanels();
    }

    public boolean isResolutionLocked() {
        return mainViewPanel.isResolutionLocked();
    }


    public void updateThumbnail() {
        mainViewPanel.updateThumbnail(hic);
    }

    public void updateZoom(HiCZoom newZoom) {
        mainViewPanel.updateZoom(newZoom);
    }

    public void updateAndResetZoom(HiCZoom newZoom) {
        mainViewPanel.updateAndResetZoom(newZoom);
    }

    public void launchFileLoadingError(String urlString) {
        JOptionPane.showMessageDialog(mainWindow, "Error while trying to load " + urlString, "Error",
                JOptionPane.ERROR_MESSAGE);
    }

    private void updateTitle(boolean control, String title) {
        if (control) controlTitle = title;
        else datasetTitle = title;
        updateTitle();
    }

    private void updateTitle() {
        String newTitle = datasetTitle;
        if (controlTitle != null) newTitle += "  (control=" + controlTitle + ")";
        mainWindow.setTitle(HiCGlobals.juiceboxTitle + newTitle);
    }

    public String getMapName() {
        return datasetTitle.split(" ")[0];
    }

    public void launchGenericMessageDialog(String message, String error, int errorMessage) {
        JOptionPane.showMessageDialog(mainWindow, message, error, errorMessage);
    }

    public HeatmapPanel getHeatmapPanel() {
        return mainViewPanel.getHeatmapPanel();
    }


    public void updateTrackPanel() {
        mainViewPanel.updateTrackPanel(hic.getLoadedTracks().size() > 0);
    }

    private void unsafeNormalizationComboBoxActionPerformed(ActionEvent e) {
        String value = (String) mainViewPanel.getNormalizationComboBox().getSelectedItem();
        NormalizationType chosen = null;
        for (NormalizationType type : NormalizationType.values()) {
            if (type.getLabel().equals(value)) {
                chosen = type;
                break;
            }
        }
        final NormalizationType passChosen = chosen;
        hic.setNormalizationType(passChosen);
        refreshMainOnly();
    }

    public MainViewPanel getMainViewPanel() {
        return mainViewPanel;
    }

    public boolean isTooltipAllowedToUpdated() {
        return mainViewPanel.isTooltipAllowedToUpdated();
    }

    public void toggleToolTipUpdates(boolean b) {
        mainViewPanel.toggleToolTipUpdates(b);
    }

    public void executeLongRunningTask(Runnable runnable, String s) {
        mainWindow.executeLongRunningTask(runnable, s);
    }

    public void updateRatioColorSlider(int max, double val) {
        mainViewPanel.updateRatioColorSlider(hic, max, val);
    }

    public void updateColorSlider(int min, double low, double high, double max) {
        mainViewPanel.updateColorSlider(hic, min, low, high, max);
    }

    public void unsafeSetSelectedChromosomes(Chromosome xC, Chromosome yC) {
        mainViewPanel.unsafeSetSelectedChromosomes(this, xC, yC);
    }

    public void setSelectedChromosomesNoRefresh(Chromosome chrX, Chromosome chrY) {
        mainViewPanel.setSelectedChromosomesNoRefresh(chrX, chrY, hic.getXContext(), hic.getYContext());
        initialZoom = null;
    }

    public void unsafeUpdateHiCChromosomes(Chromosome chrX, Chromosome chrY) {
        hic.setSelectedChromosomes(chrX, chrY);
        mainViewPanel.getRulerPanelX().setContext(hic.getXContext(), HiCRulerPanel.Orientation.HORIZONTAL);
        mainViewPanel.getRulerPanelY().setContext(hic.getYContext(), HiCRulerPanel.Orientation.VERTICAL);

        mainViewPanel.getChromosomeFigPanelX().setContext(hic.getXContext(), HiCChromosomeFigPanel.Orientation.HORIZONTAL);
        mainViewPanel.getChromosomeFigPanelY().setContext(hic.getYContext(), HiCChromosomeFigPanel.Orientation.VERTICAL);

        unsafeSetInitialZoom();
    }

    public void deleteUnsavedEdits() {
        mainMenuBar.deleteUnsavedEdits();
    }

    public void clearAllAnnotations() {
        mainMenuBar.clearAllAnnotations();
    }

    public void setSparseFeaturePlotting(boolean status) {
        hic.setSparseFeaturePlotting(status);
    }

    public void enlarge2DFeaturePlotting(boolean status) {
        hic.enlarge2DFeaturePlotting(status);
    }

    public void toggleFeatureOpacity(boolean status) {
        hic.toggleFeatureOpacity(status);
    }


    public void toggleAxisLayOut(boolean status) {
        mainViewPanel.switchToOnlyEndPtsLayOut(status);
    }

    public void showChromosomeFig(boolean status) {
        mainViewPanel.showChromosomeFig(status);
    }
}
