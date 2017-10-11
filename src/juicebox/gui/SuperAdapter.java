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

package juicebox.gui;

import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.MainWindow;
import juicebox.assembly.AssemblyStateTracker;
import juicebox.data.*;
import juicebox.data.anchor.MotifAnchorTools;
import juicebox.mapcolorui.HeatmapPanel;
import juicebox.mapcolorui.HiCColorScale;
import juicebox.mapcolorui.PearsonColorScaleEditor;
import juicebox.state.ImportFileDialog;
import juicebox.state.LoadStateFromXMLFile;
import juicebox.state.Slideshow;
import juicebox.state.XMLFileHandling;
import juicebox.track.LoadAction;
import juicebox.track.LoadEncodeAction;
import juicebox.track.feature.AnnotationLayer;
import juicebox.track.feature.AnnotationLayerHandler;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.Feature2DParser;
import juicebox.windowui.*;
import juicebox.windowui.layers.LayersPanel;
import juicebox.windowui.layers.UnsavedAnnotationWarning;
import org.apache.log4j.Logger;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.ui.util.FileDialogUtils;

import javax.swing.*;
import javax.swing.border.LineBorder;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.geom.Point2D;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
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
    private final List<AnnotationLayerHandler> annotationLayerHandlers = new ArrayList<>();
    private MainWindow mainWindow;
    private HiC hic;
    private MainMenuBar mainMenuBar;
    private MainViewPanel mainViewPanel;
    private HiCZoom initialZoom;
    private AnnotationLayerHandler activeLayer;
    private AssemblyStateTracker assemblyStateTracker;
    private HiCColorScale pearsonColorScale;
    private LayersPanel layersPanel;
    private boolean layerPanelIsVisible = false;

    public static String getDatasetTitle() {
        return datasetTitle;
    }

    public static void setDatasetTitle(String newDatasetTitle) {
        datasetTitle = newDatasetTitle;
    }

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

    public JMenuBar createMenuBar() {
        return mainMenuBar.createMenuBar(this);
    }

    public void showDataSetMetrics(boolean isControl) {
        if (hic.getDataset() == null) {
            JOptionPane.showMessageDialog(mainWindow, "File must be loaded to show info", "Error", JOptionPane.ERROR_MESSAGE);
        } else {
            try {
                String title = mainWindow.getTitle() + " QC Stats";
                if (isControl) title += " (control)";
                new QCDialog(mainWindow, hic, title, isControl);
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
            JOptionPane.showMessageDialog(mainWindow, "File must be loaded to show info",
                    "Error", JOptionPane.ERROR_MESSAGE);
        } else {
            new DumpDialog(mainWindow, hic);
        }
    }

//    public Slideshow getSlideshow() { return new Slideshow(mainWindow,this); }

    public void setEnableForAllElements(boolean status) {
        mainViewPanel.setEnableForAllElements(this, status);
        mainMenuBar.setEnableForAllElements(status);
        for (AnnotationLayerHandler handler : annotationLayerHandlers) {
            handler.setImportAnnotationsEnabled(status);
        }
    }

    public void resetControlMap() {
        hic.setControlDataset(null);
        MatrixType[] options = HiCGlobals.enabledMatrixTypesNoControl;
        mainViewPanel.setSelectedDisplayOption(options, false);
        currentlyLoadedControlFiles = null;
        controlTitle = null;
        updateTitle();
    }

    public void launchSlideShow() {
        new Slideshow(mainWindow, this);
    }

    public void launchImportState(File fileForExport) {
        new ImportFileDialog(fileForExport, mainWindow);
    }

    public void launchLoadStateFromXML(String mapPath) {
        LoadStateFromXMLFile.reloadSelectedState(this, mapPath);
    }

    public void launchPearsonColorScaleEditor() {
        if (pearsonColorScale != null) new PearsonColorScaleEditor(this, pearsonColorScale);
    }

    public void restoreLocation(String loc) {
        hic.restoreLocation(loc);
    }

    public LoadEncodeAction getEncodeAction() {
        if (layersPanel == null){
            layersPanel = new LayersPanel(this);
            setLayersPanelVisible(false);
        }
        return layersPanel.getEncodeAction();
    }

    public LoadAction getTrackLoadAction() {
        return layersPanel.getTrackLoadAction();
    }

    public void updatePrevStateNameFromImport(String path) {
        mainMenuBar.updatePrevStateNameFromImport(path);
    }

    public void loadFromListActionPerformed(boolean control) {
        UnsavedAnnotationWarning unsaved = new UnsavedAnnotationWarning(this);
        if (unsaved.checkAndDelete()) {
            HiCFileLoader.loadFromListActionPerformed(this, control);
        }
    }

    public void loadFromRecentActionPerformed(String url, String title, boolean control) {
        UnsavedAnnotationWarning unsaved = new UnsavedAnnotationWarning(this);
        if (unsaved.checkAndDelete()) {
            HiCFileLoader.loadFromRecentActionPerformed(this, url, title, control);
        }
    }

    public void launchExportPDF() {
        new SaveImageDialog(null, hic, mainWindow, mainViewPanel.getHiCPanel(), ".pdf");
    }

    public void launchExportSVG() {
        new SaveImageDialog(null, hic, mainWindow, mainViewPanel.getHiCPanel(), ".svg");
    }

    public void exitActionPerformed() {
        UnsavedAnnotationWarning unsaved = new UnsavedAnnotationWarning(this);
        if (unsaved.checkAndDelete()) {
            mainWindow.exitActionPerformed();
        }
    }

    public void generateNewCustomAnnotation(File temp) {
        getActiveLayerHandler().setAnnotationLayer(
                new AnnotationLayer(Feature2DParser.loadFeatures(temp.getAbsolutePath(), hic.getChromosomeHandler(), true, null, false)));
    }

    public int clearCustomAnnotationDialog() {
        return JOptionPane.showConfirmDialog(
                mainWindow,
                "Are you sure you want to clear this layer's annotations?",
                "Confirm",
                JOptionPane.YES_NO_OPTION);
    }

    public int deleteCustomAnnotationDialog(String layerName) {
        return JOptionPane.showConfirmDialog(
                mainWindow,
                "Are you sure you want to delete this layer (" + layerName + ")?",
                "Confirm",
                JOptionPane.YES_NO_OPTION);
    }

    public void repaint() {
        mainWindow.repaint();
    }

    public void safeLoadFromURLActionPerformed(final Runnable refresh1DLayers) {
        Runnable runnable = new Runnable() {
            public void run() {
                if (hic.getDataset() == null) {
                    JOptionPane.showMessageDialog(mainWindow, "HiC file must be loaded to load tracks", "Error", JOptionPane.ERROR_MESSAGE);
                    return;
                }

                String url = JOptionPane.showInputDialog("Enter URL: ");

                if (url != null && url.length() > 0) {
                    url = url.trim();
                    hic.unsafeLoadTrack(url);
                }
                refresh1DLayers.run();
            }
        };
        mainWindow.executeLongRunningTask(runnable, "Load from url");
    }

    public String getLocationDescription() {
        return hic.getLocationDescription();
    }

    /*
    public void setShowLoops(boolean showLoops) {
        hic.setShowLoops(showLoops);
    }

    public void addVisibleLoops(CustomAnnotationHandler handler) {
        handler.addVisibleLoops(hic);
    }
    */

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
        if (hic.isInPearsonsMode()) {
            initialZoom = hic.getMatrix().getFirstPearsonZoomData(HiC.Unit.BP).getZoom();
        } else if (ChromosomeHandler.isAllByAll(hic.getXContext().getChromosome())) {
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
        hic.unsafeActuallySetZoomAndLocation(hic.getXContext().getChromosome().toString(), hic.getYContext().getChromosome().toString(),
                initialZoom, 0, 0, -1, true, HiC.ZoomCallType.INITIAL, true, isResolutionLocked() ? 1 : 0, true);
    }

    public void refresh() {
        mainViewPanel.getHeatmapPanel().clearTileCache();
        mainWindow.repaint();
        mainViewPanel.updateThumbnail(hic);
        //System.err.println(heatmapPanel.getSize());
    }

    public void clearAllMatrixZoomCache() {
        //not sure if this is a right place for this
        hic.clearAllMatrixZoomDataCache();
    }

    private void refreshMainOnly() {
        mainViewPanel.getHeatmapPanel().clearTileCache();
        mainWindow.repaint();
    }

    private boolean unsafeLoad(final List<String> files, final boolean control, boolean restore) throws IOException {

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
            return false;
        }
        if (control && newFilesToBeLoaded.equals(currentlyLoadedControlFiles)) {
            if (!restore) {
                JOptionPane.showMessageDialog(mainWindow, "File(s) already loaded");
            }
            return false;
        }

        if (allFilesAreHiC) {
            mainViewPanel.setIgnoreUpdateThumbnail(true);
            //heatmapPanel.setBorder(LineBorder.createBlackLineBorder());
            //thumbnailPanel.setBorder(LineBorder.createBlackLineBorder());
            mainViewPanel.getMouseHoverTextPanel().setBorder(LineBorder.createGrayLineBorder());

            DatasetReader reader = DatasetReaderFactory.getReader(files);
            if (reader == null) return false;
            Dataset dataset = reader.read();
            if (reader.getVersion() < HiCGlobals.minVersion) {
                JOptionPane.showMessageDialog(mainWindow, "This version of \"hic\" format is no longer supported");
                return false;
            }
            if (control && !dataset.getGenomeId().equals(hic.getDataset().getGenomeId())) {
                JOptionPane.showMessageDialog(mainWindow, "Cannot load maps with different genomes");
                return false;
            }
            if (control && dataset.getVersion() != hic.getDataset().getVersion() &&
                    (dataset.getVersion() < 7 || hic.getDataset().getVersion() < 7)) {
                JOptionPane.showMessageDialog(mainWindow, "Cannot load control with .hic files less than version 7");
                return false;
            }

            if (!control && hic.getDataset() != null && !dataset.getGenomeId().equals(hic.getDataset().getGenomeId())) {
                resetControlMap();
            }

            MatrixType[] options;
            if (control) {
                hic.setControlDataset(dataset);
                options = HiCGlobals.enabledMatrixTypesWithControl;
            } else {
                hic.reset();
                hic.setDataset(dataset);
                hic.setChromosomeHandler(dataset.getChromosomeHandler());
                mainViewPanel.setChromosomes(hic.getChromosomeHandler());

                String[] normalizationOptions;
                if (dataset.getVersion() < HiCGlobals.minVersion) {
                    normalizationOptions = new String[]{NormalizationType.NONE.getLabel()};
                } else {
                    ArrayList<String> tmp = new ArrayList<>();
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

            mainMenuBar.updateMainMapHasBeenLoaded(true);
            if (control) {
                mainMenuBar.updateControlMapHasBeenLoaded(true);
            }
            mainViewPanel.setIgnoreUpdateThumbnail(false);
        } else {
            JOptionPane.showMessageDialog(mainWindow, "Please choose a .hic file to load");
        }
        return true;
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
            if (unsafeLoad(files, control, restore)) {
                //mainViewPanel.updateThumbnail(hic);
                refresh();
                updateTitle(control, title);
            }
        } catch (IOException e) {
            // TODO somehow still have trouble reloading the previous file
            log.error("Error loading hic file", e);
            JOptionPane.showMessageDialog(mainWindow, "Error loading .hic file", "Error", JOptionPane.ERROR_MESSAGE);
            mainViewPanel.updateThumbnail(hic);
            updateTitle(control, resetTitle);
        }
        finally {
            mainViewPanel.getDisplayOptionComboBox().addActionListener(l);
        }
    }

    public KeyEventDispatcher getNewHiCKeyDispatcher() {
        return new HiCKeyDispatcher(this, hic, mainViewPanel.getDisplayOptionComboBox());
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

    public boolean unsafeDisplayOptionComboBoxActionPerformed() {

        MatrixType option = (MatrixType) (mainViewPanel.getDisplayOptionComboBox().getSelectedItem());
        if (hic.isWholeGenome() && !MatrixType.isValidGenomeWideOption(option)) {
            JOptionPane.showMessageDialog(mainWindow, option + " matrix is not available for whole-genome view.");
            mainViewPanel.getDisplayOptionComboBox().setSelectedItem(hic.getDisplayOption());
            return false;
        }

        mainViewPanel.getColorRangePanel().handleNewFileLoading(option, MainViewPanel.preDefMapColor);

        if (MatrixType.isVSTypeDisplay(option)) {
            if (!hic.getMatrix().isIntra()) {
                JOptionPane.showMessageDialog(mainWindow, "Observed VS Control is not available for inter-chr views.");
                mainViewPanel.getDisplayOptionComboBox().setSelectedItem(hic.getDisplayOption());
                return false;
            }
        }

        if (MatrixType.isPearsonType(option)) {
            if (!hic.getMatrix().isIntra()) {
                JOptionPane.showMessageDialog(mainWindow, "Pearson's matrix is not available for inter-chr views.");
                mainViewPanel.getDisplayOptionComboBox().setSelectedItem(hic.getDisplayOption());
                return false;

            } else {
                try {
                    if (hic.isPearsonsNotAvailable(false)) {
                        JOptionPane.showMessageDialog(mainWindow, "Pearson's matrix is not available at this resolution");
                        mainViewPanel.getDisplayOptionComboBox().setSelectedItem(hic.getDisplayOption());
                        return false;
                    }
                    if (MatrixType.isControlPearsonType(option) && hic.isPearsonsNotAvailable(true)) {
                        JOptionPane.showMessageDialog(mainWindow, "Control's Pearson matrix is not available at this resolution");
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

    public LayersPanel getLayersPanel() {
        return layersPanel;
    }

    public MainMenuBar getMainMenuBar() {
        return mainMenuBar;
    }

    public void revalidate() {
        mainWindow.revalidate();
    }

    public void updateMainViewPanelToolTipText(String text) {
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

    public void repaintGridRulerPanels() {
        mainViewPanel.repaintGridRulerPanels();
    }

    // only hic should call this
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
        if (title != null && title.length() > 0) {
            if (control) controlTitle = title;
            else datasetTitle = title;
            updateTitle();
        }
    }

    private void updateTitle() {
        String newTitle = datasetTitle;
        String fileVersions = "";
        try {
            fileVersions += hic.getDataset().getVersion() + "";
        } catch (Exception e) {
        }

        if (controlTitle != null && controlTitle.length() > 0) {
            newTitle += "  (control=" + controlTitle + ")";
            try {
                fileVersions += "/" + hic.getControlDataset().getVersion();
            } catch (Exception e) {
            }
        }
        mainWindow.setTitle(HiCGlobals.juiceboxTitle + "<" + fileVersions + ">: " + newTitle);
    }

    private String getMapName() {
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
        getActiveLayerHandler().deleteTempFile();
    }

    public void setShowChromosomeFig(boolean status) {
        mainViewPanel.setShowChromosomeFig(status);
    }

    public boolean getShowGridLines() {
        return mainViewPanel.getShowGridLines();
    }

    public void setShowGridLines(boolean status) {
        mainViewPanel.setShowGridLines(status);
    }

    public AnnotationLayerHandler getActiveLayerHandler() {

        return activeLayer;
    }

    public void setActiveLayerHandler(AnnotationLayerHandler activeLayer) {
        this.activeLayer = activeLayer;
        for (AnnotationLayerHandler layer : annotationLayerHandlers) {
            layer.setActiveLayerButtonStatus(false);
        }
        activeLayer.setActiveLayerButtonStatus(true);
    }

    public AnnotationLayer.LayerType getActiveLayerType() {
        return activeLayer.getAnnotationLayer().getLayerType();
    }

    public List<AnnotationLayerHandler> getAllLayers() {
        return annotationLayerHandlers;
    }

    // mhoeger - Used for contig layer, currently returns the first element
    public List<AnnotationLayerHandler> getAssemblyLayerHandlers() {
        List<AnnotationLayerHandler> handlers = new ArrayList<>();
        for (AnnotationLayerHandler annotationLayerHandler : annotationLayerHandlers) {
            if (annotationLayerHandler.getAnnotationLayerType() == AnnotationLayer.LayerType.MAIN || annotationLayerHandler.getAnnotationLayerType() == AnnotationLayer.LayerType.GROUP || annotationLayerHandler.getAnnotationLayerType() == AnnotationLayer.LayerType.EDIT) {
                handlers.add(annotationLayerHandler);
            }
        }
        if (handlers.size() == 0) {
            handlers.add(annotationLayerHandlers.get(0));
        }
        return handlers;
    }

    private AnnotationLayerHandler getAssemblyLayerHandler(AnnotationLayer.LayerType layerType) {
        for (AnnotationLayerHandler annotationLayerHandler : getAssemblyLayerHandlers()) {
            if (annotationLayerHandler.getAnnotationLayerType() == layerType) {
                return annotationLayerHandler;
            }
        }
        return null;
    }

    public AnnotationLayerHandler getContigLayer() { //todo checkbox/ or something to specify assembly track
//        return annotationLayerHandlers.get(0);
//        List<AnnotationLayerHandler> handlers = new ArrayList<>();
//        for(AnnotationLayerHandler annotationLayerHandler : annotationLayerHandlers){
        if (getActiveLayerHandler().getAnnotationLayerType() == AnnotationLayer.LayerType.MAIN || (getActiveLayerHandler().getAnnotationLayerType() == AnnotationLayer.LayerType.GROUP)) {
            return getActiveLayerHandler();
        } else
            return annotationLayerHandlers.get(0);
    }

    public AnnotationLayerHandler getMainLayer() {
        return getAssemblyLayerHandler(AnnotationLayer.LayerType.MAIN);
    }

    public AnnotationLayerHandler getGroupLayer() {
        return getAssemblyLayerHandler(AnnotationLayer.LayerType.GROUP);
    }

    public AnnotationLayerHandler getEditLayer() {
        return getAssemblyLayerHandler(AnnotationLayer.LayerType.EDIT);
    }

    public AnnotationLayerHandler createNewLayer() {
        activeLayer = new AnnotationLayerHandler();
        annotationLayerHandlers.add(activeLayer);
        setActiveLayerHandler(activeLayer); // call this anyways because other layers need to fix button settings
        return activeLayer;
    }

    public void printNumFeatures() {
        for (AnnotationLayerHandler handler : annotationLayerHandlers) {
            System.out.println(handler.getLayerName() + " " + handler.getNumberOfFeatures());
        }
    }

    public int removeLayer(AnnotationLayerHandler handler) {
        int returnCode = -1;
        if (annotationLayerHandlers.size() > 1) {
            // must have at least 1 layer
            returnCode = annotationLayerHandlers.size() - 1 - annotationLayerHandlers.indexOf(handler);
            annotationLayerHandlers.remove(handler);
            if (handler == activeLayer) {
                // need to set a new active layer; let's use first one as default
                setActiveLayerHandler(annotationLayerHandlers.get(0));
            }
        }
        updateLayerDeleteStatus();
        return returnCode;
    }

    public void updateLayerDeleteStatus() {
        boolean isDeleteAllowed = annotationLayerHandlers.size() > 1;
        for (AnnotationLayerHandler handler : annotationLayerHandlers) {
            handler.setDeleteLayerButtonStatus(isDeleteAllowed);
        }
    }

    public int moveDownIndex(AnnotationLayerHandler handler) {
        int currIndex = annotationLayerHandlers.indexOf(handler);
        int n = annotationLayerHandlers.size();
        if (currIndex > 0) {
            Collections.swap(annotationLayerHandlers, currIndex, currIndex - 1);
            return n - currIndex;
        }
        return n - 1 - currIndex;
    }

    public int moveUpIndex(AnnotationLayerHandler handler) {
        int currIndex = annotationLayerHandlers.indexOf(handler);
        int n = annotationLayerHandlers.size();
        if (currIndex < n - 1) {
            Collections.swap(annotationLayerHandlers, currIndex, currIndex + 1);
            return n - 2 - currIndex;
        }
        return n - 1 - currIndex;
    }

    public void setPearsonColorScale(HiCColorScale pearsonColorScale) {
        this.pearsonColorScale = pearsonColorScale;
    }

    public String getTrackPanelPrintouts(int x, int y) {
        return mainViewPanel.getTrackPanelPrintouts(x, y);
    }

    public void setLayersPanelVisible(boolean status) {
        this.layerPanelIsVisible = status;
        if (layersPanel != null) {
            layersPanel.setVisible(status);
        } else {
            if (status) layersPanel = new LayersPanel(this);
            layersPanel.setVisible(status);
        }
        setLayersPanelGUIControllersSelected(status);
    }

    public void intializeLayersPanel() {
        layersPanel = new LayersPanel(this);
    }

    public void setLayersPanelGUIControllersSelected(boolean status) {
        mainViewPanel.setAnnotationsPanelToggleButtonSelected(status);
        mainMenuBar.setAnnotationPanelMenuItemSelected(status);
    }

    public void togglePanelVisible() {
        setLayersPanelVisible(!layerPanelIsVisible);
    }

    public AssemblyStateTracker getAssemblyStateTracker() {
        return assemblyStateTracker;
    }

    public void setAssemblyStateTracker(AssemblyStateTracker assemblyStateTracker) {
        this.assemblyStateTracker = assemblyStateTracker;
    }

    public void createCustomChromosomesFromBED() {

        FilenameFilter bedFilter = new FilenameFilter() {
            public boolean accept(File dir, String name) {
                return name.toLowerCase().endsWith(".bed");
            }
        };

        File[] files = FileDialogUtils.chooseMultiple("Choose .bed file(s)",
                LoadDialog.LAST_LOADED_HIC_FILE_PATH, bedFilter);

        if (files != null && files.length > 0) {
            LoadDialog.LAST_LOADED_HIC_FILE_PATH = files[0];

            int minSize = MotifAnchorTools.getMinSizeForExpansionFromGUI();

            for (File f : files) {
                Chromosome custom = hic.getChromosomeHandler().generateCustomChromosomeFromBED(f, minSize);
                updateChrHandlerAndMVP(custom);
            }
        }
    }

    public void createCustomChromosomeMap(Feature2DList featureList, String chrName) {
        Chromosome custom = hic.getChromosomeHandler().addCustomChromosome(featureList, chrName);

        updateChrHandlerAndMVP(custom);
    }

    private void updateChrHandlerAndMVP(Chromosome custom) {
        hic.setChromosomeHandler(hic.getChromosomeHandler());
        mainViewPanel.getChrBox1().addItem(custom);
        mainViewPanel.getChrBox2().addItem(custom);
    }

    public void updateMiniAnnotationsLayerPanel() {
        try {
            getMainViewPanel().updateMiniAnnotationsLayerPanel(this);
        } catch (Exception ignored) {
        }
    }

    public void updateMainLayersPanel() {
        try {
            getLayersPanel().updateLayers2DPanel(this);
        } catch (Exception ignored) {
        }
    }
}
