/*
 * Copyright (c) 2007-2012 The Broad Institute, Inc.
 * SOFTWARE COPYRIGHT NOTICE
 * This software and its documentation are the copyright of the Broad Institute, Inc. All rights are reserved.
 *
 * This software is supplied without any warranty or guaranteed support whatsoever. The Broad Institute is not responsible for its use, misuse, or functionality.
 *
 * This software is licensed under the terms of the GNU Lesser General Public License (LGPL),
 * Version 2.1 which is available at http://www.opensource.org/licenses/lgpl-2.1.php.
 */

/*
 * Created by JFormDesigner on Mon Aug 02 22:04:22 EDT 2010
 */

package juicebox;

import com.jidesoft.swing.JideButton;
import com.jidesoft.swing.JideSplitPane;
import org.apache.log4j.Logger;
import org.broad.igv.Globals;
import org.broad.igv.feature.Chromosome;
import juicebox.data.*;
import juicebox.tools.HiCTools;
import juicebox.track.*;
import org.broad.igv.ui.FontManager;
import org.broad.igv.ui.util.FileDialogUtils;
import org.broad.igv.ui.util.IconFactory;
import org.broad.igv.util.FileUtils;
import org.broad.igv.util.HttpUtils;
import org.broad.igv.util.ParsingUtils;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.LogarithmicAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import slider.RangeSlider;

import javax.accessibility.Accessible;
import javax.accessibility.AccessibleContext;
import javax.accessibility.AccessibleRole;
import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.text.html.HTMLEditorKit;
import javax.swing.text.html.StyleSheet;
import javax.swing.text.StyleConstants;
import javax.swing.text.StyledDocument;
import javax.swing.text.SimpleAttributeSet;
import javax.swing.border.EmptyBorder;
import javax.swing.border.LineBorder;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.event.TreeSelectionListener;
import javax.swing.JTree;
import javax.swing.plaf.ButtonUI;
import javax.swing.tree.TreePath;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.TreeSelectionModel;
import javax.swing.event.TreeSelectionEvent;

import java.awt.*;
import java.awt.datatransfer.DataFlavor;
import java.awt.datatransfer.Transferable;
import java.awt.dnd.*;
import java.awt.event.*;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.io.*;
import java.lang.reflect.InvocationTargetException;
import java.net.URL;
import java.text.DecimalFormat;
import java.util.*;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.prefs.BackingStoreException;
import java.util.prefs.Preferences;

/**
 * @author James Robinson
 */
public class MainWindow extends JFrame {

    private static Logger log = Logger.getLogger(MainWindow.class);
    private static final long serialVersionUID = 42L;
    private static final boolean isRestricted = true;
    private static RecentMenu recentMenu;
    private String currentlyLoadedFile = "";

    public static Color RULER_LINE_COLOR = new Color(0, 0, 0, 100);


    private ExecutorService threadExecutor = Executors.newFixedThreadPool(1);
    // The "model" object containing the state for this instance.
    private HiC hic;

    private String datasetTitle = "";
    private String controlTitle;

    public static Cursor fistCursor;

    public static final int BIN_PIXEL_WIDTH = 1;

    private static MainWindow theInstance;

    double colorRangeScaleFactor = 1;

    private LoadDialog loadDialog = null;

    private JComboBox<Chromosome> chrBox1;
    private JComboBox<Chromosome> chrBox2;
    private JideButton refreshButton;
    private JComboBox<String> normalizationComboBox;
    private JComboBox<MatrixType> displayOptionComboBox;
    private JideButton plusButton;
    private JideButton minusButton;
    private RangeSlider colorRangeSlider;
    private ResolutionControl resolutionSlider;
    private boolean resolutionLocked = false;

    TrackPanel trackPanelX;
    TrackLabelPanel trackLabelPanel;
    private HiCRulerPanel rulerPanelX;
    private HeatmapPanel heatmapPanel;
    TrackPanel trackPanelY;
    private HiCRulerPanel rulerPanelY;
    private ThumbnailPanel thumbnailPanel;
    private JLabel mouseHoverTextPanel;

    private JPanel hiCPanel;
    private JMenu annotationsMenu;
    private HiCZoom initialZoom;
    private String saveImagePath;

    private static int recentListMaxItems = 3;

    public void updateToolTipText(String s) {
        mouseHoverTextPanel.setText(s);
    }


    enum MatrixType {
        OBSERVED("Observed"),
        OE("OE"),
        PEARSON("Pearson"),
        EXPECTED("Expected"),
        RATIO("Observed / Control"),
        CONTROL("Control");
        private String value;

        MatrixType(String value) {
            this.value = value;
        }

        public String toString() {
            return value;
        }

    }


    public static void main(String[] args) throws IOException, InvocationTargetException, InterruptedException {
        initApplication();
        Runnable runnable = new Runnable() {
            public void run() {
                theInstance = getInstance();
                theInstance.setVisible(true);
                theInstance.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
                CommandListener.start(theInstance.hic);
            }
        };

        SwingUtilities.invokeAndWait(runnable);
    }


    private static void initApplication() {
        DirectoryManager.initializeLog();

        log.info("Default User Directory: " + DirectoryManager.getUserDirectory());
        System.setProperty("http.agent", Globals.applicationString());

        Runtime.getRuntime().addShutdownHook(new Thread() {
            @Override
            public void run() {
                if (Globals.IS_MAC) {
                    FileUtils.cleanupJnlpFiles();
                }
            }
        });
    }

    public static synchronized MainWindow getInstance() {
        if (theInstance == null) {
            try {
                theInstance = createMainWindow();
            } catch (IOException e) {
                log.error("Error creating main window", e);
            }
        }
        return theInstance;
    }

    private MainWindow() throws IOException {

        hic = new HiC(this);

        initComponents();
        createCursors();
        pack();

        DropTarget target = new DropTarget(this, new FileDropTargetListener());
        setDropTarget(target);

        colorRangeSlider.setUpperValue(1200);

        // Tooltip settings
        ToolTipManager.sharedInstance().setDismissDelay(60000);   // 60 seconds

        KeyboardFocusManager.getCurrentKeyboardFocusManager().addKeyEventDispatcher(new HiCKeyDispatcher());

    }

    public static MainWindow createMainWindow() throws IOException {
        return new MainWindow();
    }

    public boolean isResolutionLocked() {
        return resolutionLocked;
    }

    public void updateColorSlider(double min, double max, double value) {
        // We need to scale min and max to integers for the slider to work.  Scale such that there are
        // 100 divisions between max and 0

        colorRangeScaleFactor = 100.0 / max;

        int iMin = (int) (colorRangeScaleFactor * min);
        int iMax = (int) (colorRangeScaleFactor * max);
        int iValue = (int) (colorRangeScaleFactor * value);
        colorRangeSlider.setMinimum(iMin);
        colorRangeSlider.setMaximum(iMax);
        colorRangeSlider.setLowerValue(0);
        colorRangeSlider.setUpperValue(iValue);
    }

    public void updateColorSlider(double min, double lower, double upper, double max) {
        // We need to scale min and max to integers for the slider to work.  Scale such that there are
        // 100 divisions between max and 0

        colorRangeScaleFactor = 100.0 / max;

        int iMin = (int) (colorRangeScaleFactor * min);
        int iMax = (int) (colorRangeScaleFactor * max);
        int lValue = (int) (colorRangeScaleFactor * lower);
        int uValue = (int) (colorRangeScaleFactor * upper);
        colorRangeSlider.setMinimum(iMin);
        colorRangeSlider.setMaximum(iMax);
        colorRangeSlider.setLowerValue(lValue);
        colorRangeSlider.setUpperValue(uValue);
    }

    public void createCursors() {
        BufferedImage handImage = new BufferedImage(32, 32, BufferedImage.TYPE_INT_ARGB);

        // Make background transparent
        Graphics2D g = handImage.createGraphics();
        g.setComposite(AlphaComposite.getInstance(AlphaComposite.CLEAR, 0.0f));
        Rectangle2D.Double rect = new Rectangle2D.Double(0, 0, 32, 32);
        g.fill(rect);

        // Draw hand image in middle
        g = handImage.createGraphics();
        g.drawImage(IconFactory.getInstance().getIcon(IconFactory.IconID.FIST).getImage(), 0, 0, null);
        MainWindow.fistCursor = getToolkit().createCustomCursor(handImage, new Point(8, 6), "Move");
    }

    public HeatmapPanel getHeatmapPanel() {
        return heatmapPanel;
    }


    public void updateZoom(HiCZoom newZoom) {

        resolutionSlider.setZoom(newZoom);
    }

    /**
     * Chromosome "0" is whole genome
     *
     * @param chromosomes list of chromosomes
     */
    public void setChromosomes(List<Chromosome> chromosomes) {
        hic.setChromosomes(chromosomes);
        int[] chromosomeBoundaries = new int[chromosomes.size() - 1];
        long bound = 0;
        for (int i = 1; i < chromosomes.size(); i++) {
            Chromosome c = chromosomes.get(i);
            bound += (c.getLength() / 1000);
            chromosomeBoundaries[i - 1] = (int) bound;
        }
        heatmapPanel.setChromosomeBoundaries(chromosomeBoundaries);
    }


    public void setSelectedChromosomes(Chromosome xChrom, Chromosome yChrom) {
        chrBox1.setSelectedIndex(yChrom.getIndex());
        chrBox2.setSelectedIndex(xChrom.getIndex());
        refreshChromosomes();
    }

    public void setSelectedChromosomesNoRefresh(Chromosome xChrom, Chromosome yChrom) {
        chrBox1.setSelectedIndex(yChrom.getIndex());
        chrBox2.setSelectedIndex(xChrom.getIndex());
        rulerPanelX.setContext(hic.getXContext(), HiCRulerPanel.Orientation.HORIZONTAL);
        rulerPanelY.setContext(hic.getYContext(), HiCRulerPanel.Orientation.VERTICAL);
        resolutionSlider.setEnabled(!xChrom.getName().equals(Globals.CHR_ALL));
        initialZoom = null;
    }

    private void load(final List<String> files, final boolean control) {

        String file = files.get(0);
        
        if(file.equals(currentlyLoadedFile)){
            JOptionPane.showMessageDialog(MainWindow.this, "File already loaded");
            return;
        }
        else{
            currentlyLoadedFile = file;
        }

        if (file.endsWith("hic")) {
            Runnable runnable = new Runnable() {
                public void run() {
                    try {
                        DatasetReader reader = DatasetReaderFactory.getReader(files);
                        if (reader == null) return;
                        Dataset dataset = reader.read();

                        if (dataset.getVersion() <= 1) {
                            JOptionPane.showMessageDialog(MainWindow.this, "This version of \"hic\" format is no longer supported");
                            return;
                        }

                        MatrixType[] options;
                        if (control) {
                            hic.setControlDataset(dataset);
                            options = new MatrixType[]{MatrixType.OBSERVED, MatrixType.OE, MatrixType.PEARSON,
                                    MatrixType.EXPECTED, MatrixType.RATIO, MatrixType.CONTROL};
                        } else {
                            hic.reset();

                            hic.setDataset(dataset);

                            setChromosomes(dataset.getChromosomes());

                            chrBox1.setModel(new DefaultComboBoxModel<Chromosome>(hic.getChromosomes().toArray(new Chromosome[hic.getChromosomes().size()])));

                            chrBox2.setModel(new DefaultComboBoxModel<Chromosome>(hic.getChromosomes().toArray(new Chromosome[hic.getChromosomes().size()])));

                            String[] normalizationOptions;
                            if (dataset.getVersion() < 6) {
                                normalizationOptions = new String[]{NormalizationType.NONE.getLabel()};
                            } else {
                                ArrayList<String> tmp = new ArrayList<String>();
                                tmp.add(NormalizationType.NONE.getLabel());
                                for (NormalizationType t : hic.getDataset().getNormalizationTypes()) {
                                    tmp.add(t.getLabel());
                                }

                                normalizationOptions = tmp.toArray(new String[tmp.size()]);
                                //tmp.add(NormalizationType.LOADED.getLabel());

                            }

                            if (normalizationOptions.length == 1) {
                                normalizationComboBox.setEnabled(false);
                            } else {
                                normalizationComboBox.setModel(new DefaultComboBoxModel<String>(normalizationOptions));
                                normalizationComboBox.setSelectedIndex(0);
                                normalizationComboBox.setEnabled(hic.getDataset().getVersion() >= 6);
                            }

                            if (hic.isControlLoaded()) {
                                options = new MatrixType[]{MatrixType.OBSERVED, MatrixType.OE, MatrixType.PEARSON,
                                        MatrixType.EXPECTED, MatrixType.RATIO, MatrixType.CONTROL};
                            }
                            else {
                                options = new MatrixType[]{MatrixType.OBSERVED, MatrixType.OE, MatrixType.PEARSON, MatrixType.EXPECTED};
                            }


                            hic.resetContexts();
                            updateTrackPanel();
                            resolutionSlider.unit = HiC.Unit.BP;
                            resolutionSlider.reset();
                            refreshChromosomes();
                        }
                        displayOptionComboBox.setModel(new DefaultComboBoxModel<MatrixType>(options));
                        displayOptionComboBox.setSelectedIndex(0);
                        chrBox1.setEnabled(true);
                        chrBox2.setEnabled(true);
                        refreshButton.setEnabled(true);

                        colorRangeSlider.setEnabled(true);
                        plusButton.setEnabled(true);
                        minusButton.setEnabled(true);
                        annotationsMenu.setEnabled(true);
                        refresh(); // an additional refresh seems to remove the upper left black corner
                    } catch (IOException error) {
                        log.error("Error loading hic file", error);
                        JOptionPane.showMessageDialog(MainWindow.this, "Error loading .hic file", "Error", JOptionPane.ERROR_MESSAGE);
                        hic.reset();

                        updateThumbnail();

                    } catch (Exception error) {
                        error.printStackTrace();

                    }
                }
            };
            executeLongRunningTask(runnable);

        } else {
            JOptionPane.showMessageDialog(this, "Please choose a .hic file to load");

        }

    }


    public void refreshChromosomes() {

        Chromosome chr1 = (Chromosome) chrBox1.getSelectedItem();
        Chromosome chr2 = (Chromosome) chrBox2.getSelectedItem();

        Chromosome chrX = chr1.getIndex() < chr2.getIndex() ? chr1 : chr2;
        Chromosome chrY = chr1.getIndex() < chr2.getIndex() ? chr2 : chr1;

        hic.setSelectedChromosomes(chrX, chrY);
        rulerPanelX.setContext(hic.getXContext(), HiCRulerPanel.Orientation.HORIZONTAL);
        rulerPanelY.setContext(hic.getYContext(), HiCRulerPanel.Orientation.VERTICAL);
        setInitialZoom();

        // Test for new dataset ("All"),  or change in chromosome
        final boolean wholeGenome = chrY.getName().equals("All");
        final boolean intraChr = chr1.getIndex() != chr2.getIndex();
        if (wholeGenome || intraChr) {
            if (hic.getDisplayOption() == MatrixType.PEARSON) {
                hic.setDisplayOption(MatrixType.OBSERVED);
                displayOptionComboBox.setSelectedIndex(0);
            }
        }

        normalizationComboBox.setEnabled(!wholeGenome);
        // Actually we'd like to enable
        displayOptionComboBox.setEnabled(true);

        refresh();


    }

    void refresh() {
        getHeatmapPanel().clearTileCache();
        repaint();
        updateThumbnail();
    }

    private void updateThumbnail() {
        if (hic.getMatrix() != null) {

            //   MatrixZoomData zd0 = initialZoom == null ? hic.getMatrix().getFirstZoomData(hic.getZoom().getUnit()) :
            //           hic.getMatrix().getZoomData(initialZoom);
            MatrixZoomData zd0 = hic.getMatrix().getFirstZoomData(hic.getZoom().getUnit());
            MatrixZoomData zdControl = null;
            if (hic.getControlMatrix() != null) zdControl = hic.getControlMatrix().getFirstZoomData(hic.getZoom().getUnit());
            Image thumbnail = heatmapPanel.getThumbnailImage(
                    zd0,
                    zdControl,
                    thumbnailPanel.getWidth(),
                    thumbnailPanel.getHeight(),
                    hic.getDisplayOption());
            if (thumbnail != null) {
                thumbnailPanel.setImage(thumbnail);
                thumbnailPanel.repaint();
            }
        } else {
            thumbnailPanel.setImage(null);
        }
    }

    private void setInitialZoom() {

        if (hic.getXContext().getChromosome().getName().equals("All")) {
            resolutionSlider.setEnabled(false);
            initialZoom = hic.getMatrix().getFirstZoomData(HiC.Unit.BP).getZoom();
        } else {
            resolutionSlider.setEnabled(true);

            HiC.Unit currentUnit = hic.getZoom().getUnit();

            List<HiCZoom> zooms = (currentUnit == HiC.Unit.BP ? hic.getDataset().getBpZooms() :
                    hic.getDataset().getFragZooms());


//            Find right zoom level

            int pixels = getHeatmapPanel().getMinimumDimension();
            int len;
            if (currentUnit == HiC.Unit.BP) {
                len = (Math.max(hic.getXContext().getChrLength(), hic.getYContext().getChrLength()));
            } else {
                len = Math.max(hic.getDataset().getFragmentCounts().get(hic.getXContext().getChromosome().getName()),
                        hic.getDataset().getFragmentCounts().get(hic.getYContext().getChromosome().getName()));
            }

            int maxNBins = pixels / BIN_PIXEL_WIDTH;
            int bp_bin = len / maxNBins;
            initialZoom = zooms.get(zooms.size() - 1);
            for (int z = 1; z < zooms.size(); z++) {
                if (zooms.get(z).getBinSize() < bp_bin) {
                    initialZoom = zooms.get(z - 1);
                    break;
                }
            }

        }
        hic.setZoom(initialZoom, 0, 0);
        resolutionSlider.setZoom(initialZoom);
        resolutionSlider.reset();

    }

    private void refreshButtonActionPerformed() {
        refreshChromosomes();
    }

    private void loadMenuItemActionPerformed(boolean control) {
        // TODO: make this a real file filter so can filter by .hic
        File[] files = FileDialogUtils.chooseMultiple("Choose Hi-C file(s)", DirectoryManager.getUserDirectory(), null);
        if (files != null && files.length > 0) {
            List<String> fileNames = new ArrayList<String>();
            String str = "";
            for (File f : files) {
                fileNames.add(f.getAbsolutePath());
                str += f.getName() + " ";
            }
            load(fileNames, control);

            if (control) controlTitle = str;
            else datasetTitle = str;
            updateTitle();

        }
    }

    private void loadFromRecentActionPerformed(String url, String title, boolean control) {

        if (url != null) {
            try {
                load(Arrays.asList(url), control);

                String path = (new URL(url)).getPath();
                if (control) controlTitle = title;
                else datasetTitle = title;
                updateTitle();
            } catch (IOException e1) {
                JOptionPane.showMessageDialog(this, "Error while trying to load " + url, "Error", JOptionPane.ERROR_MESSAGE);
            }
        }
    }

    private void loadFromURLActionPerformed(boolean control) {
        String url = JOptionPane.showInputDialog("Enter URL: ");
        if (url != null) {
            try {
                load(Arrays.asList(url), control);

                String path = (new URL(url)).getPath();
                if (control) controlTitle = path;
                else datasetTitle = path;
                updateTitle();
            } catch (IOException e1) {
                JOptionPane.showMessageDialog(this, "Error while trying to load " + url, "Error", JOptionPane.ERROR_MESSAGE);
            }
        }
    }

    private void loadFromListActionPerformed(boolean control) {

        if (loadDialog == null) {
            Properties properties;
            try {
                String url = System.getProperty("jnlp.loadMenu");
                if (url == null) {
                    url = "http://hicfiles.econpy.org/juicebox.properties";
                }
                InputStream is = ParsingUtils.openInputStream(url);
                properties  = new Properties();
                properties.load(is);
            } catch (Exception error) {
                JOptionPane.showMessageDialog(this, "Can't find properties file for loading list", "Error", JOptionPane.ERROR_MESSAGE);
                return;
            }
            loadDialog = new LoadDialog(properties);
            if (!loadDialog.getSuccess()) {
                loadDialog = null;
                return;
            }
        }
        loadDialog.setControl(control);
        loadDialog.setVisible(true);

    }


    private void updateTitle() {
        String newTitle = datasetTitle;
        if (controlTitle != null) newTitle += "  (control=" + controlTitle + ")";
        setTitle(newTitle);
    }

    private void clearActionPerformed() {
        String HIC_RECENT = "hicRecent";
        Preferences prefs = Preferences.userNodeForPackage(Globals.class);
            for (int i = 0; i < recentListMaxItems; i++)
            {
                prefs.remove(HIC_RECENT+i);
            }
    }


    private void exitActionPerformed() {
        setVisible(false);
        dispose();
        System.exit(0);
    }



    private void colorRangeSliderStateChanged(ChangeEvent e) {
        double min = colorRangeSlider.getLowerValue() / colorRangeScaleFactor;
        double max = colorRangeSlider.getUpperValue() / colorRangeScaleFactor;

        heatmapPanel.setObservedRange(min, max);
/*
        if (hic.getDisplayOption() == MatrixType.OE) {
            heatmapPanel.setOEMax(colorRangeSlider.getUpperValue() / 8);
        }*/
    }

    private void chrBox1ActionPerformed(ActionEvent e) {
        if (chrBox1.getSelectedIndex() == 0) {
            chrBox2.setSelectedIndex(0);
        }
    }

    private void chrBox2ActionPerformed(ActionEvent e) {
        if (chrBox2.getSelectedIndex() == 0) {
            chrBox1.setSelectedIndex(0);
        }
    }

    private void displayOptionComboBoxActionPerformed(ActionEvent e) {

        MatrixType option = (MatrixType) (displayOptionComboBox.getSelectedItem());
       // ((ColorRangeModel)colorRangeSlider.getModel()).setObserved(option == MatrixType.OBSERVED || option == MatrixType.CONTROL || option == MatrixType.EXPECTED);
        colorRangeSlider.setEnabled(option == MatrixType.OBSERVED || option == MatrixType.CONTROL);
        plusButton.setEnabled(option == MatrixType.OBSERVED || option == MatrixType.CONTROL);
        minusButton.setEnabled(option == MatrixType.OBSERVED || option == MatrixType.CONTROL);
        if (option == MatrixType.PEARSON) {
            if (hic.isWholeGenome()) {
                JOptionPane.showMessageDialog(this, "Pearson's matrix is not available for whole-genome view.");
                displayOptionComboBox.setSelectedItem(hic.getDisplayOption());
                return;

            }
            if (!hic.getMatrix().isIntra()) {
                JOptionPane.showMessageDialog(this, "Pearson's matrix is not available for inter-chr views.");
                displayOptionComboBox.setSelectedItem(hic.getDisplayOption());
                return;

            } else if (hic.getZd().getPearsons(hic.getDataset().getExpectedValues(hic.getZd().getZoom(), hic.getNormalizationType())) == null) {
                JOptionPane.showMessageDialog(this, "Pearson's matrix is not available at this resolution");
                displayOptionComboBox.setSelectedItem(hic.getDisplayOption());
                return;
            }
        }

        hic.setDisplayOption(option);
        refresh();

    }

    private void normalizationComboBoxActionPerformed(ActionEvent e) {
        String value = (String)normalizationComboBox.getSelectedItem();
        NormalizationType chosen = null;
        for (NormalizationType type:NormalizationType.values()) {
            if (type.getLabel().equals(value)) {
                chosen = type;
                break;
            }
        }
        hic.setNormalizationType(chosen);
        refresh();


    }

    /**
     * Utility function to execute a task in a worker thread.  The method is on MainWindow because the glassPane
     * is used to display a wait cursor and block events.
     *
     * @param runnable Thread
     * @return thread
     */

    public Future<?> executeLongRunningTask(final Runnable runnable) {

        Callable<Object> wrapper = new Callable<Object>() {
            public Object call() throws Exception {
                final Component glassPane = showGlassPane();
                try {
                    runnable.run();
                    return "done";
                } finally {
                    glassPane.setVisible(false);
                }

            }
        };

        return threadExecutor.submit(wrapper);
    }

    public Component showGlassPane() {
        final Component glassPane = getGlassPane();
        glassPane.setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));
        glassPane.setVisible(true);
        return glassPane;
    }

    public void hideGlassPane() {
        getGlassPane().setVisible(false);
    }

    public void updateTrackPanel() {
        boolean hasTracks = hic.getLoadedTracks().size() > 0;

        trackLabelPanel.updateLabels();

        if (hasTracks) {
            if (!trackPanelX.isVisible()) {
                trackPanelX.setVisible(true);
                trackLabelPanel.setVisible(true);
            }
            if (!trackPanelY.isVisible()) {
                trackPanelY.setVisible(true);
            }
        } else {
            if (trackPanelX.isVisible()) {
                trackPanelX.setVisible(false);
                trackLabelPanel.setVisible(false);
            }
            if (trackPanelY.isVisible()) {
                trackPanelY.setVisible(false);
            }
        }


        trackPanelX.invalidate();
        trackLabelPanel.invalidate();
        trackPanelY.invalidate();
        getContentPane().invalidate();
        repaint();

    }


    /**
     * Listener for drag&drop actions
     */
    class FileDropTargetListener implements DropTargetListener {


        public FileDropTargetListener() {
        }

        public void dragEnter(DropTargetDragEvent event) {

            if (!isDragAcceptable(event)) {
                event.rejectDrag();
            }
        }

        public void dragExit(DropTargetEvent event) {
        }

        public void dragOver(DropTargetDragEvent event) {
            // you can provide visual feedback here
        }

        public void dropActionChanged(DropTargetDragEvent event) {
            if (!isDragAcceptable(event)) {
                event.rejectDrag();
            }
        }

        public void drop(DropTargetDropEvent event) {
            if (!isDropAcceptable(event)) {
                event.rejectDrop();
                return;
            }

            event.acceptDrop(DnDConstants.ACTION_COPY);

            Transferable transferable = event.getTransferable();

            try {
                @SuppressWarnings("unchecked") // Transferable when called with DataFlavor javaFileList is guaranteed to retunr a File List.
                java.util.List<File> files = (java.util.List<File>) transferable.getTransferData(DataFlavor.javaFileListFlavor);
                List<String> paths = new ArrayList<String>();
                for (File f : files) {
                    paths.add(f.getAbsolutePath());
                }
                load(paths, false);

            } catch (Exception e) {
                String obj;
                try {
                    obj = transferable.getTransferData(DataFlavor.stringFlavor).toString();
                    if (HttpUtils.isRemoteURL(obj)) {
                        load(Arrays.asList(obj), false);
                    }
                } catch (Exception e1) {
                    e1.printStackTrace();
                }

            }
            repaint();
            event.dropComplete(true);
        }


        public boolean isDragAcceptable(DropTargetDragEvent event) {
            //  Check the  available data flavors here
            //  Currently accepting all flavors
            return (event.getDropAction() & DnDConstants.ACTION_COPY_OR_MOVE) != 0;
        }

        public boolean isDropAcceptable(DropTargetDropEvent event) {
            //  Check the  available data flavors here
            //  Currently accepting all flavors
            return (event.getDropAction() & DnDConstants.ACTION_COPY_OR_MOVE) != 0;
        }
    }

    private void initComponents() {

        Container contentPane = getContentPane();
        contentPane.setLayout(new BorderLayout());

        final JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new BorderLayout());
        contentPane.add(mainPanel, BorderLayout.CENTER);
        mainPanel.setBackground(Color.white);

        final JPanel toolbarPanel = new JPanel();
        toolbarPanel.setBorder(null);
        toolbarPanel.setLayout(new GridLayout());
        mainPanel.add(toolbarPanel, BorderLayout.NORTH);

        JideSplitPane splitPanel = new JideSplitPane(JideSplitPane.HORIZONTAL_SPLIT);
        splitPanel.setShowGripper(true);
        splitPanel.setDividerSize(5);
        splitPanel.setBackground(Color.darkGray);
        mainPanel.add(splitPanel, BorderLayout.CENTER);

        JMenuBar menuBar = null;
        try {
            menuBar = createMenuBar();
        } catch (BackingStoreException e) {
            e.printStackTrace();
        }
        contentPane.add(menuBar, BorderLayout.NORTH);


        // --- Chromosome panel ---
        JPanel chrSelectionPanel = new JPanel();
        toolbarPanel.add(chrSelectionPanel);

        chrSelectionPanel.setBorder(LineBorder.createGrayLineBorder());
        chrSelectionPanel.setMinimumSize(new Dimension(130, 57));
        chrSelectionPanel.setPreferredSize(new Dimension(130, 57));
        chrSelectionPanel.setLayout(new BorderLayout());

        JPanel chrLabelPanel = new JPanel();
        JLabel chrLabel = new JLabel("Chromosomes");
        chrLabel.setHorizontalAlignment(SwingConstants.CENTER);
        chrLabelPanel.setBackground(new Color(204, 204, 204));
        chrLabelPanel.setLayout(new BorderLayout());
        chrLabelPanel.add(chrLabel, BorderLayout.CENTER);
        chrSelectionPanel.add(chrLabelPanel, BorderLayout.PAGE_START);

        JPanel chrButtonPanel = new JPanel();
        chrButtonPanel.setBackground(new Color(238, 238, 238));
        chrButtonPanel.setLayout(new BoxLayout(chrButtonPanel, BoxLayout.X_AXIS));

        //---- chrBox1 ----
        chrBox1 = new JComboBox<Chromosome>();
        chrBox1.setModel(new DefaultComboBoxModel<Chromosome>(new Chromosome[]{ new Chromosome(0,"All",0)}));
        chrBox1.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                chrBox1ActionPerformed(e);
            }
        });
        chrButtonPanel.add(chrBox1);

        //---- chrBox2 ----
        chrBox2 = new JComboBox<Chromosome>();
        chrBox2.setModel(new DefaultComboBoxModel<Chromosome>(new Chromosome[]{ new Chromosome(0,"All",0)}));
        chrBox2.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                chrBox2ActionPerformed(e);
            }
        });
        chrButtonPanel.add(chrBox2);


        //---- refreshButton ----
        refreshButton = new JideButton();
        refreshButton.setIcon(new ImageIcon(getClass().getResource("/toolbarButtonGraphics/general/Refresh24.gif")));
        refreshButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                refreshButtonActionPerformed();
            }
        });
        chrButtonPanel.add(refreshButton);

        chrBox1.setEnabled(false);
        chrBox2.setEnabled(false);
        refreshButton.setEnabled(false);
        chrSelectionPanel.add(chrButtonPanel, BorderLayout.CENTER);


        //======== normalizationPanel ========
        JPanel normalizationPanel = new JPanel();
        normalizationPanel.setBackground(new Color(238, 238, 238));
        normalizationPanel.setBorder(LineBorder.createGrayLineBorder());
        normalizationPanel.setLayout(new BorderLayout());

        JPanel normalizationLabelPanel = new JPanel();
        normalizationLabelPanel.setBackground(new Color(204, 204, 204));
        normalizationLabelPanel.setLayout(new BorderLayout());

        JLabel normalizationLabel = new JLabel("Normalization");
        normalizationLabel.setHorizontalAlignment(SwingConstants.CENTER);
        normalizationLabelPanel.add(normalizationLabel, BorderLayout.CENTER);
        normalizationPanel.add(normalizationLabelPanel, BorderLayout.PAGE_START);

        JPanel normalizationButtonPanel = new JPanel();
        normalizationButtonPanel.setBorder(new EmptyBorder(0, 10, 0, 10));
        normalizationButtonPanel.setLayout(new GridLayout(1, 0, 20, 0));
        normalizationComboBox = new JComboBox<String>();
        normalizationComboBox.setModel(new DefaultComboBoxModel<String>(new String[]{NormalizationType.NONE.getLabel()}));
        normalizationComboBox.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                normalizationComboBoxActionPerformed(e);
            }
        });
        normalizationButtonPanel.add(normalizationComboBox);
        normalizationPanel.add(normalizationButtonPanel, BorderLayout.CENTER);
        toolbarPanel.add(normalizationPanel);
        normalizationComboBox.setEnabled(false);


        //======== displayOptionPanel ========
        JPanel displayOptionPanel = new JPanel();
        displayOptionPanel.setBackground(new Color(238, 238, 238));
        displayOptionPanel.setBorder(LineBorder.createGrayLineBorder());
        displayOptionPanel.setLayout(new BorderLayout());
        JPanel displayOptionLabelPanel = new JPanel();
        displayOptionLabelPanel.setBackground(new Color(204, 204, 204));
        displayOptionLabelPanel.setLayout(new BorderLayout());

        JLabel displayOptionLabel = new JLabel("Show");
        displayOptionLabel.setHorizontalAlignment(SwingConstants.CENTER);
        displayOptionLabelPanel.add(displayOptionLabel, BorderLayout.CENTER);
        displayOptionPanel.add(displayOptionLabelPanel, BorderLayout.PAGE_START);
        JPanel displayOptionButtonPanel = new JPanel();
        displayOptionButtonPanel.setBorder(new EmptyBorder(0, 10, 0, 10));
        displayOptionButtonPanel.setLayout(new GridLayout(1, 0, 20, 0));
        displayOptionComboBox = new JComboBox<MatrixType>();
        displayOptionComboBox.setModel(new DefaultComboBoxModel<MatrixType>(new MatrixType[]{MatrixType.OBSERVED}));
        displayOptionComboBox.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                displayOptionComboBoxActionPerformed(e);
            }
        });
        displayOptionButtonPanel.add(displayOptionComboBox);
        displayOptionPanel.add(displayOptionButtonPanel, BorderLayout.CENTER);
        toolbarPanel.add(displayOptionPanel);
        displayOptionComboBox.setEnabled(false);

        //======== colorRangePanel ========

        JPanel colorRangePanel = new JPanel();
        colorRangePanel.setLayout(new BorderLayout());

        JPanel sliderPanel = new JPanel();
        sliderPanel.setLayout(new BoxLayout(sliderPanel, BoxLayout.X_AXIS));

        colorRangeSlider = new RangeSlider();
        //colorRangeSlider.setModel(new ColorRangeModel());
        colorRangeSlider.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseEntered(MouseEvent mouseEvent) {
                super.mouseEntered(mouseEvent);
                if (hic.getDisplayOption() == MatrixType.OBSERVED) {
                    colorRangeSlider.setToolTipText("<html>Range: " + (int) (colorRangeSlider.getMinimum() / colorRangeScaleFactor) + " "
                            + (int) (colorRangeSlider.getMaximum() / colorRangeScaleFactor) + "<br>Showing: "+
                            (int) (colorRangeSlider.getLowerValue() / colorRangeScaleFactor) + " "
                            + (int) (colorRangeSlider.getUpperValue() / colorRangeScaleFactor)
                            +"</html>" );
                } else if (hic.getDisplayOption() == MatrixType.OE) {
                    double mymaximum = colorRangeSlider.getMaximum() / 8;
                    colorRangeSlider.setToolTipText("Range: " + new DecimalFormat("##.##").format(1 / mymaximum) + " "
                            + new DecimalFormat("##.##").format(mymaximum));
                }
            }
        });
        colorRangeSlider.setEnabled(false);

        //---- colorRangeLabel ----
        JLabel colorRangeLabel = new JLabel("Color Range");
        colorRangeLabel.setHorizontalAlignment(SwingConstants.CENTER);
        colorRangeLabel.setToolTipText("Range of color scale in counts per mega-base squared.");
        colorRangeLabel.setHorizontalTextPosition(SwingConstants.CENTER);
        colorRangeLabel.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                if (e.isPopupTrigger() && colorRangeSlider.isEnabled()) {
                    ColorRangeDialog rangeDialog = new ColorRangeDialog(MainWindow.this, colorRangeSlider, colorRangeScaleFactor, hic.getDisplayOption() == MatrixType.OBSERVED);
                    rangeDialog.setVisible(true);
                }
            }

            @Override
            public void mouseClicked(MouseEvent e) {
                if (colorRangeSlider.isEnabled()) {
                    ColorRangeDialog rangeDialog = new ColorRangeDialog(MainWindow.this, colorRangeSlider, colorRangeScaleFactor, hic.getDisplayOption() == MatrixType.OBSERVED);
                    rangeDialog.setVisible(true);
                }
            }
        });
        JPanel colorLabelPanel = new JPanel();
        colorLabelPanel.setBackground(new Color(204, 204, 204));
        colorLabelPanel.setLayout(new BorderLayout());
        colorLabelPanel.add(colorRangeLabel, BorderLayout.CENTER);

        colorRangePanel.add(colorLabelPanel, BorderLayout.PAGE_START);

        //---- colorRangeSlider ----
        colorRangeSlider.setPaintTicks(false);
        colorRangeSlider.setPaintLabels(false);
        colorRangeSlider.setMaximumSize(new Dimension(32767, 52));
        colorRangeSlider.setPreferredSize(new Dimension(200, 52));
        colorRangeSlider.setMinimumSize(new Dimension(36, 52));

        colorRangeSlider.setMaximum(2000);
        colorRangeSlider.setLowerValue(0);
        colorRangeSlider.setUpperValue(500);
        colorRangeSlider.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent e) {
                colorRangeSliderStateChanged(e);
            }
        });
        sliderPanel.add(colorRangeSlider);
        JPanel plusMinusPanel = new JPanel();
        plusMinusPanel.setLayout(new BoxLayout(plusMinusPanel, BoxLayout.Y_AXIS));
        plusButton = new JideButton();

        plusButton.setIcon(new ImageIcon(getClass().getResource("/images/zoom-plus.png")));
        plusButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                colorRangeSlider.setMaximum(colorRangeSlider.getMaximum() * 2);
            }
        });

        minusButton = new JideButton();

        minusButton.setIcon(new ImageIcon(getClass().getResource("/images/zoom-minus.png")));
        minusButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                colorRangeSlider.setMaximum(colorRangeSlider.getMaximum() / 2);
            }
        });

        plusMinusPanel.add(plusButton);
        plusMinusPanel.add(minusButton);
        plusButton.setEnabled(false);
        minusButton.setEnabled(false);
        sliderPanel.add(plusMinusPanel);
        colorRangePanel.add(sliderPanel, BorderLayout.PAGE_END);


        colorRangePanel.setBorder(LineBorder.createGrayLineBorder());
        colorRangePanel.setMinimumSize(new Dimension(96, 70));
        colorRangePanel.setPreferredSize(new Dimension(202, 70));
        colorRangePanel.setMaximumSize(new Dimension(32769, 70));
        toolbarPanel.add(colorRangePanel);



        // Resolution  panel
        resolutionSlider = new ResolutionControl();
        toolbarPanel.add(resolutionSlider);


        toolbarPanel.setEnabled(false);

        //======== hiCPanel ========
        hiCPanel = new JPanel();
        hiCPanel.setBackground(Color.white);
        hiCPanel.setLayout(new HiCLayout());
        splitPanel.insertPane(hiCPanel, 0);
        splitPanel.setBackground(Color.white);

        //---- rulerPanel2 ----
        JPanel topPanel = new JPanel();
        topPanel.setBackground(Color.white);
        topPanel.setLayout(new BorderLayout());
        hiCPanel.add(topPanel, BorderLayout.NORTH);
        trackLabelPanel = new TrackLabelPanel(hic);
        trackLabelPanel.setBackground(Color.white);
        hiCPanel.add(trackLabelPanel, HiCLayout.NORTH_WEST);

        trackPanelX = new TrackPanel(this, hic, TrackPanel.Orientation.X);
        trackPanelX.setMaximumSize(new Dimension(4000, 50));
        trackPanelX.setPreferredSize(new Dimension(1, 50));
        trackPanelX.setMinimumSize(new Dimension(1, 50));
        trackPanelX.setVisible(false);
        topPanel.add(trackPanelX, BorderLayout.NORTH);


        rulerPanelX = new HiCRulerPanel(hic);
        rulerPanelX.setMaximumSize(new Dimension(4000, 50));
        rulerPanelX.setMinimumSize(new Dimension(1, 50));
        rulerPanelX.setPreferredSize(new Dimension(1, 50));
        rulerPanelX.setBorder(null);
        topPanel.add(rulerPanelX, BorderLayout.SOUTH);


        //---- rulerPanel1 ----
        JPanel leftPanel = new JPanel();
        leftPanel.setBackground(Color.white);
        leftPanel.setLayout(new BorderLayout());
        hiCPanel.add(leftPanel, BorderLayout.WEST);

        trackPanelY = new TrackPanel(this, hic, TrackPanel.Orientation.Y);
        trackPanelY.setMaximumSize(new Dimension(50, 4000));
        trackPanelY.setPreferredSize(new Dimension(50, 1));
        trackPanelY.setMinimumSize(new Dimension(50, 1));
        trackPanelY.setVisible(false);
        leftPanel.add(trackPanelY, BorderLayout.WEST);

        rulerPanelY = new HiCRulerPanel(hic);
        rulerPanelY.setMaximumSize(new Dimension(50, 4000));
        rulerPanelY.setPreferredSize(new Dimension(50, 800));
        rulerPanelY.setBorder(null);
        rulerPanelY.setMinimumSize(new Dimension(50, 1));
        leftPanel.add(rulerPanelY, BorderLayout.EAST);

        //---- heatmapPanel ----
        heatmapPanel = new HeatmapPanel(this, hic);
        heatmapPanel.setBorder(LineBorder.createBlackLineBorder());
        heatmapPanel.setMaximumSize(new Dimension(800, 800));
        heatmapPanel.setMinimumSize(new Dimension(800, 800));
        heatmapPanel.setPreferredSize(new Dimension(800, 800));
        // heatmapPanel.setBackground(new Color(238, 238, 238));
        heatmapPanel.setBackground(Color.white);
        hiCPanel.add(heatmapPanel, BorderLayout.CENTER);


        //======== panel8 ========

        JPanel rightSidePanel = new JPanel(new BorderLayout());
        rightSidePanel.setBackground(Color.white);
        rightSidePanel.setPreferredSize(new Dimension(200, 1000));
        rightSidePanel.setMaximumSize(new Dimension(10000, 10000));
        rightSidePanel.setBorder(new EmptyBorder(0, 10, 0, 0));
        //LayoutManager lm = new FlowLayout(FlowLayout.LEFT, 10, 20);
        //rightSidePanel.setLayout(lm);
        //rightSidePanel.setLayout(null);

        JPanel thumbPanel = new JPanel();
        //thumbPanel.setLayout(null);
        //---- thumbnailPanel ----
        thumbnailPanel = new ThumbnailPanel(this, hic);
        thumbnailPanel.setBackground(Color.white);
        thumbnailPanel.setMaximumSize(new Dimension(200, 200));
        thumbnailPanel.setMinimumSize(new Dimension(200, 200));
        thumbnailPanel.setPreferredSize(new Dimension(200, 200));
        thumbnailPanel.setBorder(LineBorder.createBlackLineBorder());
        thumbnailPanel.setPreferredSize(new Dimension(200, 200));
        thumbnailPanel.setBounds(new Rectangle(new Point(20, 0), thumbnailPanel.getPreferredSize()));
        thumbPanel.add(thumbnailPanel);
        thumbPanel.setBackground(Color.white);
        rightSidePanel.add(thumbPanel, BorderLayout.NORTH);

        //========= mouse hover text ======

        mouseHoverTextPanel = new JLabel();
        mouseHoverTextPanel.setBackground(Color.white);
        mouseHoverTextPanel.setVerticalAlignment(SwingConstants.TOP);
        mouseHoverTextPanel.setHorizontalAlignment(SwingConstants.LEFT);
        mouseHoverTextPanel.setBorder(LineBorder.createBlackLineBorder());
        int mouseTextY = thumbnailPanel.getBounds().y + thumbnailPanel.getBounds().height + 20;
        Dimension prefSize = new Dimension(200, 500);
        mouseHoverTextPanel.setBounds(new Rectangle(new Point(20, mouseTextY), prefSize));
        rightSidePanel.add(mouseHoverTextPanel);

        //======== xPlotPanel ========
//
//        xPlotPanel = new JPanel();
//        xPlotPanel.setPreferredSize(new Dimension(250, 100));
//        xPlotPanel.setLayout(null);
//
//        rightSidePanel.add(xPlotPanel);
//        xPlotPanel.setBounds(10, 100, xPlotPanel.getPreferredSize().width, 228);
//
//        //======== yPlotPanel ========
//
//        yPlotPanel = new JPanel();
//        yPlotPanel.setPreferredSize(new Dimension(250, 100));
//        yPlotPanel.setLayout(null);
//
//        rightSidePanel.add(yPlotPanel);
//        yPlotPanel.setBounds(10, 328, yPlotPanel.getPreferredSize().width, 228);

        // compute preferred size
        Dimension preferredSize = new Dimension();
        for (int i = 0; i < rightSidePanel.getComponentCount(); i++) {
            Rectangle bounds = rightSidePanel.getComponent(i).getBounds();
            preferredSize.width = Math.max(bounds.x + bounds.width, preferredSize.width);
            preferredSize.height = Math.max(bounds.y + bounds.height, preferredSize.height);
        }
        Insets insets = rightSidePanel.getInsets();
        preferredSize.width += insets.right + 20;
        preferredSize.height += insets.bottom;
        rightSidePanel.setMinimumSize(preferredSize);
        rightSidePanel.setPreferredSize(preferredSize);


        splitPanel.insertPane(rightSidePanel, 1);
        // hiCPanel.add(rightSidePanel, BorderLayout.EAST);


        // setup the glass pane to display a wait cursor when visible, and to grab all mouse events
        rootPane.getGlassPane().setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));
        rootPane.getGlassPane().addMouseListener(new MouseAdapter() {
        });

    }


    private JMenuBar createMenuBar() throws BackingStoreException {


        JMenuBar menuBar = new JMenuBar();

        //======== fileMenu ========
        JMenu fileMenu = new JMenu("File");
        fileMenu.setMnemonic('F');

        //---- openMenuItem ----
        JMenuItem openItem = new JMenuItem("Open...");

        openItem.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                loadFromListActionPerformed(false);
            }
        });
        fileMenu.add(openItem);

        JMenuItem loadControlFromList = new JMenuItem();
        loadControlFromList.setText("Open Control...");
        loadControlFromList.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                loadFromListActionPerformed(true);
            }
        });
        fileMenu.add(loadControlFromList);

        fileMenu.addSeparator();
        //---- recent positions ----
        try {
            recentMenu=new RecentMenu(recentListMaxItems){
                public void onSelectPosition(String mapPath){

                    //TBD - Prepare call to setstate.
                    //hic.setState(chrXName, chrYName, unitName, binSize, xOrigin, yOrigin, scaleFactor);
                    String delimiter = "@@";
                    String[] temp;
                    temp = mapPath.split(delimiter);

                    loadFromRecentActionPerformed((temp[1]), (temp[0]), false);
                }
            };
        } catch (BackingStoreException e) {
            e.printStackTrace();
        }

        fileMenu.add(recentMenu);

        //---- Clear Recent ----
        JMenuItem clear = new JMenuItem();
        clear.setText("Clear Recent maps list");
        clear.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                //Clear all items from preferences:
                clearActionPerformed();
                //clear the existing items
                recentMenu.removeAll();
            }
        });
        fileMenu.add(clear);


        fileMenu.addSeparator();

        annotationsMenu = new JMenu("Annotations");

        JMenuItem newLoadMI = new JMenuItem();
        newLoadMI.setAction(new LoadAction("Load Basic Annotations...", this, hic));
        annotationsMenu.add(newLoadMI);

        /*
        JMenuItem loadSpecificMI = new JMenuItem();
        loadSpecificMI.setAction(new LoadEncodeAction("Load Tracks by Cell Type...", this, hic, "hic"));
        annotationsMenu.add(loadSpecificMI);
        */

        JMenuItem loadEncodeMI = new JMenuItem();
        loadEncodeMI.setAction(new LoadEncodeAction("Load ENCODE Tracks...", this, hic));
        annotationsMenu.add(loadEncodeMI);


        final JCheckBoxMenuItem showLoopsItem = new JCheckBoxMenuItem("Show 2D Annotations");

        showLoopsItem.setSelected(true);
        showLoopsItem.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                hic.setShowLoops(showLoopsItem.isSelected());
                repaint();
            }
        });
        showLoopsItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_F2, 0));

        annotationsMenu.add(showLoopsItem);

        annotationsMenu.setEnabled(false);

        JMenuItem loadFromURLItem = new JMenuItem("Load Annotation from URL...");
        loadFromURLItem.addActionListener(new AbstractAction() {
            private static final long serialVersionUID = 42L;

            @Override
            public void actionPerformed(ActionEvent e) {
                if (hic.getDataset() == null) {
                    JOptionPane.showMessageDialog(MainWindow.this, "HiC file must be loaded to load tracks", "Error", JOptionPane.ERROR_MESSAGE);
                    return;
                }

                String url = JOptionPane.showInputDialog("Enter URL: ");
                if (url != null) {
                    hic.loadTrack(url);

                }

            }
        });


        JMenuItem showStats = new JMenuItem("Show Dataset Metrics");
        showStats.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                if (hic.getDataset() == null) {
                    JOptionPane.showMessageDialog(MainWindow.this, "File must be loaded to show info", "Error", JOptionPane.ERROR_MESSAGE);
                } else {
                    JDialog qcDialog = new QCDialog(hic.getDataset());
                    qcDialog.setTitle(MainWindow.this.getTitle() + " info");
                    qcDialog.setVisible(true);

                }
            }
        });


        fileMenu.add(showStats);
        fileMenu.addSeparator();

        JMenuItem saveToImage = new JMenuItem();
        saveToImage.setText("Export Image...");
        saveToImage.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                new SaveImageDialog();

            }
        });
        fileMenu.add(saveToImage);


        if (!isRestricted) {
            JMenuItem dump = new JMenuItem("Export Data...");
            dump.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent actionEvent) {
                    if (hic.getDataset() == null) {
                        JOptionPane.showMessageDialog(MainWindow.this, "File must be loaded to show info", "Error", JOptionPane.ERROR_MESSAGE);
                    } else {
                        new DumpDialog();
                    }

                }
            });
            fileMenu.add(dump);
        }

        //---- exit ----
        JMenuItem exit = new JMenuItem();
        exit.setText("Exit");
        exit.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                exitActionPerformed();
            }
        });
        fileMenu.add(exit);

        menuBar.add(fileMenu);
        menuBar.add(annotationsMenu);
        return menuBar;
    }

    private void loadNormalizationVector(File file) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
        String nextLine = reader.readLine();
        String[] tokens = Globals.singleTabMultiSpacePattern.split(nextLine);
        int resolution = Integer.valueOf(tokens[0]);
        int vectorLength = Integer.valueOf(tokens[1]);
        int expectedLength = Integer.valueOf(tokens[2]);
        List<Chromosome> chromosomes = hic.getChromosomes();

        double[] nv = new double[vectorLength];
        double[] exp = new double[expectedLength];
        for (int i = 0; i < nv.length; i++) {
            nextLine = reader.readLine();
            tokens = Globals.singleTabMultiSpacePattern.split(nextLine);
            nv[i] = Double.valueOf(tokens[0]);
        }
        for (int i = 0; i < exp.length; i++) {
            nextLine = reader.readLine();
            tokens = Globals.singleTabMultiSpacePattern.split(nextLine);
            exp[i] = Double.valueOf(tokens[0]);
        }

        int location1 = 0;
        for (Chromosome c1 : chromosomes) {
            if (c1.getName().equals(Globals.CHR_ALL)) continue;
            int chrBinned = c1.getLength() / resolution + 1;
            double[] chrNV = new double[chrBinned];
            for (int i = 0; i < chrNV.length; i++) {
                chrNV[i] = nv[location1];
                location1++;
            }
            hic.getDataset().putLoadedNormalizationVector(c1.getIndex(), resolution, chrNV, exp);
        }

    }

    private class QCDialog extends JDialog {
        private final long[] logXAxis = {10, 12, 15, 19, 23, 28, 35, 43, 53, 66, 81, 100, 123, 152, 187, 231,
                285, 351, 433, 534, 658, 811, 1000, 1233,
                1520, 1874, 2310, 2848, 3511, 4329, 5337, 6579, 8111, 10000, 12328, 15199, 18738, 23101, 28480, 35112,
                43288, 53367, 65793, 81113, 100000, 123285, 151991, 187382, 231013, 284804, 351119, 432876, 533670,
                657933, 811131, 1000000, 1232847, 1519911, 1873817, 2310130, 2848036, 3511192, 4328761, 5336699,
                6579332, 8111308, 10000000, 12328467, 15199111, 18738174, 23101297, 28480359, 35111917, 43287613,
                53366992, 65793322, 81113083, 100000000, 123284674, 151991108, 187381742, 231012970, 284803587,
                351119173, 432876128, 533669923, 657933225, 811130831, 1000000000, 1232846739, 1519911083,
                1873817423, 2310129700l, 2848035868l, 3511191734l, 4328761281l, 5336699231l, 6579332247l, 8111308308l,
                10000000000l};

        static final long serialVersionUID = 42L;

        public QCDialog(Dataset dataset) {
            super(MainWindow.this);

            String text = dataset.getStatistics();
            String textDescription = null;
            String textStatistics = null;
            String graphs = dataset.getGraphs();
            JTextPane description = null;
            JTabbedPane tabbedPane = new JTabbedPane();
            HTMLEditorKit kit = new HTMLEditorKit();


            StyleSheet styleSheet = kit.getStyleSheet();
            styleSheet.addRule("table { border-collapse: collapse;}");
            styleSheet.addRule("body {font-family: Sans-Serif; font-size: 12;}");
            styleSheet.addRule("td { padding: 2px; }");
            styleSheet.addRule("th {border-bottom: 1px solid #000; text-align: left; background-color: #D8D8D8; font-weight: normal;}");


            if (text.contains("Protocol")) {
                int split = text.indexOf("</table>") + 8;
                textDescription = text.substring(0,split);
                textStatistics = text.substring(split);
                description = new JTextPane();
                description.setEditable(false);
                description.setContentType("text/html");
                description.setEditorKit(kit);
                description.setText(textDescription);
                tabbedPane.addTab("About Library", description);
            }
            else {
                textStatistics = text;
            }

            JTextPane textPane = new JTextPane();
            textPane.setEditable(false);
            textPane.setContentType("text/html");

            textPane.setEditorKit(kit);
            textPane.setText(textStatistics);
            JScrollPane pane = new JScrollPane(textPane);
            tabbedPane.addTab("Statistics", pane);


            if (graphs != null) {

                long[] A = new long[2000];
                long sumA = 0;
                long[] mapq1 = new long[201];
                long[] mapq2 = new long[201];
                long[] mapq3 = new long[201];
                long[] intraCount = new long[100];
                final XYSeries intra = new XYSeries("Intra Count");
                final XYSeries leftRead = new XYSeries("Left");
                final XYSeries rightRead = new XYSeries("Right");
                final XYSeries innerRead = new XYSeries("Inner");
                final XYSeries outerRead = new XYSeries("Outer");
                final XYSeries allMapq = new XYSeries("All MapQ");
                final XYSeries intraMapq = new XYSeries("Intra MapQ");
                final XYSeries interMapq = new XYSeries("Inter MapQ");

                Scanner scanner = new Scanner(graphs);
                try {
                    while (!scanner.next().equals("[")) ;

                    for (int idx = 0; idx < 2000; idx++) {
                        A[idx] = scanner.nextLong();
                        sumA += A[idx];
                    }

                    while (!scanner.next().equals("[")) ;
                    for (int idx = 0; idx < 201; idx++) {
                        mapq1[idx] = scanner.nextInt();
                        mapq2[idx] = scanner.nextInt();
                        mapq3[idx] = scanner.nextInt();

                    }

                    for (int idx = 199; idx >= 0; idx--) {
                        mapq1[idx] = mapq1[idx] + mapq1[idx + 1];
                        mapq2[idx] = mapq2[idx] + mapq2[idx + 1];
                        mapq3[idx] = mapq3[idx] + mapq3[idx + 1];
                        allMapq.add(idx, mapq1[idx]);
                        intraMapq.add(idx, mapq2[idx]);
                        interMapq.add(idx, mapq3[idx]);
                    }
                    while (!scanner.next().equals("[")) ;
                    for (int idx = 0; idx < 100; idx++) {
                        int tmp = scanner.nextInt();
                        if (tmp != 0) innerRead.add(logXAxis[idx], tmp);
                        intraCount[idx] = tmp;
                        tmp = scanner.nextInt();
                        if (tmp != 0) outerRead.add(logXAxis[idx], tmp);
                        intraCount[idx] += tmp;
                        tmp = scanner.nextInt();
                        if (tmp != 0) rightRead.add(logXAxis[idx], tmp);
                        intraCount[idx] += tmp;
                        tmp = scanner.nextInt();
                        if (tmp != 0) leftRead.add(logXAxis[idx], tmp);
                        intraCount[idx] += tmp;
                        if (idx > 0) intraCount[idx] += intraCount[idx - 1];
                        if (intraCount[idx] != 0) intra.add(logXAxis[idx], intraCount[idx]);
                    }
                } catch (NoSuchElementException exception) {
                    JOptionPane.showMessageDialog(getParent(), "Graphing file improperly formatted", "Error", JOptionPane.ERROR_MESSAGE);
                    dispose();
                    return;
                }

                final XYSeriesCollection readTypeCollection = new XYSeriesCollection();
                readTypeCollection.addSeries(innerRead);
                readTypeCollection.addSeries(outerRead);
                readTypeCollection.addSeries(leftRead);
                readTypeCollection.addSeries(rightRead);

                final JFreeChart readTypeChart = ChartFactory.createXYLineChart(
                        "Types of reads vs distance",          // chart title
                        "Distance (log)",               // domain axis label
                        "Binned Reads (log)",                  // range axis label
                        readTypeCollection,                  // data
                        PlotOrientation.VERTICAL,
                        true,                     // include legend
                        true,
                        false
                );

                final XYPlot readTypePlot = readTypeChart.getXYPlot();

                readTypePlot.setDomainAxis(new LogarithmicAxis("Distance (log)"));
                readTypePlot.setRangeAxis(new LogarithmicAxis("Binned Reads (log)"));
                readTypePlot.setBackgroundPaint(Color.white);
                readTypePlot.setRangeGridlinePaint(Color.lightGray);
                readTypePlot.setDomainGridlinePaint(Color.lightGray);
                readTypeChart.setBackgroundPaint(Color.white);
                readTypePlot.setOutlinePaint(Color.black);
                final ChartPanel chartPanel = new ChartPanel(readTypeChart);

                final XYSeriesCollection reCollection = new XYSeriesCollection();
                final XYSeries reDistance = new XYSeries("Distance");

                for (int i = 0; i < A.length; i++) {
                    if (A[i] != 0) reDistance.add(i, A[i] / (float) sumA);
                }
                reCollection.addSeries(reDistance);

                final JFreeChart reChart = ChartFactory.createXYLineChart(
                        "Distance from closest RE site",          // chart title
                        "Distance (bp)",               // domain axis label
                        "Fraction of Reads (log)",                  // range axis label
                        reCollection,                  // data
                        PlotOrientation.VERTICAL,
                        true,                     // include legend
                        true,
                        false
                );

                final XYPlot rePlot = reChart.getXYPlot();
                rePlot.setDomainAxis(new NumberAxis("Distance (bp)"));
                rePlot.setRangeAxis(new LogarithmicAxis("Fraction of Reads (log)"));
                rePlot.setBackgroundPaint(Color.white);
                rePlot.setRangeGridlinePaint(Color.lightGray);
                rePlot.setDomainGridlinePaint(Color.lightGray);
                reChart.setBackgroundPaint(Color.white);
                rePlot.setOutlinePaint(Color.black);
                final ChartPanel chartPanel2 = new ChartPanel(reChart);

                final XYSeriesCollection intraCollection = new XYSeriesCollection();

                intraCollection.addSeries(intra);

                final JFreeChart intraChart = ChartFactory.createXYLineChart(
                        "Intra reads vs distance",          // chart title
                        "Distance (log)",               // domain axis label
                        "Cumulative Sum of Binned Reads (log)",                  // range axis label
                        intraCollection,                  // data
                        PlotOrientation.VERTICAL,
                        true,                     // include legend
                        true,
                        false
                );

                final XYPlot intraPlot = intraChart.getXYPlot();
                intraPlot.setDomainAxis(new LogarithmicAxis("Distance (log)"));
                intraPlot.setRangeAxis(new NumberAxis("Cumulative Sum of Binned Reads (log)"));
                intraPlot.setBackgroundPaint(Color.white);
                intraPlot.setRangeGridlinePaint(Color.lightGray);
                intraPlot.setDomainGridlinePaint(Color.lightGray);
                intraChart.setBackgroundPaint(Color.white);
                intraPlot.setOutlinePaint(Color.black);
                final ChartPanel chartPanel3 = new ChartPanel(intraChart);

                final XYSeriesCollection mapqCollection = new XYSeriesCollection();
                mapqCollection.addSeries(allMapq);
                mapqCollection.addSeries(intraMapq);
                mapqCollection.addSeries(interMapq);

                final JFreeChart mapqChart = ChartFactory.createXYLineChart(
                        "MapQ Threshold Count",          // chart title
                        "MapQ threshold",               // domain axis label
                        "Count",                  // range axis label
                        mapqCollection,                  // data
                        PlotOrientation.VERTICAL,
                        true,                     // include legend
                        true,                     // include tooltips
                        false
                );

                final XYPlot mapqPlot = mapqChart.getXYPlot();
                mapqPlot.setBackgroundPaint(Color.white);
                mapqPlot.setRangeGridlinePaint(Color.lightGray);
                mapqPlot.setDomainGridlinePaint(Color.lightGray);
                mapqChart.setBackgroundPaint(Color.white);
                mapqPlot.setOutlinePaint(Color.black);
                final ChartPanel chartPanel4 = new ChartPanel(mapqChart);


                tabbedPane.addTab("Pair Type", chartPanel);
                tabbedPane.addTab("Restriction", chartPanel2);
                tabbedPane.addTab("Intra vs Distance", chartPanel3);
                tabbedPane.addTab("MapQ", chartPanel4);

            }

            final ExpectedValueFunction df = hic.getDataset().getExpectedValues(hic.getZoom(),
                    hic.getNormalizationType());
            if (df != null) {
                double[] expected = df.getExpectedValues();
                final XYSeriesCollection collection = new XYSeriesCollection();
                final XYSeries expectedValues = new XYSeries("Expected");
                for (int i = 0; i < expected.length; i++) {
                    if (expected[i] > 0) expectedValues.add(i + 1, expected[i]);
                }
                collection.addSeries(expectedValues);
                String title = "Expected at " + hic.getZoom() + " norm " + hic.getNormalizationType();
                final JFreeChart readTypeChart = ChartFactory.createXYLineChart(
                        title,          // chart title
                        "Distance between reads (log)",               // domain axis label
                        "Genome-wide expected (log)",                  // range axis label
                        collection,                  // data
                        PlotOrientation.VERTICAL,
                        false,                     // include legend
                        true,
                        false
                );
                final XYPlot readTypePlot = readTypeChart.getXYPlot();

                readTypePlot.setDomainAxis(new LogarithmicAxis("Distance between reads (log)"));
                readTypePlot.setRangeAxis(new LogarithmicAxis("Genome-wide expected (log)"));
                readTypePlot.setBackgroundPaint(Color.white);
                readTypePlot.setRangeGridlinePaint(Color.lightGray);
                readTypePlot.setDomainGridlinePaint(Color.lightGray);
                readTypeChart.setBackgroundPaint(Color.white);
                readTypePlot.setOutlinePaint(Color.black);
                final ChartPanel chartPanel5 = new ChartPanel(readTypeChart);

                tabbedPane.addTab("Expected", chartPanel5);
            }

            getContentPane().add(tabbedPane);
            pack();
            setModal(false);
            setLocation(100, 100);

        }
    }

    private class DumpDialog extends JFileChooser {
        JComboBox<String> box;

        static final long serialVersionUID = 42L;
        public DumpDialog() {
            super();
            int result = showSaveDialog(MainWindow.this);
            if (result == JFileChooser.APPROVE_OPTION) {
                try {
                    if (box.getSelectedItem().equals("Matrix")) {
                        if (hic.getDisplayOption() == MatrixType.OBSERVED) {
                            double[] nv1 = null;
                            double[] nv2 = null;
                            if (!(hic.getNormalizationType() == NormalizationType.NONE)) {
                                NormalizationVector nv = hic.getNormalizationVector(hic.getZd().getChr1Idx());
                                nv1 = nv.getData();
                                if (hic.getZd().getChr1Idx() != hic.getZd().getChr2Idx()) {
                                    nv = hic.getNormalizationVector(hic.getZd().getChr2Idx());
                                    nv2 = nv.getData();
                                } else {
                                    nv2 = nv1;
                                }
                            }
                            hic.getZd().dump(new PrintWriter(getSelectedFile()), nv1, nv2);

                        } else if (hic.getDisplayOption() == MatrixType.OE || hic.getDisplayOption() == MatrixType.PEARSON) {
                            final ExpectedValueFunction df = hic.getDataset().getExpectedValues(hic.getZd().getZoom(),
                                    hic.getNormalizationType());
                            if (df == null) {
                                JOptionPane.showMessageDialog(this, box.getSelectedItem() + " not available", "Error",
                                        JOptionPane.ERROR_MESSAGE);
                                return;
                            }
                            if (hic.getDisplayOption() == MatrixType.OE) {
                                hic.getZd().dumpOE(df, "oe",
                                        hic.getNormalizationType(), null, new PrintWriter(getSelectedFile()));
                            } else {
                                hic.getZd().dumpOE(df, "pearson",
                                        hic.getNormalizationType(), null, new PrintWriter(getSelectedFile()));
                            }
                        }

                    } else if (box.getSelectedItem().equals("Norm vector")) {

                        if (hic.getNormalizationType() == NormalizationType.NONE) {
                            JOptionPane.showMessageDialog(this, "Selected normalization is None, nothing to write",
                                    "Error", JOptionPane.ERROR_MESSAGE);
                        } else {
                            NormalizationVector nv = hic.getNormalizationVector(hic.getZd().getChr1Idx());
                            HiCTools.dumpVector(new PrintWriter(getSelectedFile()), nv.getData(), false);
                        }
                    } else if (box.getSelectedItem().toString().contains("Expected")) {

                        final ExpectedValueFunction df = hic.getDataset().getExpectedValues(hic.getZd().getZoom(),
                                hic.getNormalizationType());
                        if (df == null) {
                            JOptionPane.showMessageDialog(this, box.getSelectedItem() + " not available", "Error",
                                    JOptionPane.ERROR_MESSAGE);
                            return;
                        }

                        if (box.getSelectedItem().equals("Expected vector")) {
                            int length = df.getLength();
                            int c = hic.getZd().getChr1Idx();
                            PrintWriter pw = new PrintWriter(getSelectedFile());
                            for (int i = 0; i < length; i++) {
                                pw.println((float) df.getExpectedValue(c, i));
                            }
                            pw.flush();
                        } else {
                            HiCTools.dumpVector(new PrintWriter(getSelectedFile()), df.getExpectedValues(), false);
                        }
                    } else if (box.getSelectedItem().equals("Eigenvector")) {
                        int chrIdx = hic.getZd().getChr1Idx();
                        double[] eigenvector = hic.getEigenvector(chrIdx, 0);

                        if (eigenvector != null) {
                            HiCTools.dumpVector(new PrintWriter(getSelectedFile()), eigenvector, true);
                        }
                    }
                } catch (IOException error) {
                    JOptionPane.showMessageDialog(this, "Error while writing:\n" + error, "Error", JOptionPane.ERROR_MESSAGE);
                }
            }
        }

        protected JDialog createDialog(Component component) throws HeadlessException {
            JDialog dialog = super.createDialog(component);
            JPanel panel1 = new JPanel();
            JLabel label = new JLabel("Dump ");
            box = new JComboBox<String>(new String[]{"Matrix", "Norm vector", "Expected vector", "Expected genome-wide vector", "Eigenvector"});
            panel1.add(label);
            panel1.add(box);
            dialog.add(panel1, BorderLayout.NORTH);
            setCurrentDirectory(DirectoryManager.getUserDirectory());
            setDialogTitle("Choose location for dump of matrix or vector");
            setFileSelectionMode(JFileChooser.FILES_ONLY);
            return dialog;
        }

    }

    private class SaveImageDialog extends JFileChooser {
        JTextField width;
        JTextField height;

        static final long serialVersionUID = 42L;
        public SaveImageDialog() {
            super();
            if (saveImagePath != null) {
                setSelectedFile(new File(saveImagePath));
            }
            else {
                setSelectedFile(new File("image.png"));
            }
            int actionDialog = showSaveDialog(MainWindow.getInstance());
            if (actionDialog == JFileChooser.APPROVE_OPTION) {
                File file = getSelectedFile();
                saveImagePath = file.getPath();
                if (file.exists()) {
                    actionDialog = JOptionPane.showConfirmDialog(MainWindow.getInstance(), "Replace existing file?");
                    if (actionDialog == JOptionPane.NO_OPTION || actionDialog == JOptionPane.CANCEL_OPTION)
                        return;
                }
                try {
                    int w = Integer.valueOf(width.getText());
                    int h = Integer.valueOf(height.getText());
                    saveImage(file, w, h);
                } catch (IOException error) {
                    JOptionPane.showMessageDialog(MainWindow.getInstance(), "Error while saving file:\n" + error, "Error",
                            JOptionPane.ERROR_MESSAGE);
                } catch (NumberFormatException error) {
                    JOptionPane.showMessageDialog(MainWindow.getInstance(), "Width and Height must be integers", "Error",
                            JOptionPane.ERROR_MESSAGE);
                }
            }
        }

        protected JDialog createDialog(Component parent) {
            JDialog myDialog = super.createDialog(parent);
            JLabel wLabel = new JLabel("Width");
            JLabel hLabel = new JLabel("Height");
            width = new JTextField("" + MainWindow.getInstance().getWidth());
            width.setColumns(6);
            height = new JTextField("" + MainWindow.getInstance().getHeight());
            height.setColumns(6);
            JPanel panel = new JPanel();
            panel.add(wLabel);
            panel.add(width);
            panel.add(hLabel);
            panel.add(height);
            myDialog.add(panel, BorderLayout.NORTH);
            return myDialog;
        }

        private void saveImage(File file, final int w, final int h) throws IOException {

            // default if they give no format or invalid format
            String fmt = "jpg";
            int ind = file.getName().indexOf(".");
            if (ind != -1) {
                String ext = file.getName().substring(ind + 1);
                String[] strs = ImageIO.getWriterFormatNames();
                for (String aStr : strs)
                    if (ext.equals(aStr))
                        fmt = ext;
            }
            BufferedImage image = (BufferedImage) MainWindow.getInstance().createImage(w, h);
            Graphics g = image.createGraphics();

            Dimension size = MainWindow.getInstance().getSize();

            if (w == MainWindow.getInstance().getWidth() && h == MainWindow.getInstance().getHeight()) {
                hiCPanel.paint(g);
            } else {
                JDialog waitDialog = new JDialog();
                JPanel panel1 = new JPanel();
                panel1.add(new JLabel("  Creating and saving " + w + " by " + h + " image  "));
                //panel1.setPreferredSize(new Dimension(250,50));
                waitDialog.add(panel1);
                waitDialog.setTitle("Please wait...");
                waitDialog.pack();
                waitDialog.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);

                waitDialog.setLocation(100, 100);
                waitDialog.setVisible(true);
                MainWindow.getInstance().setVisible(false);

                Dimension minSize = MainWindow.getInstance().getMinimumSize();
                Dimension prefSize = MainWindow.getInstance().getPreferredSize();

                hic.centerBP(0, 0);
                MainWindow.getInstance().setMinimumSize(new Dimension(w, h));
                MainWindow.getInstance().setPreferredSize(new Dimension(w, h));
                MainWindow.getInstance().pack();

                MainWindow.getInstance().setState(Frame.ICONIFIED);
                MainWindow.getInstance().setState(Frame.NORMAL);
                MainWindow.getInstance().setVisible(true);
                MainWindow.getInstance().setVisible(false);

                final Runnable painter = new Runnable() {
                    public void run() {
                        hiCPanel.paintImmediately(0, 0, w, h);
                    }
                };

                Thread thread = new Thread(painter) {
                    public void run() {

                        try {
                            SwingUtilities.invokeAndWait(painter);
                        } catch (Exception e) {
                            e.printStackTrace();
                        }

                    }
                };

                thread.start();

                hiCPanel.paint(g);
                MainWindow.getInstance().setPreferredSize(prefSize);
                MainWindow.getInstance().setMinimumSize(minSize);
                MainWindow.getInstance().setSize(size);
                waitDialog.setVisible(false);
                waitDialog.dispose();
                MainWindow.getInstance().setVisible(true);
            }

            ImageIO.write(image.getSubimage(0, 0, w, h), fmt, file);
            g.dispose();
        }

    }

    private class LoadDialog extends JDialog implements TreeSelectionListener, ActionListener {

        private JTree tree;
        private JButton cancelButton;
        private JSplitButton openButton;
        private JSplitButton localButton;
        private JMenuItem openURL;
        private JMenuItem open30;
        private boolean success;
        private boolean control;
        static final long serialVersionUID = 42L;
        public LoadDialog(Properties properties) {
            super(MainWindow.this, "Select file(s) to open");

            //Create the nodes.
            DefaultMutableTreeNode top =
                    new DefaultMutableTreeNode(new ItemInfo("root", "root", ""));
            if (!createNodes(top, properties)) {
                dispose();
                success = false;
                return;
            }

            //Create a tree that allows one selection at a time.
            tree = new JTree(top);
            tree.getSelectionModel().setSelectionMode(TreeSelectionModel.DISCONTIGUOUS_TREE_SELECTION);

            //Listen for when the selection changes.
            tree.addTreeSelectionListener(this);
            tree.setRootVisible(false);
            tree.addMouseListener(new MouseAdapter() {
                @Override
                public void mousePressed(MouseEvent mouseEvent) {
                    TreePath selPath = tree.getPathForLocation(mouseEvent.getX(), mouseEvent.getY());
                    if (selPath != null) {
                        if (mouseEvent.getClickCount() == 2) {
                            DefaultMutableTreeNode node = (DefaultMutableTreeNode) selPath.getLastPathComponent();
                            if (node != null && node.isLeaf()) {
                                TreePath[] paths = new TreePath[1];
                                paths[0] = selPath;
                                loadFiles(paths, null);
                            }

                        }
                    }
                }
            });

            //Create the scroll pane and add the tree to it.
            JScrollPane treeView = new JScrollPane(tree);
            treeView.setPreferredSize(new Dimension(400, 400));
            JPanel centerPanel = new JPanel();
            centerPanel.add(treeView);
            add(centerPanel, BorderLayout.CENTER);

            JPanel buttonPanel = new JPanel();

            openButton = new JSplitButton("Open MAPQ > 0");
            openButton.addActionListener(this);
            openButton.setEnabled(false);

            JPopupMenu popupMenu = new JPopupMenu("Popup Menu");
            open30 = new JMenuItem("Open MAPQ \u2265 30");
            open30.addActionListener(this);
            popupMenu.add(open30);
            openButton.setComponentPopupMenu(popupMenu);

            localButton = new JSplitButton("Load Local...");
            localButton.addActionListener(this);

            JPopupMenu popupMenu1 = new JPopupMenu("Popup1");
            openURL = new JMenuItem("Load URL...");
            openURL.addActionListener(this);
            popupMenu1.add(openURL);
            localButton.setComponentPopupMenu(popupMenu1);

            cancelButton = new JButton("Cancel");
            cancelButton.addActionListener(this);
            cancelButton.setPreferredSize(new Dimension((int)cancelButton.getPreferredSize().getWidth(), (int)openButton.getPreferredSize().getHeight()));

            buttonPanel.add(openButton);
            if (!isRestricted) {
                buttonPanel.add(localButton);
            }
            buttonPanel.add(cancelButton);

            add(buttonPanel, BorderLayout.SOUTH);
            Dimension minimumSize = new Dimension(400, 400);
            setMinimumSize(minimumSize);
            setLocation(100, 100);
            pack();
            success = true;
        }

        private void setControl(boolean control) {
            this.control = control;
        }

        public boolean getSuccess() {
            return success;
        }

        private boolean createNodes(DefaultMutableTreeNode top, Properties properties) {
           // Enumeration<DefaultMutableTreeNode> enumeration = top.breadthFirstEnumeration();
            // TreeSet is sorted, so properties file is implemented in order
            TreeSet<String> keys = new TreeSet<String>(properties.stringPropertyNames());
            HashMap<String, DefaultMutableTreeNode> hashMap = new HashMap<String, DefaultMutableTreeNode>();
            hashMap.put(((ItemInfo)top.getUserObject()).uid, top);

            for (String key : keys) {
                String value = properties.getProperty(key);
                DefaultMutableTreeNode node;
                final String[] values = value.split(",");
                if (values.length != 3 && values.length != 2) {
                    JOptionPane.showMessageDialog(this, "Improperly formatted properties file; incorrect # of fields", "Error", JOptionPane.ERROR_MESSAGE);
                    return false;
                }
                if (values.length == 2) {
                    node = new DefaultMutableTreeNode(new ItemInfo(key, values[0], values[1]));
                } else {
                    node = new DefaultMutableTreeNode(new ItemInfo(key, values[0], values[1], values[2]));
                }
                hashMap.put(key, node);
            }
            for (String key : keys) {
                DefaultMutableTreeNode node = hashMap.get(key);
                DefaultMutableTreeNode parent = hashMap.get(((ItemInfo)node.getUserObject()).parentKey);

                if (parent == null) {
                    JOptionPane.showMessageDialog(this, "Improperly formatted properties file; unable to find parent menu "
                            + ((ItemInfo)hashMap.get(key).getUserObject()).parentKey + " for " +
                            key, "Error", JOptionPane.ERROR_MESSAGE);
                    return false;
                }
                else {
                    parent.add(node);
                }
            }
            return true;
        }

        /**
         * Required by TreeSelectionListener interface.
         */
        public void valueChanged(TreeSelectionEvent e) {
            DefaultMutableTreeNode node = (DefaultMutableTreeNode)
                    tree.getLastSelectedPathComponent();

            if (node == null) return;

            if (node.isLeaf()) {
                openButton.setEnabled(true);
                open30.setEnabled(true);
            }
            else {
                openButton.setEnabled(false);
                open30.setEnabled(false);
            }
        }

        public void actionPerformed(ActionEvent e) {
            if (e.getSource() == openButton) {
                loadFiles(tree.getSelectionPaths(), null);
            }
            else if (e.getSource() == open30) {
                loadFiles(tree.getSelectionPaths(), "30");
            }
            if (e.getSource() == localButton) {
                loadMenuItemActionPerformed(control);
                setVisible(false);
            }
            else if (e.getSource() == openURL) {
                loadFromURLActionPerformed(control);
                setVisible(false);
            }
            else if (e.getSource() == cancelButton) {
                setVisible(false);
                dispose();
            }
        }
        private void loadFiles(String path, String title, boolean control) {
            List<String> paths = new ArrayList<String>();
            paths.add(path);
            load(paths, control);

            if (control) controlTitle = title;
            else datasetTitle = title;
            updateTitle();
        }
        private void loadFiles(TreePath[] paths, String ext) {
            ArrayList<ItemInfo> filesToLoad = new ArrayList<ItemInfo>();
            String title = "";

            for (TreePath path : paths) {
                DefaultMutableTreeNode node = (DefaultMutableTreeNode) path.getLastPathComponent();
                if (node != null && node.isLeaf()) {
                    filesToLoad.add((ItemInfo) node.getUserObject());
                    title += path.toString().replace("[", "").replace("]", "").replace(",", "");
                    if (ext != null) title += " MAPQ \u2265 " + ext;
                }
            }


            setVisible(false);
            List<String> urls = new ArrayList<String>();
            for (ItemInfo info : filesToLoad) {
                if (info.itemURL == null || !info.itemURL.endsWith(".hic")) {
                    JOptionPane.showMessageDialog(this, info.itemName + " is not a hic file, or the path to the file is not specified.");
                    continue;
                }
                String toadd = info.itemURL;
                if (ext != null) {
                    toadd = toadd.replace(".hic", "_" + ext + ".hic");
                }
                urls.add(toadd);
            }

            //code to add a recent file to the menu
            recentMenu.addEntry(title.trim()+"@@"+urls.get(0), true);
            load(urls, control);

            if (control) controlTitle = title;
            else datasetTitle = title;
            updateTitle();


        }


        private class ItemInfo {
            public String uid;
            public String itemName;
            public String itemURL;
            public String parentKey;

            public ItemInfo(String uid, String parentKey, String itemName, String itemURL) {
                this.uid = uid;
                this.parentKey = parentKey;
                this.itemName = itemName.trim();
                this.itemURL = itemURL.trim();
            }

            public ItemInfo(String uid, String parentKey, String itemName) {
                this.parentKey = parentKey;
                this.itemName = itemName;
                this.uid = uid;
            }

            public String toString() {
                return itemName;
            }

        }
    }

    private class ResolutionControl extends JPanel {
        static final long serialVersionUID = 42L;
        private final ImageIcon lockOpenIcon;
        private final ImageIcon lockIcon;

        JSlider resolutionSlider;
        JideButton lockButton;
        final JLabel resolutionLabel;
        Map<Integer, HiCZoom> idxZoomMap = new HashMap<Integer, HiCZoom>();
        Map<Integer, String> bpLabelMap;
        private int lastValue = 0;
        HiC.Unit unit = HiC.Unit.BP;

        {
            bpLabelMap = new Hashtable<Integer, String>();
            bpLabelMap.put(2500000, "2.5 MB");
            bpLabelMap.put(1000000, "1 MB");
            bpLabelMap.put(500000, "500 KB");
            bpLabelMap.put(250000, "250 KB");
            bpLabelMap.put(100000, "100 KB");
            bpLabelMap.put(50000, "50 KB");
            bpLabelMap.put(25000, "25 KB");
            bpLabelMap.put(10000, "10 KB");
            bpLabelMap.put(5000, "5 KB");

        }

        private String getUnitLabel() {
            return unit == HiC.Unit.FRAG ? "Resolution (Frag)" : "Resolution (BP)";
        }

        public void setEnabled(boolean enabled) {
            resolutionSlider.setEnabled(enabled);
            lockButton.setEnabled(enabled);
        }

        private ResolutionControl() {

            this.setBorder(LineBorder.createGrayLineBorder());
            this.setLayout(new BorderLayout());

            resolutionLabel = new JLabel(getUnitLabel());
            resolutionLabel.setHorizontalAlignment(SwingConstants.CENTER);
            resolutionLabel.setBackground(new Color(204, 204, 204));

            JPanel resolutionLabelPanel = new JPanel();
            resolutionLabelPanel.setBackground(new Color(204, 204, 204));
            resolutionLabelPanel.setLayout(new BorderLayout());
            resolutionLabelPanel.add(resolutionLabel, BorderLayout.CENTER);
            resolutionLabelPanel.addMouseListener(new MouseAdapter() {
                @Override
                public void mousePressed(MouseEvent e) {
                    if (resolutionSlider.isEnabled() && hic != null && hic.getDataset() != null) {
                        if (hic.getDataset().hasFrags()) {
                            unit = (unit == HiC.Unit.FRAG ? HiC.Unit.BP : HiC.Unit.FRAG);
                            resolutionLabel.setText(getUnitLabel());
                            reset();
                            refresh();
                        }
                    }
                }
            });


            this.add(resolutionLabelPanel, BorderLayout.PAGE_START);


            JPanel resolutionButtonPanel = new JPanel();
            resolutionButtonPanel.setLayout(new BoxLayout(resolutionButtonPanel, BoxLayout.X_AXIS));

            //---- resolutionSlider ----
            JPanel sliderPanel = new JPanel();
            sliderPanel.setLayout(new BoxLayout(sliderPanel, BoxLayout.X_AXIS));
            resolutionSlider = new JSlider();
            sliderPanel.add(resolutionSlider);

            lockButton = new JideButton();

            lockIcon = new ImageIcon(getClass().getResource("/images/lock.png"));
            lockOpenIcon = new ImageIcon(getClass().getResource("/images/lock_open.png"));
            resolutionLocked = false;
            lockButton.setIcon(lockOpenIcon);
            lockButton.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    resolutionLocked = !resolutionLocked;
                    lockButton.setIcon(resolutionLocked ? lockIcon : lockOpenIcon);
                }
            });
            sliderPanel.add(lockButton);


            resolutionButtonPanel.add(sliderPanel);
            this.add(resolutionButtonPanel, BorderLayout.CENTER);


            // Setting the zoom should always be done by calling resolutionSlider.setValue() so work isn't done twice.
            resolutionSlider.addChangeListener(new ChangeListener() {
                // Change zoom level while staying centered on current location.
                // Centering is relative to the bounds of the data, which might not be the bounds of the window

                public void stateChanged(ChangeEvent e) {

                    if (hic == null || hic.getMatrix() == null || hic.getZd() == null) return;

                    if (!resolutionSlider.getValueIsAdjusting()) {

                        int idx = resolutionSlider.getValue();

                        HiCZoom zoom = idxZoomMap.get(idx);
                        if (zoom == null) return;

                        if (hic.getXContext() != null) {

                            hic.setScaleFactor(1.0);
                            hic.setScaleFactor(1.0);

                            double centerBinX = hic.getXContext().getBinOrigin() + (heatmapPanel.getWidth() / (2 * hic.getScaleFactor()));
                            double centerBinY = hic.getYContext().getBinOrigin() + (heatmapPanel.getHeight() / (2 * hic.getScaleFactor()));
                            int xGenome = hic.getZd().getXGridAxis().getGenomicMid(centerBinX);
                            int yGenome = hic.getZd().getYGridAxis().getGenomicMid(centerBinY);

                            if (hic.getZd() == null) {
                                hic.setZoom(zoom, 0, 0);
                            } else {

                                if (hic.setZoom(zoom, xGenome, yGenome)) {
                                    lastValue = resolutionSlider.getValue();
                                } else {
                                    resolutionSlider.setValue(lastValue);
                                }
                            }
                        }

                    }
                }
            });
            setEnabled(false);
        }

        /**
         * Called when a new dataset is loaded, or when units are switched bp<->frag
         */
        private void reset() {
            if (hic == null || hic.getDataset() == null) return;

            heatmapPanel.reset();

            int currentIdx = resolutionSlider.getValue();

            List<HiCZoom> binSizes =
                    unit == HiC.Unit.BP ? hic.getDataset().getBpZooms() : hic.getDataset().getFragZooms();
            idxZoomMap.clear();
            for (int i = 0; i < binSizes.size(); i++) {
                HiCZoom zoom = binSizes.get(i);
                idxZoomMap.put(i, zoom);
            }

            int maxIdx = binSizes.size() - 1;
            resolutionSlider.setMaximum(maxIdx);
            resolutionSlider.setMajorTickSpacing(1);
            resolutionSlider.setPaintTicks(true);
            resolutionSlider.setSnapToTicks(true);
            resolutionSlider.setPaintLabels(true);
            resolutionSlider.setMinorTickSpacing(1);

            // Create labels
            Dictionary<Integer, JLabel> resolutionLabels = new Hashtable<Integer, JLabel>();
            Font f = FontManager.getFont(8);
            int skip = maxIdx > 6 ? 2 : 1;   // Skip every other if more than 6 levels
            for (int i = 0; i <= maxIdx; i++) {
                if (i % skip == 0) {
                    String label = sizeToLabel(binSizes.get(i).getBinSize());
                    final JLabel tickLabel = new JLabel(label);
                    tickLabel.setFont(f);
                    resolutionLabels.put(i, tickLabel);
                }
                HiCZoom zoom = binSizes.get(i);
                idxZoomMap.put(i, zoom);
            }

            resolutionSlider.setLabelTable(resolutionLabels);

            // Really we should find the closest matching resolution
            int newIdx = Math.min(currentIdx, maxIdx);
            HiCZoom newZoom = idxZoomMap.get(newIdx);
            setZoom(newZoom);

        }


        String sizeToLabel(int binSize) {

            if (unit == HiC.Unit.FRAG) {
                return binSize + " f";
            }

            if (bpLabelMap.containsKey(binSize)) {
                return bpLabelMap.get(binSize);
            }

            if (binSize >= 1000000) {
                return ((float) binSize / 1000000) + " MB";
            } else if (binSize >= 1000) {
                return ((float) binSize / 1000) + " KB";
            } else {
                return binSize + " BP";
            }

        }


        public void setZoom(HiCZoom newZoom) {
            unit = newZoom.getUnit();
            resolutionLabel.setText(getUnitLabel());
            for (Map.Entry<Integer, HiCZoom> entry : idxZoomMap.entrySet()) {
                if (entry.getValue().equals(newZoom)) {
                    resolutionSlider.setValue(entry.getKey());
                }
            }
        }
    }

    public class HiCKeyDispatcher implements KeyEventDispatcher {

        @Override
        public boolean dispatchKeyEvent(KeyEvent e) {

            if (e.getID() == KeyEvent.KEY_PRESSED && e.getKeyCode() == KeyEvent.VK_F1) {

                if (hic.getControlZd() != null) {
                    MatrixType displayOption = (MatrixType) displayOptionComboBox.getSelectedItem();
                    if (displayOption == MainWindow.MatrixType.CONTROL) {
                        displayOptionComboBox.setSelectedItem(MainWindow.MatrixType.OBSERVED);

                    } else if (displayOption == MainWindow.MatrixType.OBSERVED) {
                        displayOptionComboBox.setSelectedItem(MainWindow.MatrixType.CONTROL);
                    }

                }
                return true;
            } else {

                return false;
            }
        }
    }

    abstract class RecentMenu extends JMenu
    {
        final private static String HIC_RECENT = "hicRecent";
        private String defaultText = "";
        private String[] recentEntries;
        private int m_maxItems;
        private boolean b_isEnabled = false;
        private Preferences prefs = Preferences.userNodeForPackage(Globals.class);
        private List<String> m_items = new ArrayList<String>();

        public RecentMenu(int count) throws BackingStoreException {
            super();
            this.setText("Recent");
            this.setMnemonic('R');
            this.m_maxItems=count;
            //initialize default entries
            this.recentEntries=new String[count];
            for(int index=0;index<this.m_maxItems;index++){
                this.recentEntries[index]=defaultText;
            }

            // load recent positions from properties
            for (int i = 0; i < this.m_maxItems; i++)
            {
                String val = prefs.get(HIC_RECENT+i, "");
                if (!val.equals(""))
                {
                    addEntry(val,false);
                }
                else
                {
                    if (i ==0) {
                        // No items.
                        this.setEnabled(false);
                    }
                    break;
                }
            }
        }

        /**
         * Add new recent entry, update file and menu
         * @param savedMap url and title of map.
         * @param updateFile also save to file, Constructor call with false - no need to re-write.
         */
        private void addEntry(String savedMap,boolean updateFile){
            //check if this is disabled
            if(!this.isEnabled()){
                this.setEnabled(true);
            }

            //clear the existing items
            this.removeAll();

            //Add item, remove previous existing duplicate:
            m_items.remove(savedMap);
            m_items.add(0, savedMap);

            //Chop last item if list is over size:
            if (this.m_items.size() > this.m_maxItems)
            {
                this.m_items.remove(this.m_items.size() - 1);
            }

            //add items back to the menu
            for(int index=0;index<this.m_items.size();index++){
                JMenuItem menuItem=new JMenuItem();

                String delimiter = "@@";
                String[] temp;
                temp = this.m_items.get(index).split(delimiter);

                menuItem.setText(temp[0]);
                if(temp[0].equals(defaultText)){
                    menuItem.setVisible(false);
                } else{
                    menuItem.setVisible(true);
                    menuItem.setToolTipText(temp[0]);
                    menuItem.setActionCommand(this.m_items.get(index));
                    menuItem.addActionListener(new ActionListener(){
                        public void actionPerformed(ActionEvent actionEvent){
                            onSelectPosition(actionEvent.getActionCommand());
                        }
                    });
                }
                this.add(menuItem);
            }
            //update the file
            if(updateFile){
                try{
                    for (int i = 0; i < this.m_maxItems; i++)
                    {
                        if (i < this.m_items.size())
                        {
                            prefs.put(HIC_RECENT+i, this.m_items.get(i));
                        }
                        else
                        {
                            prefs.remove(HIC_RECENT+i);
                        }
                    }
                } catch(Exception x){
                    x.printStackTrace();
                }
            }
        }

        public boolean isEnabled() {
            return this.b_isEnabled;
        }
        public void setEnabled(boolean b_newState) {
            this.b_isEnabled = b_newState;
        }

        /**
         * Abstract event, fires when recent map is selected.
         * @param mapPath The file that was selected.
         */
        public abstract void onSelectPosition(String mapPath);
    }
    /**
     * A split button. The user can either click the text, which executes an
     * action, or click the icon, which opens a popup menu.
     *
     * @author eoogbe
     */
    public class JSplitButton extends AbstractButton implements Accessible {

        protected class AccessibleJSplitButton extends AccessibleAbstractButton {

            private static final long serialVersionUID = 1L;

            /* (non-Javadoc)
             * @see javax.swing.JComponent.
             * AccessibleJComponent#getAccessibleRole()
             */
            @Override
            public AccessibleRole getAccessibleRole() {
                return AccessibleRole.PUSH_BUTTON;
            }

        }

        private class PopupAction implements ActionListener {

            private JPopupMenu popupMenu;

            public PopupAction(JPopupMenu popupMenu) {
                this.popupMenu = popupMenu;
            }

            /* (non-Javadoc)
             * @see java.awt.event.ActionListener#actionPerformed(
             * java.awt.event.ActionEvent)
             */
            @Override
            public void actionPerformed(ActionEvent e) {
                Component comp = (Component) e.getSource();
                Point popupLocation = getPopupLocationRelativeTo(comp);
                popupMenu.show(comp, popupLocation.x, popupLocation.y);
            }

            public JPopupMenu getPopupMenu() {
                return popupMenu;
            }

        }

        private static final long serialVersionUID = 1L;

        private static final String uiClassID = "ButtonUI";

        private static final int DEFAULT_POPUP_ICON_LENGTH = 10;

        private static final String ALWAYS_SHOWS_POPUP_CHANGED_PROPERTY =
                "alwaysShowPopup";

        private static final String SPLIT_GAP_CHANGED_PROPERTY = "splitGap";

        private static final String POPUP_ICON_CHANGED_PROPERTY = "popupIcon";

        private static final String DISABLED_POPUP_ICON_CHANGED_PROPERTY =
                "disabledPopupIcon";

        private static final String
                DISABLED_SELECTED_POPUP_ICON_CHANGED_PROPERTY = "disabledSelectedPopupIcon";

        private static final String PRESSED_POPUP_ICON_CHANGED_PROPERTY =
                "pressedPopupIcon";

        private static final String ROLLOVER_POPUP_ICON_CHANGED_PROPERTY =
                "rolloverPopupIcon";

        private static final String
                ROLLOVER_SELECTED_POPUP_ICON_CHANGED_PROPERTY = "rolloverSelectedPopupIcon";

        private static final String SELECTED_POPUP_ICON_CHANGED_PROPERTY =
                "selectedPopupIcon";

        private static final String MAIN_TEXT_CHANGED_PROPERTY = "mainText";

        private PopupAction popupAction;

        private JButton mainButton, popupButton;

        private boolean alwaysShowPopup;

        private int splitGap = 5;

        /**
         * Creates a new JSplitButton with no set text.
         */
        public JSplitButton() {
            this("");
        }

        /**
         * Creates a new JSplitButton with initial text.
         *
         * @param text the text displayed on this JSplitButton
         */
        public JSplitButton(String text) {
            mainButton = new JButton(text);
            popupButton = new JButton(createDefaultPopupIcon());
         //   popupButton.setPreferredSize(new Dimension((int)popupButton.getPreferredSize().getWidth()+10,(int)popupButton.getPreferredSize().getHeight()));

            setModel(new DefaultButtonModel());

            mainButton.setBorder(BorderFactory.createEmptyBorder());
           // popupButton.setBorder(BorderFactory.createEmptyBorder());

            mainButton.setContentAreaFilled(false);

            mainButton.addActionListener(new ActionListener() {

                @Override
                public void actionPerformed(ActionEvent e) {
                    doClick();
                }

            });

            mainButton.addMouseListener(new MouseAdapter() {

                @Override
                public void mouseEntered(MouseEvent e) {
                    if (isRolloverEnabled()) model.setRollover(true);
                }

                @Override
                public void mouseExited(MouseEvent e) {
                    model.setRollover(false);
                }

            });

            setLayout(new BoxLayout(this, BoxLayout.LINE_AXIS));
            resetComponents();

            init(null, null);
        }

        /**
         * Creates a new JSplitButton with properties taken from the Action
         * supplied.
         *
         * @param action used to specify the properties of this JSplitButton
         */
        public JSplitButton(Action action) {
            this();
            setAction(action);
        }

        private Icon createDefaultPopupIcon() {
            Image image = new BufferedImage(DEFAULT_POPUP_ICON_LENGTH,
                    DEFAULT_POPUP_ICON_LENGTH, BufferedImage.TYPE_INT_ARGB);
            Graphics g = image.getGraphics();

            g.setColor(new Color(255, 255, 255, 0));
            g.fillRect(0, 0, DEFAULT_POPUP_ICON_LENGTH,
                    DEFAULT_POPUP_ICON_LENGTH);

            g.setColor(Color.BLACK);
            Polygon traingle = new Polygon(
                    new int[]{0, DEFAULT_POPUP_ICON_LENGTH,
                            DEFAULT_POPUP_ICON_LENGTH / 2, 0},
                    new int[]{0, 0, DEFAULT_POPUP_ICON_LENGTH, 0}, 4
            );
            g.fillPolygon(traingle);

            g.dispose();
            return new ImageIcon(image);
        }

        private void resetComponents() {
            removeAll();
            add(mainButton);
            add(Box.createRigidArea(new Dimension(splitGap, 0)));
            add(new JSeparator(VERTICAL));
            add(Box.createRigidArea(new Dimension(splitGap, 0)));
            add(popupButton);
            revalidate();
        }

        /**
         * Returns true if this JSplitButton shows the popup menu for every
         * click.
         *
         * @return true if this JSplitButton shows the popup menu for every
         * click
         * @see JSplitButton#setAlwaysShowPopup(boolean)
         */
        public boolean isAlwaysShowPopup() {
            return alwaysShowPopup;
        }

        /**
         * Sets whether this JSplitButton shows the popup menu for every click.
         *
         * @param alwaysShowPopup true if this JSplitButton shows the popup menu
         *                        for every click
         * @see JSplitButton#isAlwaysShowPopup()
         */
        public void setAlwaysShowPopup(boolean alwaysShowPopup) {
            boolean oldValue = this.alwaysShowPopup;
            this.alwaysShowPopup = alwaysShowPopup;
            firePropertyChange(ALWAYS_SHOWS_POPUP_CHANGED_PROPERTY, oldValue,
                    alwaysShowPopup);

            if (popupAction != null && oldValue != alwaysShowPopup) {
                setComponentPopupMenu(popupAction.getPopupMenu());
            }
        }

        /**
         * Returns the gap between the separator and the labels on each side
         *
         * @return the gap between the separator and the labels on each side
         * @see JSplitButton#setSplitGap(int)
         */
        public int getSplitGap() {
            return splitGap;
        }

        /**
         * Sets the gap between the separator and the labels on each side.
         *
         * @param splitGap the gap set
         * @see JSplitButton#getSplitGap()
         */
        public void setSplitGap(int splitGap) {
            int oldValue = this.splitGap;
            this.splitGap = splitGap;
            firePropertyChange(SPLIT_GAP_CHANGED_PROPERTY, oldValue, splitGap);

            if (oldValue != splitGap) {
                resetComponents();
            }
        }

        /**
         * Returns the Icon used on the popup side. This Icon is also used as the
         * "pressed" and "disabled" Icon if they are not explicitly set.
         *
         * @return the popupIcon property
         * @see AbstractButton#getIcon()
         * @see JSplitButton#setPopupIcon(Icon)
         */
        public Icon getPopupIcon() {
            return popupButton.getIcon();
        }

        /**
         * Sets the Icon used on the popup side. If null, the Icon will be set to
         * a default value. This Icon is also used as the "pressed" and "disabled"
         * Icon if they are not explicitly set.
         *
         * @param icon the Icon set
         * @see AbstractButton#setIcon(Icon)
         * @see JSplitButton#getPopupIcon()
         */
        public void setPopupIcon(Icon icon) {
            Icon oldValue = getPopupIcon();
            if (icon == null) icon = createDefaultPopupIcon();
            firePropertyChange(POPUP_ICON_CHANGED_PROPERTY, oldValue, icon);

            if (!oldValue.equals(icon)) {
                popupButton.setIcon(icon);
            }
        }

        /**
         * Returns the icon used on the popup side when this JSplitButton is
         * disabled. If no disabled icon has been set this will forward the call
         * to the look and feel to construct an appropriate disabled Icon. Some
         * look and feels might not render the disabled Icon, in which case they
         * will ignore this.
         *
         * @return the disabledPopupIcon property
         * @see AbstractButton#getDisabledIcon()
         * @see JSplitButton#setDisabledPopupIcon(Icon)
         */
        public Icon getDisabledPopupIcon() {
            return popupButton.getDisabledIcon();
        }

        /**
         * Sets the icon used on the popup side when this JSplitButton is
         * disabled. Some look and feels might not render the disabled Icon, in
         * which case they will ignore this.
         *
         * @param icon the icon set
         * @see AbstractButton#setDisabledIcon(Icon)
         * @see JSplitButton#getDisabledPopupIcon()
         */
        public void setDisabledPopupIcon(Icon icon) {
            Icon oldValue = getDisabledPopupIcon();
            firePropertyChange(DISABLED_POPUP_ICON_CHANGED_PROPERTY, oldValue,
                    icon);

            if (!oldValue.equals(icon)) {
                popupButton.setDisabledIcon(icon);
            }
        }

        /**
         * Returns the icon used on the popup side when this JSplitButton is
         * disabled and selected. If no disabled selected Icon has been set, this
         * will forward the call to the LookAndFeel to construct an appropriate
         * disabled Icon from the selection icon if it has been set and to
         * getDisabledPopupIcon() otherwise.
         *
         * @return the disabledSelctedPopupIcon property
         * @see AbstractButton#getDisabledSelectedIcon()
         * @see JSplitButton#setDisabledSelectedPopupIcon(Icon)
         */
        public Icon getDisabledSelectedPopupIcon() {
            return popupButton.getDisabledSelectedIcon();
        }

        /**
         * Sets the icon used on the popup side when this JSplitButton is
         * disabled and selected.
         *
         * @param icon the icon set
         * @see AbstractButton#setDisabledSelectedIcon(Icon)
         * @see JSplitButton#getDisabledSelectedPopupIcon()
         */
        public void setDisabledSelectedPopupIcon(Icon icon) {
            Icon oldValue = getDisabledSelectedPopupIcon();
            firePropertyChange(DISABLED_SELECTED_POPUP_ICON_CHANGED_PROPERTY,
                    oldValue, icon);

            if (!oldValue.equals(icon)) {
                popupButton.setDisabledSelectedIcon(icon);
            }
        }

        /**
         * Returns the icon used on the popup side when this JSpitButton is
         * pressed.
         *
         * @return the pressedPopupIcon property
         * @see AbstractButton#getPressedIcon()
         * @see JSplitButton#setPressedPopupIcon(Icon)
         */
        public Icon getPressedPopupIcon() {
            return popupButton.getPressedIcon();
        }

        /**
         * Sets the icon used on the popup side when this JSplitButton is pressed.
         *
         * @param icon the icon set
         * @see AbstractButton#setPressedIcon(Icon)
         * @see JSplitButton#getPressedPopupIcon()
         */
        public void setPressedPopupIcon(Icon icon) {
            Icon oldValue = getPressedPopupIcon();
            firePropertyChange(PRESSED_POPUP_ICON_CHANGED_PROPERTY, oldValue,
                    icon);

            if (!oldValue.equals(icon)) {
                popupButton.setPressedIcon(icon);
            }
        }

        /**
         * Returns the rollover icon used on the popup side.
         *
         * @return the rolloverPopupIcon property
         * @see AbstractButton#getRolloverIcon()
         * @see JSplitButton#setRolloverPopupIcon(Icon)
         */
        public Icon getRolloverPopupIcon() {
            return popupButton.getRolloverIcon();
        }

        /**
         * Sets the rollover icon used on the popup side.
         *
         * @param icon the icon set
         * @see AbstractButton#setRolloverIcon(Icon)
         * @see JSplitButton#getRolloverPopupIcon()
         */
        public void setRolloverPopupIcon(Icon icon) {
            Icon oldValue = getRolloverPopupIcon();
            firePropertyChange(ROLLOVER_POPUP_ICON_CHANGED_PROPERTY, oldValue,
                    icon);

            if (!oldValue.equals(icon)) {
                popupButton.setRolloverIcon(icon);
            }
        }

        /**
         * Returns the rollover selection icon used on the popup side.
         *
         * @return the rolloverSelectedPopupIcon property
         * @see AbstractButton#getRolloverSelectedIcon()
         * @see JSplitButton#setRolloverSelectedPopupIcon(Icon)
         */
        public Icon getRolloverSelectedPopupIcon() {
            return popupButton.getRolloverSelectedIcon();
        }

        /**
         * Sets the rollover selection icon used on the popup side.
         *
         * @param icon the icon set
         * @see AbstractButton#setRolloverSelectedIcon(Icon)
         * @see JSplitButton#getRolloverSelectedPopupIcon()
         */
        public void setRolloverSelectedPopupIcon(Icon icon) {
            Icon oldValue = getRolloverSelectedPopupIcon();
            firePropertyChange(ROLLOVER_SELECTED_POPUP_ICON_CHANGED_PROPERTY,
                    oldValue, icon);

            if (!oldValue.equals(icon)) {
                popupButton.setRolloverSelectedIcon(icon);
            }
        }

        /**
         * Returns the selected icon used on the popup side.
         *
         * @return the selectedPopupIcon property
         * @see AbstractButton#getSelectedIcon()
         * @see JSplitButton#setSelectedPopupIcon(Icon)
         */
        public Icon getSelectedPopupIcon() {
            return popupButton.getSelectedIcon();
        }

        /**
         * Sets the selected icon used on the popup side.
         *
         * @param icon the icon set
         * @see AbstractButton#setSelectedIcon(Icon)
         * @see JSplitButton#getSelectedPopupIcon()
         */
        public void setSelectedPopupIcon(Icon icon) {
            Icon oldValue = getSelectedPopupIcon();
            firePropertyChange(SELECTED_POPUP_ICON_CHANGED_PROPERTY, oldValue,
                    icon);

            if (!oldValue.equals(icon)) {
                popupButton.setSelectedIcon(icon);
            }
        }

        /**
         * Returns the text of the main part of this JSplitButton. Use this
         * method instead of {@link AbstractButton#getText()}.
         *
         * @return the text of the main part of this JSplitButton
         * @see AbstractButton#getText()
         * @see JSplitButton#setMainText(String)
         */
        public String getMainText() {
            return mainButton.getText();
        }

        /**
         * Sets the text of the main part of this JSplitButton. Use this method
         * instead of {@link AbstractButton#setText(String)}.
         *
         * @param text the text set
         */
        public void setMainText(String text) {
            String oldValue = getMainText();
            firePropertyChange(MAIN_TEXT_CHANGED_PROPERTY, oldValue, text);

            if (!oldValue.equals(text)) {
                mainButton.setText(text);
            }
        }

        /* (non-Javadoc)
         * @see javax.swing.AbstractButton#addActionListener(java.awt.event.ActionListener)
         */
        @Override
        public void addActionListener(ActionListener listener) {
            mainButton.addActionListener(listener);
            super.addActionListener(listener);
        }

        /* (non-Javadoc)
         * @see javax.swing.AbstractButton#removeActionListener(java.awt.event.ActionListener)
         */
        @Override
        public void removeActionListener(ActionListener listener) {
            mainButton.removeActionListener(listener);
            super.removeActionListener(listener);
        }

        /* (non-Javadoc)
         * @see javax.swing.AbstractButton#addChangeListener(javax.swing.event.ChangeListener)
         */
        @Override
        public void addChangeListener(ChangeListener listener) {
            mainButton.addChangeListener(listener);
            super.addChangeListener(listener);
        }

        /* (non-Javadoc)
         * @see javax.swing.AbstractButton#removeChangeListener(javax.swing.event.ChangeListener)
         */
        @Override
        public void removeChangeListener(ChangeListener listener) {
            mainButton.removeChangeListener(listener);
            super.removeChangeListener(listener);
        }

        /* (non-Javadoc)
         * @see javax.swing.AbstractButton#addItemListener(java.awt.event.ItemListener)
         */
        @Override
        public void addItemListener(ItemListener listener) {
            mainButton.addItemListener(listener);
            super.addItemListener(listener);
        }

        /* (non-Javadoc)
         * @see javax.swing.AbstractButton#removeItemListener(java.awt.event.ItemListener)
         */
        @Override
        public void removeItemListener(ItemListener listener) {
            mainButton.removeItemListener(listener);
            super.removeItemListener(listener);
        }

        /* (non-Javadoc)
         * @see javax.swing.AbstractButton#getAction()
         */
        @Override
        public Action getAction() {
            return mainButton.getAction();
        }

        /* (non-Javadoc)
         * @see javax.swing.AbstractButton#setAction(javax.swing.Action)
         */
        @Override
        public void setAction(Action action) {
            Action oldValue = getAction();
            firePropertyChange("action", oldValue, action);
            mainButton.setAction(action);
        }

        /* (non-Javadoc)
         * @see javax.swing.JComponent#getAccessibleContext()
         */
        @Override
        public AccessibleContext getAccessibleContext() {
            if (accessibleContext == null) {
                accessibleContext = new AccessibleJSplitButton();
            }

            return accessibleContext;
        }

        /* (non-Javadoc)
         * @see javax.swing.JComponent#getComponentPopupMenu()
         */
        @Override
        public JPopupMenu getComponentPopupMenu() {
            return (popupAction == null) ? null : popupAction.getPopupMenu();
        }

        /* (non-Javadoc)
         * @see javax.swing.JComponent#setComponentPopupMenu(javax.swing.JPopupMenu)
         */
        @Override
        public void setComponentPopupMenu(JPopupMenu popupMenu) {
            if (popupAction != null) {
                super.removeActionListener(popupAction);
                mainButton.removeActionListener(popupAction);
                popupButton.removeActionListener(popupAction);
                popupAction = null;
            }

            if (popupMenu != null) {
                popupAction = new PopupAction(popupMenu);
                popupButton.addActionListener(popupAction);

                if (alwaysShowPopup) {
                    super.addActionListener(popupAction);
                    mainButton.addActionListener(popupAction);
                }
            }
        }

        /* (non-Javadoc)
         * @see javax.swing.JComponent#getPopupLocation(java.awt.event.MouseEvent)
         */
        @Override
        public Point getPopupLocation(MouseEvent event) {
            return getPopupLocationRelativeTo(event.getComponent());
        }

        /* (non-Javadoc)
         * @see javax.swing.AbstractButton#getSelectedObjects()
         */
        @Override
        public Object[] getSelectedObjects() {
            Object[] result = new Object[1];

            if (mainButton.isSelected()) {
                result[0] = mainButton;
            } else if (popupButton.isSelected()) {
                result[0] = popupButton;
            } else {
                return null;
            }

            return result;
        }

        /* (non-Javadoc)
         * @see javax.swing.AbstractButton#getMnemonic()
         */
        @Override
        public int getMnemonic() {
            return mainButton.getMnemonic();
        }

        /* (non-Javadoc)
         * @see javax.swing.AbstractButton#setMnemonic(int)
         */
        @Override
        public void setMnemonic(int mnemonic) {
            int oldValue = getMnemonic();
            firePropertyChange(MNEMONIC_CHANGED_PROPERTY, oldValue, mnemonic);
            mainButton.setMnemonic(mnemonic);
        }

        /* (non-Javadoc)
         * @see javax.swing.AbstractButton#getDisplayedMnemonicIndex()
         */
        @Override
        public int getDisplayedMnemonicIndex() {
            return mainButton.getDisplayedMnemonicIndex();
        }

        /* (non-Javadoc)
         * @see javax.swing.AbstractButton#setDisplayedMnemonicIndex(int)
         */
        @Override
        public void setDisplayedMnemonicIndex(int index)
                throws IllegalArgumentException {
            int oldValue = getDisplayedMnemonicIndex();
            mainButton.setDisplayedMnemonicIndex(index);
            firePropertyChange("displayedMnemonicIndex", oldValue, index);
        }

        /* (non-Javadoc)
         * @see javax.swing.AbstractButton#setHideActionText(boolean)
         */
        @Override
        public void setHideActionText(boolean hideActionText) {
            mainButton.setHideActionText(hideActionText);
            super.setHideActionText(hideActionText);
        }

        /* (non-Javadoc)
         * @see javax.swing.JComponent#getUIClassID()
         */
        @Override
        public String getUIClassID() {
            return uiClassID;
        }

        /* (non-Javadoc)
         * @see javax.swing.AbstractButton#updateUI()
         */
        @Override
        public void updateUI() {
            setUI((ButtonUI) UIManager.getUI(this));
        }

        private Point getPopupLocationRelativeTo(Component comp) {
            Insets insets = getInsets();
            int height = getHeight();

            if (comp == popupButton) {
                Insets mainButtonInsets = mainButton.getInsets();
                int width = mainButton.getWidth();
                return new Point(-splitGap-width-mainButtonInsets.left-mainButtonInsets.right-30, height);
            } else {
                Insets mainButtonInsets = mainButton.getInsets();
                int width = mainButton.getWidth() + mainButtonInsets.left +
                        mainButtonInsets.right + splitGap;
                if (comp == this) width += insets.left;
                return new Point(width, height);
            }
        }
    }
}
