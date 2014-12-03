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
import juicebox.data.Dataset;
import juicebox.data.DatasetReader;
import juicebox.data.DatasetReaderFactory;
import juicebox.data.MatrixZoomData;
import juicebox.mapcolorui.*;
import juicebox.track.LoadAction;
import juicebox.track.LoadEncodeAction;
import juicebox.track.TrackLabelPanel;
import juicebox.track.TrackPanel;
import juicebox.windowui.*;
import org.apache.log4j.Logger;
import org.broad.igv.Globals;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.ui.FontManager;
import org.broad.igv.ui.util.FileDialogUtils;
import org.broad.igv.ui.util.IconFactory;
import org.broad.igv.util.FileUtils;
import org.broad.igv.util.ParsingUtils;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.border.LineBorder;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.dnd.DropTarget;
import java.awt.event.*;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.io.*;
import java.lang.reflect.InvocationTargetException;
import java.net.URL;
import java.util.*;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.prefs.Preferences;

/**
 * @author James Robinson
 */
public class MainWindow extends JFrame {

    public static final Color RULER_LINE_COLOR = new Color(0, 0, 0, 100);
    public static final int BIN_PIXEL_WIDTH = 1;
    private static final Logger log = Logger.getLogger(MainWindow.class);
    private static final long serialVersionUID = 1428522656885950466L;
    private static final int recentMapListMaxItems = 20;
    private static final int recentLocationMaxItems = 20;
    private static final String recentMapEntityNode = "hicMapRecent";
    private static final String recentLocationEntityNode = "hicLocationRecent";
    public static Cursor fistCursor; // for panning
    private static RecentMenu recentMapMenu;
    private static MainWindow theInstance;
    private static RecentMenu recentLocationMenu;
    private static JMenuItem saveLocationList;
    private static JMenuItem clearLocationList;
    private final ExecutorService threadExecutor = Executors.newFixedThreadPool(1);
    private final HiC hic; // The "model" object containing the state for this instance.
    private static String currentlyLoadedFile = "";
    private static String datasetTitle = "";
    private static String controlTitle;
    private double colorRangeScaleFactor = 1;
    private static LoadDialog loadDialog = null;
    private static JComboBox<Chromosome> chrBox1;
    private static JComboBox<Chromosome> chrBox2;
    private static JideButton refreshButton;
    private static JComboBox<String> normalizationComboBox;
    private static JComboBox<MatrixType> displayOptionComboBox;
    private static JideButton plusButton;
    private static JideButton minusButton;
    private static RangeSlider colorRangeSlider;
    private static ResolutionControl resolutionSlider;
    private static TrackPanel trackPanelX;
    private static TrackPanel trackPanelY;
    private static TrackLabelPanel trackLabelPanel;
    private static HiCRulerPanel rulerPanelX;
    private static HeatmapPanel heatmapPanel;
    private static HiCRulerPanel rulerPanelY;
    private static ThumbnailPanel thumbnailPanel;
    private static JLabel mouseHoverTextPanel;
    private static JTextField positionChrLeft;
    private static JTextField positionChrTop;
    private static JPanel hiCPanel;
    private static JMenu annotationsMenu;
    private static JMenu bookmarksMenu;
    private HiCZoom initialZoom;
    private boolean tooltipAllowedToUpdated = true;

    private MainWindow() {

        hic = new HiC(this);

        initComponents();
        createCursors();
        pack();

        DropTarget target = new DropTarget(this, new FileDropTargetListener(this));
        setDropTarget(target);

        colorRangeSlider.setUpperValue(1200);

        // Tooltip settings
        ToolTipManager.sharedInstance().setDismissDelay(60000);   // 60 seconds

        KeyboardFocusManager.getCurrentKeyboardFocusManager().addKeyEventDispatcher(new HiCKeyDispatcher(hic, displayOptionComboBox));
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
            } catch (Exception e) {
                log.error("Error creating main window", e);
            }
        }
        return theInstance;
    }

    private static MainWindow createMainWindow() {
        return new MainWindow();
    }

    public void updateToolTipText(String s) {
        if(tooltipAllowedToUpdated)
            mouseHoverTextPanel.setText(s);
    }

    public boolean isResolutionLocked() {
        return resolutionSlider.isResolutionLocked();
    }

    public void updateColorSlider(double min, double max, double value) {
        // We need to scale min and max to integers for the slider to work.  Scale such that there are
        // 100 divisions between max and 0

        colorRangeScaleFactor = 100.0 / max;

        colorRangeSlider.setSnapToTicks(true);
        colorRangeSlider.setPaintLabels(true);

        int iMin = (int) (colorRangeScaleFactor * min);
        int iMax = (int) (colorRangeScaleFactor * max);
        int uValue = (int) (colorRangeScaleFactor * value);

        colorRangeSlider.setLowerValue(0);
        colorRangeSlider.setMinimum(iMin);
        colorRangeSlider.setUpperValue(uValue);
        colorRangeSlider.setMaximum(iMax);


        //Change slider lables to reflect change:
        Hashtable<Integer, JLabel> labelTable = new Hashtable<Integer, JLabel>();

        Font f = FontManager.getFont(8);

        final JLabel minTickLabel = new JLabel(String.valueOf((int) min));
        minTickLabel.setFont(f);
        final JLabel maxTickLabel = new JLabel(String.valueOf((int) max));
        maxTickLabel.setFont(f);
        final JLabel LoTickLabel = new JLabel(String.valueOf(0));
        LoTickLabel.setFont(f);
        final JLabel UpTickLabel = new JLabel(String.valueOf((int) value));
        UpTickLabel.setFont(f);

        labelTable.put(0, LoTickLabel);
        labelTable.put(iMin, minTickLabel);
        labelTable.put(uValue, UpTickLabel);
        labelTable.put(iMax, maxTickLabel);


        colorRangeSlider.setLabelTable(labelTable);
    }

    public void updateColorSlider(double min, double lower, double upper, double max) {
        // We need to scale min and max to integers for the slider to work.  Scale such that there are
        // 100 divisions between max and 0

        colorRangeScaleFactor = 100.0 / max;

        colorRangeSlider.setPaintTicks(true);
        colorRangeSlider.setSnapToTicks(true);

        int iMin = (int) (colorRangeScaleFactor * min);
        int iMax = (int) (colorRangeScaleFactor * max);
        int lValue = (int) (colorRangeScaleFactor * lower);
        int uValue = (int) (colorRangeScaleFactor * upper);

        colorRangeSlider.setMinimum(iMin);
        colorRangeSlider.setLowerValue(lValue);
        colorRangeSlider.setUpperValue(uValue);
        colorRangeSlider.setMaximum(iMax);

        Font f = FontManager.getFont(8);

        Hashtable<Integer, JLabel> labelTable = new Hashtable<Integer, JLabel>();

        final JLabel minTickLabel = new JLabel(String.valueOf((int) min));
        minTickLabel.setFont(f);
        final JLabel maxTickLabel = new JLabel(String.valueOf((int) max));
        maxTickLabel.setFont(f);
        final JLabel LoTickLabel = new JLabel(String.valueOf((int) lower));
        LoTickLabel.setFont(f);
        final JLabel UpTickLabel = new JLabel(String.valueOf((int) upper));
        UpTickLabel.setFont(f);

        labelTable.put(iMin, minTickLabel);
        labelTable.put(iMax, maxTickLabel);
        labelTable.put(lValue, LoTickLabel);
        labelTable.put(uValue, UpTickLabel);

        colorRangeSlider.setLabelTable(labelTable);
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

    public void load(final List<String> files, final boolean control) {

        String file = files.get(0);

        if (file.equals(currentlyLoadedFile)) {
            JOptionPane.showMessageDialog(MainWindow.this, "File already loaded");
            return;
        } else {
            currentlyLoadedFile = file;
        }

        hic.setNormalizationType(NormalizationType.NONE);

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
                            } else {
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

                        saveLocationList.setEnabled(true);
                        recentLocationMenu.setEnabled(true);
                        clearLocationList.setEnabled(true);

                        positionChrTop.setEnabled(true);
                        positionChrLeft.setEnabled(true);

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

        setNormalizationDisplayState();

        hic.setSelectedChromosomes(chrX, chrY);
        rulerPanelX.setContext(hic.getXContext(), HiCRulerPanel.Orientation.HORIZONTAL);
        rulerPanelY.setContext(hic.getYContext(), HiCRulerPanel.Orientation.VERTICAL);
        setInitialZoom();

        refresh();


    }

    public void setNormalizationDisplayState() {

        Chromosome chr1 = (Chromosome) chrBox1.getSelectedItem();
        Chromosome chr2 = (Chromosome) chrBox2.getSelectedItem();

        Chromosome chrX = chr1.getIndex() < chr2.getIndex() ? chr1 : chr2;
        Chromosome chrY = chr1.getIndex() < chr2.getIndex() ? chr2 : chr1;

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
    }

    public void repaintTrackPanels() {
        trackPanelX.repaint();
        trackPanelY.repaint();
    }

    public void refresh() {
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
            if (hic.getControlMatrix() != null)
                zdControl = hic.getControlMatrix().getFirstZoomData(hic.getZoom().getUnit());
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

    public void loadMenuItemActionPerformed(boolean control) {
        FilenameFilter hicFilter = new FilenameFilter() {
            public boolean accept(File dir, String name) {
                return name.toLowerCase().endsWith(".hic");
            }
        };

        File[] files = FileDialogUtils.chooseMultiple("Choose Hi-C file(s)", DirectoryManager.getUserDirectory(), hicFilter);
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
            try {//TODO S7 - MSS
                load(Arrays.asList(url), control);

                String path = (new URL(url)).getPath();
                if (control) controlTitle = title;// TODO should the other one be set to empty/null
                else datasetTitle = title;
                updateTitle();
            } catch (IOException e1) {
                JOptionPane.showMessageDialog(this, "Error while trying to load " + url, "Error", JOptionPane.ERROR_MESSAGE);
            }
        }
    }

    public void loadFromURLActionPerformed(boolean control) {
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
                    url = "http://hicfiles.tc4ga.com/juicebox.properties";
                }
                InputStream is = ParsingUtils.openInputStream(url);
                properties = new Properties();
                if (is == null) {
                    //No slection made:
                    return;
                } else {
                    properties.load(is);
                }
            } catch (Exception error) {
                JOptionPane.showMessageDialog(this, "Can't find properties file for loading list", "Error", JOptionPane.ERROR_MESSAGE);
                return;
            }
            loadDialog = new LoadDialog(this, properties);
            if (!loadDialog.getSuccess()) {
                loadDialog = null;
                return;
            }
        }
        loadDialog.setControl(control);
        loadDialog.setVisible(true);
    }

    public void updateTitle(boolean control, String title) {
        if (control) controlTitle = title;
        else datasetTitle = title;
        updateTitle();
    }

    private void updateTitle() {
        String newTitle = datasetTitle;
        if (controlTitle != null) newTitle += "  (control=" + controlTitle + ")";
        setTitle(newTitle);
    }

    private void clearMapActionPerformed() {
        Preferences prefs = Preferences.userNodeForPackage(Globals.class);
        for (int i = 0; i < recentMapListMaxItems; i++) {
            prefs.remove(recentMapEntityNode + i);
        }
    }

    private void clearLocationActionPerformed() {
        Preferences prefs = Preferences.userNodeForPackage(Globals.class);
        for (int i = 0; i < recentLocationMaxItems; i++) {
            prefs.remove(recentLocationEntityNode + i);
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
        colorRangeSlider.setEnabled(option == MatrixType.OBSERVED || option == MatrixType.CONTROL || option == MatrixType.OE);
        colorRangeSlider.setDisplayToOE(option == MatrixType.OE);
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
        String value = (String) normalizationComboBox.getSelectedItem();
        NormalizationType chosen = null;
        for (NormalizationType type : NormalizationType.values()) {
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
        // TODO S7 - MSS
        Callable<Object> wrapper = new Callable<Object>() {
            public Object call() throws Exception {
                //System.out.println("Glassify : " + rootPane);
                MainWindow.this.showGlassPane();
                try {
                    runnable.run();
                    return "done";
                } finally {
                    MainWindow.this.hideGlassPane();
                }
            }
        };

        return threadExecutor.submit(wrapper);
    }

    public RecentMenu getRecentMapMenu() {
        return recentMapMenu;
    }

    public RecentMenu getRecentStateMenu() {
        return recentLocationMenu;
    }

    public void showGlassPane() {
        setGlassPaneVisibility(this.getGlassPane(), Cursor.WAIT_CURSOR, true);
    }

    public void hideGlassPane() {
        setGlassPaneVisibility(this.getGlassPane(), Cursor.DEFAULT_CURSOR, false);
    }

    public void setGlassPaneVisibility(Component glassPane, int cursorState, boolean isVisible) {
        glassPane.setCursor(Cursor.getPredefinedCursor(cursorState));
        glassPane.setVisible(isVisible);
        setWaitingStatus(cursorState);
    }

    public void setWaitingStatus(int cursorState) {
        rootPane.getTopLevelAncestor().setCursor(Cursor.getPredefinedCursor(cursorState));
        rootPane.setCursor(Cursor.getPredefinedCursor(cursorState));
        hiCPanel.getTopLevelAncestor().setCursor(Cursor.getPredefinedCursor(cursorState));
        hiCPanel.setCursor(Cursor.getPredefinedCursor(cursorState));
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
        } catch (Exception e) {
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
        chrBox1.setModel(new DefaultComboBoxModel<Chromosome>(new Chromosome[]{new Chromosome(0, "All", 0)}));
        chrBox1.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                chrBox1ActionPerformed(e);
            }
        });
        chrButtonPanel.add(chrBox1);

        //---- chrBox2 ----
        chrBox2 = new JComboBox<Chromosome>();
        chrBox2.setModel(new DefaultComboBoxModel<Chromosome>(new Chromosome[]{new Chromosome(0, "All", 0)}));
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

        colorRangeSlider.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseEntered(MouseEvent mouseEvent) {
                super.mouseEntered(mouseEvent);
                colorRangeSliderUpdateToolTip();
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
                colorRangeSliderUpdateToolTip();
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
                colorRangeSliderUpdateToolTip();
            }
        });

        minusButton = new JideButton();
        minusButton.setIcon(new ImageIcon(getClass().getResource("/images/zoom-minus.png")));
        minusButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                //Set limit to maximum range:
                if (colorRangeSlider.getMaximum() > 2) {
                    colorRangeSlider.setMaximum(colorRangeSlider.getMaximum() / 2);
                }
                colorRangeSliderUpdateToolTip();
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


        // needs to be created after heatmap panel
        // Resolution  panel
        resolutionSlider = new ResolutionControl(hic, this, heatmapPanel);
        toolbarPanel.add(resolutionSlider);
        toolbarPanel.setEnabled(false);


        //======== Right side panel ========

        JPanel rightSidePanel = new JPanel(new BorderLayout());//(new BorderLayout());
        rightSidePanel.setBackground(Color.white);
        rightSidePanel.setPreferredSize(new Dimension(200, 1000));
        rightSidePanel.setMaximumSize(new Dimension(10000, 10000));
        //rightSidePanel.getLayout().setResizable(true);
        //rightSidePanel.setBorder(new EmptyBorder(0, 10, 0, 0));
        //LayoutManager lm = new GridLayout(FlowLayout.LEFT, 10, 20);
        //rightSidePanel.setLayout(lm);
        //rightSidePanel.setLayout(null);

        //======== Bird's view mini map ========

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
        thumbnailPanel.setBounds(new Rectangle(new Point(0, 0), thumbnailPanel.getPreferredSize()));
        thumbPanel.add(thumbnailPanel);
        thumbPanel.setBackground(Color.white);
        rightSidePanel.add(thumbPanel, BorderLayout.NORTH);//, BorderLayout.PAGE_START

        //========= mouse hover text ======

        mouseHoverTextPanel = new JLabel();
        mouseHoverTextPanel.setBackground(Color.white);
        mouseHoverTextPanel.setVerticalAlignment(SwingConstants.CENTER);
        mouseHoverTextPanel.setHorizontalAlignment(SwingConstants.CENTER);
        mouseHoverTextPanel.setBorder(LineBorder.createBlackLineBorder());
        int mouseTextY = rightSidePanel.getBounds().y + rightSidePanel.getBounds().height;

        Dimension prefSize = new Dimension(170, 490);
        mouseHoverTextPanel.setPreferredSize(prefSize);
        mouseHoverTextPanel.setBounds(new Rectangle(new Point(0, mouseTextY), prefSize));
        rightSidePanel.add(mouseHoverTextPanel, BorderLayout.CENTER);//, BorderLayout.PAGE_END

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
//        final PopupMenu testPopUp = new PopupMenu();
//        testPopUp.setLabel("Please wait");
//        rootPane.getGlassPane().add(testPopUp);
        rootPane.getGlassPane().addMouseListener(new MouseAdapter() {
        });

    }

    public void setPositionChrLeft(String newPositionDate) {
        this.positionChrLeft.setText(newPositionDate);
    }


    public void setPositionChrTop(String newPositionDate) {
        this.positionChrTop.setText(newPositionDate);
    }


    public void parsePositionText() {
        //Expected format: <chr>:<start>-<end>:<resolution>

        String delimiters = "\\s+|:\\s*|\\-\\s*";
        Integer outBinSize = 0;
        Long outBinLeft = 0L;
        Long outBinTop = 0L;
        Long topStart = 0L;
        Long topEnd = 0L;
        Long leftStart = 0L;
        Long leftEnd = 0L;

        String[] leftChrTokens = this.positionChrLeft.getText().split(delimiters);
        String[] topChrTokens = this.positionChrTop.getText().split(delimiters);


        String LeftChrName = "";
        String TopChrName = "";
        int LeftChrInt = 0;
        int TopChrInt = 0;

        //Read Chromosomes:
        //First chromosome:
        if (topChrTokens.length > 0) {
            if (topChrTokens[0].toLowerCase().contains("chr")) {
                TopChrName = topChrTokens[0].substring(3);
            } else {
                TopChrName = topChrTokens[0].toLowerCase();
            }
        } else {
            this.positionChrTop.setBackground(Color.yellow);
            return;
        }
        try {
            TopChrInt = Integer.parseInt(TopChrName);
            //TBD - replace with actual chromosome range
            if (TopChrInt > 22) {
                this.positionChrTop.setBackground(Color.yellow);
                return;
            }

        } catch (Exception e) {
            if (TopChrName.toLowerCase().equals("x")) {
                TopChrName = "X";
            } else if (TopChrName.toLowerCase().equals("y")) {
                TopChrName = "Y";
            } else if (TopChrName.toLowerCase().equals("mt") || TopChrName.toLowerCase().equals("m")) {
                TopChrName = "MT";
            } else {
                this.positionChrTop.setBackground(Color.yellow);
                return;
            }
        }

        //Second chromosome:
        if (leftChrTokens.length > 0) {
            if (leftChrTokens[0].toLowerCase().contains("chr")) {
                LeftChrName = leftChrTokens[0].substring(3);
            } else {
                LeftChrName = leftChrTokens[0].toLowerCase();
            }
        } else {
            this.positionChrLeft.setBackground(Color.yellow);
            return;
        }
        try {
            LeftChrInt = Integer.parseInt(LeftChrName);

            //TBD - replace with actual chromosome range
            if (LeftChrInt > 22) {
                this.positionChrLeft.setBackground(Color.yellow);
                return;
            }
        } catch (Exception e) {
            if (LeftChrName.toLowerCase().equals("x")) {
                LeftChrName = "X";
            } else if (LeftChrName.toLowerCase().equals("y")) {
                LeftChrName = "Y";
            } else if (LeftChrName.toLowerCase().equals("mt") || LeftChrName.toLowerCase().equals("m")) {
                LeftChrName = "MT";
            } else {
                this.positionChrLeft.setBackground(Color.yellow);
                return;
            }
        }

        //Read positions:
        if (topChrTokens.length > 2) {
            //Make sure values are numerical:
            try {
                Long.parseLong(topChrTokens[1].replaceAll(",", ""));
            } catch (Exception e) {
                this.positionChrTop.setBackground(Color.yellow);
                return;
            }
            try {
                Long.parseLong(topChrTokens[2].replaceAll(",", ""));
            } catch (Exception e) {
                this.positionChrLeft.setBackground(Color.yellow);
                return;
            }
            topStart = Long.min(Long.valueOf(topChrTokens[1].replaceAll(",", "")), Long.valueOf(topChrTokens[2].replaceAll(",", "")));
            topEnd = Long.max(Long.valueOf(topChrTokens[1].replaceAll(",", "")), Long.valueOf(topChrTokens[2].replaceAll(",", "")));
            outBinTop = topStart + ((topEnd - topStart) / 2);

        } else if (topChrTokens.length > 1) {
            outBinTop = Long.valueOf(topChrTokens[1].replaceAll(",", ""));
        }


        if (leftChrTokens.length > 2) {
            leftStart = Long.min(Long.valueOf(leftChrTokens[1].replaceAll(",", "")), Long.valueOf(leftChrTokens[2].replaceAll(",", "")));
            leftEnd = Long.max(Long.valueOf(leftChrTokens[1].replaceAll(",", "")), Long.valueOf(leftChrTokens[2].replaceAll(",", "")));
            outBinLeft = leftStart + ((leftEnd - leftStart) / 2);
        } else if (topChrTokens.length > 1) {
            //Make sure values are numerical:
            try {
                Long.parseLong(topChrTokens[1].replaceAll(",", ""));
            } catch (Exception e) {
                this.positionChrTop.setBackground(Color.yellow);
                return;
            }
            outBinLeft = Long.valueOf(leftChrTokens[1].replaceAll(",", ""));
        }

        //Read resolution:
        if (topChrTokens.length > 3) {
            //Make sure value is numeric:
            try {
                Integer.parseInt(topChrTokens[3]);
            } catch (Exception e) {
                this.positionChrTop.setBackground(Color.yellow);
                return;
            }
            outBinSize = Integer.parseInt(topChrTokens[3]);
        } else if (leftChrTokens.length > 3) {
            //Make sure value is numeric:
            try {
                Integer.parseInt(leftChrTokens[3]);
            } catch (Exception e) {
                this.positionChrLeft.setBackground(Color.yellow);
                return;
            }
            outBinSize = Integer.parseInt(leftChrTokens[3]);
        } else if (hic.getZoom().getBinSize() != 0) {
            outBinSize = hic.getZoom().getBinSize();
        }

        this.positionChrTop.setBackground(Color.white);
        this.positionChrLeft.setBackground(Color.white);

        hic.setState(TopChrName, LeftChrName, "BP", outBinSize, 0, 0, hic.getScaleFactor());
        if (outBinTop > 0 && outBinLeft > 0) {
            hic.centerBP(Math.round(outBinTop), Math.round(outBinLeft));
        }

        //We might end with ALL->All view, make sure normalization state is updates acordingly...
        setNormalizationDisplayState();
    }


    private void colorRangeSliderUpdateToolTip() {
        if (hic.getDisplayOption() == MatrixType.OBSERVED ||
                hic.getDisplayOption() == MatrixType.CONTROL ||
                hic.getDisplayOption() == MatrixType.OE) {

            int iMin = colorRangeSlider.getMinimum();
            int lValue = colorRangeSlider.getLowerValue();
            int uValue = colorRangeSlider.getUpperValue();
            int iMax = colorRangeSlider.getMaximum();

            colorRangeSlider.setToolTipText("<html>Range: " + (int) (iMin / colorRangeScaleFactor) + " "
                    + (int) (iMax / colorRangeScaleFactor) + "<br>Showing: " +
                    (int) (lValue / colorRangeScaleFactor) + " "
                    + (int) (uValue / colorRangeScaleFactor)
                    + "</html>");

            Font f = FontManager.getFont(8);

            Hashtable<Integer, JLabel> labelTable = new Hashtable<Integer, JLabel>();

            final JLabel minTickLabel = new JLabel(String.valueOf((int) (iMin / colorRangeScaleFactor)));
            minTickLabel.setFont(f);
            final JLabel LoTickLabel = new JLabel(String.valueOf((int) (lValue / colorRangeScaleFactor)));
            LoTickLabel.setFont(f);
            final JLabel UpTickLabel = new JLabel(String.valueOf((int) (uValue / colorRangeScaleFactor)));
            UpTickLabel.setFont(f);
            final JLabel maxTickLabel = new JLabel(String.valueOf((int) (iMax / colorRangeScaleFactor)));
            maxTickLabel.setFont(f);

            labelTable.put(iMin, minTickLabel);
            labelTable.put(lValue, LoTickLabel);
            labelTable.put(uValue, UpTickLabel);
            labelTable.put(iMax, maxTickLabel);

            colorRangeSlider.setLabelTable(labelTable);

        }

    }

    private JMenuBar createMenuBar() {


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

        //TODO S7 - MSS
        recentMapMenu = new RecentMenu("Open recently used map", recentMapListMaxItems, recentMapEntityNode) {
            public void onSelectPosition(String mapPath) {
                String delimiter = "@@";
                String[] temp;
                temp = mapPath.split(delimiter);
                loadFromRecentActionPerformed((temp[1]), (temp[0]), false);
            }
        };
        recentMapMenu.setMnemonic('R');
        fileMenu.add(recentMapMenu);

        //---- Clear Recent ----
        JMenuItem clearMapList = new JMenuItem();
        clearMapList.setText("Clear recently used maps list");
        clearMapList.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                //Clear all items from preferences:
                clearMapActionPerformed();
                //clear the existing items
                recentMapMenu.removeAll();
            }
        });
        fileMenu.add(clearMapList);

        fileMenu.addSeparator();

        JMenuItem showStats = new JMenuItem("Show Dataset Metrics");
        showStats.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                if (hic.getDataset() == null) {
                    JOptionPane.showMessageDialog(MainWindow.this, "File must be loaded to show info", "Error", JOptionPane.ERROR_MESSAGE);
                } else {
                    JDialog qcDialog = new QCDialog(MainWindow.this, hic);
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
                new SaveImageDialog(null, hic, hiCPanel);
            }
        });
        fileMenu.add(saveToImage);


        if (!HiCGlobals.isRestricted) {
            JMenuItem dump = new JMenuItem("Export Data...");
            dump.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent actionEvent) {
                    if (hic.getDataset() == null) {
                        JOptionPane.showMessageDialog(MainWindow.this, "File must be loaded to show info", "Error", JOptionPane.ERROR_MESSAGE);
                    } else {
                        new DumpDialog(MainWindow.this, hic);
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

        bookmarksMenu = new JMenu("Bookmarks");
        //---- Save location ----
        saveLocationList = new JMenuItem();
        saveLocationList.setText("Save current location");
        saveLocationList.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                //code to add a recent location to the menu
                String stateString = hic.saveState();
                String stateDescriptionString = hic.getDefaultLocationDescription();
                String stateDescription = JOptionPane.showInputDialog(MainWindow.this,
                        "Enter description for saved location:", stateDescriptionString);
                if (null != stateDescription) {
                    getRecentStateMenu().addEntry(stateDescription + "@@" + stateString, true);
                }
            }
        });
        saveLocationList.setEnabled(false);
        bookmarksMenu.add(saveLocationList);


        recentLocationMenu = new RecentMenu("Restore saved location", recentLocationMaxItems, recentLocationEntityNode) {
            public void onSelectPosition(String mapPath) {
                String delimiter = "@@";
                String[] temp;
                temp = mapPath.split(delimiter);
                hic.restoreState(temp[1]);//temp[1]
                setNormalizationDisplayState();
            }
        };
        recentLocationMenu.setMnemonic('S');
        recentLocationMenu.setEnabled(false);
        bookmarksMenu.add(recentLocationMenu);

        //---- Clear Recent state ----
        clearLocationList = new JMenuItem();
        clearLocationList.setText("Clear saved locations list");
        clearLocationList.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                //Clear all items from preferences:
                clearLocationActionPerformed();
                //clear the existing items
                recentLocationMenu.removeAll();
            }
        });

        clearLocationList.setEnabled(false);
        bookmarksMenu.add(clearLocationList);
        bookmarksMenu.addSeparator();


        //========= Positioning panel ======

        JLabel positionLabel = new JLabel("Jump To:");
        //positionLabel.setFont(new Font("Arial", Font.ITALIC, 14));

        positionChrTop = new JTextField();
        positionChrTop.setEnabled(false);
        positionChrTop.addActionListener(new ActionListener() {

            public void actionPerformed(ActionEvent e) {

                parsePositionText();

            }
        });

        positionChrTop.setPreferredSize(new Dimension(180, 25));
        positionChrTop.setPreferredSize(new Dimension(180, 25));
        positionChrTop.setFont(new Font("Arial", Font.ITALIC, 10));

        positionChrLeft = new JTextField();
        positionChrLeft.setEnabled(false);
        positionChrLeft.addActionListener(new ActionListener() {

            public void actionPerformed(ActionEvent e) {

                parsePositionText();

            }
        });
        positionChrLeft.setPreferredSize(new Dimension(180, 25));
        positionChrLeft.setPreferredSize(new Dimension(180, 25));
        positionChrLeft.setFont(new Font("Arial", Font.ITALIC, 10));

        positionLabel.setPreferredSize(new Dimension(200, 25));
        positionChrTop.setPreferredSize(new Dimension(200, 30));
        positionChrLeft.setPreferredSize(new Dimension(200, 30));

        bookmarksMenu.add(positionLabel);
        bookmarksMenu.add(positionChrTop);
        bookmarksMenu.add(positionChrLeft);

        bookmarksMenu.setBackground(Color.white);
        bookmarksMenu.setBorder(LineBorder.createBlackLineBorder());

        menuBar.add(fileMenu);
        menuBar.add(annotationsMenu);
        menuBar.add(bookmarksMenu);
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

    public String getToolTip(){
        return mouseHoverTextPanel.getText();
    }

    public boolean isTooltipAllowedToUpdated(){
        return  tooltipAllowedToUpdated;
    }

    public boolean toggleToolTipUpdates(){
        tooltipAllowedToUpdated = !tooltipAllowedToUpdated;
        return tooltipAllowedToUpdated;
    }


    private abstract class protectedGlassProcessing {

        abstract void encapsulatedCommand();

        public void process() {
            try {
                MainWindow.this.showGlassPane();
                encapsulatedCommand();
            } finally {
                MainWindow.this.hideGlassPane();
            }

        }
    }
}
