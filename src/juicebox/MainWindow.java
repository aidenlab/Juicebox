/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2014 Broad Institute, Aiden Lab
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

package juicebox;

import com.jidesoft.swing.JideButton;
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
import org.broad.igv.ui.util.MessageUtils;
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
import java.awt.font.TextAttribute;
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

    public static final Color RULER_LINE_COLOR = new Color(0, 0, 230, 100);
    public static final int BIN_PIXEL_WIDTH = 1;
    private static final Logger log = Logger.getLogger(MainWindow.class);
    private static final long serialVersionUID = 1428522656885950466L;
    private static final int recentMapListMaxItems = 10;
    private static final int recentLocationMaxItems = 20;
    private static final String recentMapEntityNode = "hicMapRecent";
    private static final String recentLocationEntityNode = "hicLocationRecent";
    public static Cursor fistCursor; // for panning
    private static RecentMenu recentMapMenu;
    private static MainWindow theInstance;
    private static RecentMenu recentLocationMenu;
    private static JMenuItem saveLocationList;
    private static JMenuItem clearLocationList;
    private static String currentlyLoadedFile = "";
    private static String datasetTitle = "";
    private static String controlTitle;
    private static LoadDialog loadDialog = null;
    private static JComboBox<Chromosome> chrBox1;
    private static JComboBox<Chromosome> chrBox2;
    private static JideButton refreshButton;
    private static JideButton goButton;
    private static JComboBox<String> normalizationComboBox;
    private static JComboBox<MatrixType> displayOptionComboBox;
    private static JideButton plusButton;
    private static JideButton minusButton;
    private static RangeSlider colorRangeSlider;
    private static JLabel colorRangeLabel;
    private static ResolutionControl resolutionSlider;
    private static TrackPanel trackPanelX;
    private static TrackPanel trackPanelY;
    private static TrackLabelPanel trackLabelPanel;
    private static HiCRulerPanel rulerPanelX;
    private static HeatmapPanel heatmapPanel;
    private static HiCRulerPanel rulerPanelY;
    private static ThumbnailPanel thumbnailPanel;
    private static JEditorPane mouseHoverTextPanel;
    private static JTextField positionChrLeft;
    private static JTextField positionChrTop;
    private static JPanel hiCPanel;
    private static JMenu annotationsMenu;
    private static JMenu bookmarksMenu;
    private static final DisabledGlassPane disabledGlassPane = new DisabledGlassPane();
    private final ExecutorService threadExecutor = Executors.newFixedThreadPool(1);
    private final HiC hic; // The "model" object containing the state for this instance.
    private double colorRangeScaleFactor = 1;
    private HiCZoom initialZoom;
    private boolean tooltipAllowedToUpdated = true;
    private int[] colorValuesToRestore = null;
    private Properties properties;

    private MainWindow() {

        hic = new HiC(this);

        initComponents();
        createCursors();
        pack();

        DropTarget target = new DropTarget(this, new FileDropTargetListener(this));
        setDropTarget(target);

        colorRangeSlider.setUpperValue(1200);
        colorRangeSlider.setDisplayToBlank(true);

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

        log.debug("Default User Directory: " + DirectoryManager.getUserDirectory());
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
        if (tooltipAllowedToUpdated)
            mouseHoverTextPanel.setText(s);
        mouseHoverTextPanel.setCaretPosition(0);
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
        //colorRangeSlider.setSnapToTicks(true);
        colorRangeSlider.setPaintLabels(true);

        int iMin = (int) (colorRangeScaleFactor * min);
        int iMax = (int) (colorRangeScaleFactor * max);
        int lValue = (int) (colorRangeScaleFactor * lower);
        int uValue = (int) (colorRangeScaleFactor * upper);

        colorRangeSlider.setMinimum(iMin);
        colorRangeSlider.setLowerValue(lValue);
        colorRangeSlider.setUpperValue(uValue);
        colorRangeSlider.setMaximum(iMax);

        Hashtable<Integer, JLabel> labelTable = new Hashtable<Integer, JLabel>();

        Font f = FontManager.getFont(8);

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
        resolutionSlider.reset();
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

    /*
     * Only accessed from within another unsafe method in Heatmap Panel class,
     * which in turn is encapsulated (i.e. made safe)
     */
    public void unsafeSetSelectedChromosomes(Chromosome xChrom, Chromosome yChrom) {
        chrBox1.setSelectedIndex(yChrom.getIndex());
        chrBox2.setSelectedIndex(xChrom.getIndex());
        unsafeRefreshChromosomes();
    }

    public void setSelectedChromosomesNoRefresh(Chromosome xChrom, Chromosome yChrom) {
        chrBox1.setSelectedIndex(yChrom.getIndex());
        chrBox2.setSelectedIndex(xChrom.getIndex());
        rulerPanelX.setContext(hic.getXContext(), HiCRulerPanel.Orientation.HORIZONTAL);
        rulerPanelY.setContext(hic.getYContext(), HiCRulerPanel.Orientation.VERTICAL);
        resolutionSlider.setEnabled(!xChrom.getName().equals(Globals.CHR_ALL));
        initialZoom = null;
    }

    public void safeLoad(final List<String> files, final boolean control){
        Runnable runnable = new Runnable() {
            public void run() {
                try {
                    unsafeload(files, control);
                    updateThumbnail();
                    refresh();
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
        executeLongRunningTask(runnable, "MainWindow safe load");
    }

    private void unsafeload(final List<String> files, final boolean control) throws IOException{

        String file = files.get(0);

        if (file.equals(currentlyLoadedFile)) {
            JOptionPane.showMessageDialog(MainWindow.this, "File already loaded");
            return;
        } else {
            currentlyLoadedFile = file;
        }

        colorValuesToRestore = null;
        //heatmapPanel.setBorder(LineBorder.createBlackLineBorder());
        //thumbnailPanel.setBorder(LineBorder.createBlackLineBorder());
        mouseHoverTextPanel.setBorder(LineBorder.createGrayLineBorder());
        hic.setNormalizationType(NormalizationType.NONE);

        if (file.endsWith("hic")) {
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
                unsafeRefreshChromosomes();
            }
            displayOptionComboBox.setModel(new DefaultComboBoxModel<MatrixType>(options));
            displayOptionComboBox.setSelectedIndex(0);
            chrBox1.setEnabled(true);
            chrBox2.setEnabled(true);
            refreshButton.setEnabled(true);

            setColorRangeSliderVisible(true);
            colorRangeSlider.setDisplayToBlank(false);
            plusButton.setEnabled(true);
            minusButton.setEnabled(true);
            annotationsMenu.setEnabled(true);

            saveLocationList.setEnabled(true);
            recentLocationMenu.setEnabled(true);

            positionChrTop.setEnabled(true);
            positionChrLeft.setEnabled(true);
            goButton.setEnabled(true);

            //refresh(); // an additional refresh seems to remove the upper left black corner
        } else {
            JOptionPane.showMessageDialog(this, "Please choose a .hic file to load");

        }
    }

    private void unsafeRefreshChromosomes() {

        if (chrBox1.getSelectedIndex() == 0 || chrBox2.getSelectedIndex() == 0) {
            chrBox1.setSelectedIndex(0);
            chrBox2.setSelectedIndex(0);
        }

        Chromosome chr1 = (Chromosome) chrBox1.getSelectedItem();
        Chromosome chr2 = (Chromosome) chrBox2.getSelectedItem();

        Chromosome chrX = chr1.getIndex() < chr2.getIndex() ? chr1 : chr2;
        Chromosome chrY = chr1.getIndex() < chr2.getIndex() ? chr2 : chr1;

        setNormalizationDisplayState();

        hic.setSelectedChromosomes(chrX, chrY);
        rulerPanelX.setContext(hic.getXContext(), HiCRulerPanel.Orientation.HORIZONTAL);
        rulerPanelY.setContext(hic.getYContext(), HiCRulerPanel.Orientation.VERTICAL);
        setInitialZoom();

        updateThumbnail();
    }

    public void setNormalizationDisplayState() {

        Chromosome chr1 = (Chromosome) chrBox1.getSelectedItem();
        Chromosome chr2 = (Chromosome) chrBox2.getSelectedItem();

//        Chromosome chrX = chr1.getIndex() < chr2.getIndex() ? chr1 : chr2;
        Chromosome chrY = chr1.getIndex() < chr2.getIndex() ? chr2 : chr1;

        // Test for new dataset ("All"),  or change in chromosome
        final boolean wholeGenome = chrY.getName().equals("All");
        final boolean intraChr = chr1.getIndex() != chr2.getIndex();
        if (wholeGenome) { // for now only allow observed
            hic.setDisplayOption(MatrixType.OBSERVED);
            displayOptionComboBox.setSelectedIndex(0);
            normalizationComboBox.setSelectedIndex(0);
        } else if (intraChr) {
            if (hic.getDisplayOption() == MatrixType.PEARSON) {
                hic.setDisplayOption(MatrixType.OBSERVED);
                displayOptionComboBox.setSelectedIndex(0);
            }
        }

        normalizationComboBox.setEnabled(!wholeGenome);
        // Actually we'd like to enable
        displayOptionComboBox.setEnabled(true); // TODO add capability to view whole genome in OE, etc
    }

    public void repaintTrackPanels() {
        trackPanelX.repaint();
        trackPanelY.repaint();
    }

    public void refresh() {
        getHeatmapPanel().clearTileCache();
        repaint();
        updateThumbnail();
        //System.err.println(heatmapPanel.getSize());
    }

    public void updateThumbnail() {
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

        //For now, in case of Pearson - set initial to 500KB resolution.
        if ((hic.getDisplayOption() == MatrixType.PEARSON)) {
            initialZoom = hic.getMatrix().getFirstPearsonZoomData(HiC.Unit.BP).getZoom();
        } else if (hic.getXContext().getChromosome().getName().equals("All")) {
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
        colorValuesToRestore = null;
        Runnable runnable = new Runnable() {
            @Override
            public void run() {
                unsafeRefreshChromosomes();
            }
        };
        executeLongRunningTask(runnable, "Refresh Button");

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
            safeLoad(fileNames, control);

            if (control) controlTitle = str;
            else datasetTitle = str;
            updateTitle();

        }
    }

    private void loadFromRecentActionPerformed(String url, String title, boolean control) {

        if (url != null) {
            recentMapMenu.addEntry(title.trim() + "@@" + url, true);
            safeLoad(Arrays.asList(url), control);

            if (control) controlTitle = title;// TODO should the other one be set to empty/null
            else datasetTitle = title;
            updateTitle();

        }
    }

    public void loadFromURLActionPerformed(boolean control) {
        String url = JOptionPane.showInputDialog("Enter URL: ");
        if (url != null) {
            try {
                safeLoad(Arrays.asList(url), control);

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
            initProperties();
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

        if (hic.getDisplayOption() == MatrixType.OE || hic.getDisplayOption() == MatrixType.RATIO) {
            //System.out.println(colorRangeSlider.getUpperValue());
            heatmapPanel.setOEMax(colorRangeSlider.getUpperValue());
        }
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

    private void safeDisplayOptionComboBoxActionPerformed(final ActionEvent e) {
        Runnable runnable = new Runnable() {
            public void run() {
                unsafeDisplayOptionComboBoxActionPerformed(e);
            }
        };
        executeLongRunningTask(runnable, "DisplayOptionsComboBox");
    }

    private void unsafeDisplayOptionComboBoxActionPerformed(ActionEvent e) {

        MatrixType option = (MatrixType) (displayOptionComboBox.getSelectedItem());
        // ((ColorRangeModel)colorRangeSlider.getModel()).setObserved(option == MatrixType.OBSERVED || option == MatrixType.CONTROL || option == MatrixType.EXPECTED);
        boolean activateOE = option == MatrixType.OE || option == MatrixType.RATIO;
        boolean isObservedOrControl = option == MatrixType.OBSERVED || option == MatrixType.CONTROL;

        colorRangeSlider.setEnabled(option == MatrixType.OBSERVED || option == MatrixType.CONTROL || activateOE);
        colorRangeSlider.setDisplayToOE(activateOE);

        if (activateOE) {
            resetOEColorRangeSlider();
        } else {
            resetRegularColorRangeSlider();
        }

        plusButton.setEnabled(activateOE || isObservedOrControl);
        minusButton.setEnabled(activateOE || isObservedOrControl);
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

        final MatrixType passOption = option;
        hic.setDisplayOption(passOption);
        //refresh();
    }

    private void safeNormalizationComboBoxActionPerformed(final ActionEvent e) {
        Runnable runnable = new Runnable() {
            public void run() {
                unsafeNormalizationComboBoxActionPerformed(e);
            }
        };
        executeLongRunningTask(runnable,"Normalization ComboBox");
    }

    private void unsafeNormalizationComboBoxActionPerformed(ActionEvent e) {
        String value = (String) normalizationComboBox.getSelectedItem();
        NormalizationType chosen = null;
        for (NormalizationType type : NormalizationType.values()) {
            if (type.getLabel().equals(value)) {
                chosen = type;
                break;
            }
        }
        final NormalizationType passChosen = chosen;
        hic.setNormalizationType(passChosen);
        //refresh();
    }

    /**
     * Utility function to execute a task in a worker thread.  The method is on MainWindow because the glassPane
     * is used to display a wait cursor and block events.
     *
     * @param runnable Thread
     * @return thread
     */

    public Future<?> executeLongRunningTask(final Runnable runnable, final String caller) {

        Callable<Object> wrapper = new Callable<Object>() {
            public Object call() throws Exception {
                MainWindow.this.showDisabledGlassPane(caller);
                try {
                    runnable.run();
                    return "done";
                } finally {
                    MainWindow.this.hideDisabledGlassPane();
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

    int i = 0, j = 0;
    private void showDisabledGlassPane(String caller) {
        //System.out.println("SA " + "" +disabledGlassPane.isValid()+" "+disabledGlassPane.isVisible()+" "+ disabledGlassPane.isValidateRoot()+" "+i+" "+caller);
        disabledGlassPane.activate("Loading...");
        //System.out.println("SB " + "" +disabledGlassPane.isValid()+" "+disabledGlassPane.isVisible()+" "+ disabledGlassPane.isValidateRoot()+" "+i++ +" "+caller);

        // TODO MSS glass pane debugging
        try {
            Thread.sleep(50);                 //1000 milliseconds is one second.
        } catch(InterruptedException ex) {
            Thread.currentThread().interrupt();
        }
    }

    private void initializeGlassPaneListening() {
        rootPane.setGlassPane(disabledGlassPane);
        disabledGlassPane.setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));
    }

    private void hideDisabledGlassPane() {//getRootPane().getContentPane()
        //System.out.println("HA " + "" +disabledGlassPane.isValid()+" "+disabledGlassPane.isVisible()+" "+ disabledGlassPane.isValidateRoot()+" "+j);
        disabledGlassPane.deactivate();
        //System.out.println("HB " + "" +disabledGlassPane.isValid()+" "+disabledGlassPane.isVisible()+" "+ disabledGlassPane.isValidateRoot()+" "+j++);

        /*
         * TODO MSS debugging

        try {
            Thread.sleep(2000);                 //1000 milliseconds is one second.
        } catch(InterruptedException ex) {
            Thread.currentThread().interrupt();
        }
         */
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

    private void resetRegularColorRangeSlider() {
        if (colorValuesToRestore != null) {
            //refreshChromosomes();
            //setInitialZoom();
            colorRangeSlider.setDisplayToBlank(false);
            colorRangeSlider.setMinimum(colorValuesToRestore[0]);
            colorRangeSlider.setMaximum(colorValuesToRestore[1]);
            colorRangeSlider.setLowerValue(colorValuesToRestore[2]);
            colorRangeSlider.setUpperValue(colorValuesToRestore[3]);
            colorRangeScaleFactor = colorValuesToRestore[4];

            //refresh();
            colorValuesToRestore = null;
        }
    }

    private void resetOEColorRangeSlider() {

        colorRangeSlider.setDisplayToBlank(false);
        if (colorValuesToRestore == null) {
            colorValuesToRestore = new int[5];
            colorValuesToRestore[0] = colorRangeSlider.getMinimum();
            colorValuesToRestore[1] = colorRangeSlider.getMaximum();
            colorValuesToRestore[2] = colorRangeSlider.getLowerValue();
            colorValuesToRestore[3] = colorRangeSlider.getUpperValue();
            colorValuesToRestore[4] = (int) colorRangeScaleFactor;
        }

        colorRangeSlider.setMinimum(-20);
        colorRangeSlider.setMaximum(20);
        colorRangeSlider.setLowerValue(-5);
        colorRangeSlider.setUpperValue(5);

    }


    private void initComponents() {

        System.out.println("Initializing Components");

        //size of the screen
        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();

        //height of the task bar
        Insets scnMax = Toolkit.getDefaultToolkit().getScreenInsets(getGraphicsConfiguration());
        int taskBarSize = scnMax.bottom;

        //available size of the screen
        //setLocation(screenSize.width - getWidth(), screenSize.height - taskBarSize - getHeight());

        Container contentPane = getContentPane();
        contentPane.setLayout(new BorderLayout());

        final JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new BorderLayout());
        contentPane.add(mainPanel, BorderLayout.CENTER);
        mainPanel.setBackground(Color.white);

        final JPanel toolbarPanel = new JPanel();
        toolbarPanel.setBorder(null);

        toolbarPanel.setLayout(new GridBagLayout());
        mainPanel.add(toolbarPanel, BorderLayout.NORTH);

        JPanel bigPanel = new JPanel();
        bigPanel.setLayout(new BorderLayout());
        bigPanel.setBackground(Color.white);

        int bigPanelWidth = screenSize.width - getWidth() - 230;
        int bigPanelHeight = screenSize.height - taskBarSize - getHeight() - 120;


        bigPanel.setPreferredSize(new Dimension(bigPanelWidth, bigPanelHeight));
        bigPanel.setMaximumSize(new Dimension(bigPanelWidth, bigPanelHeight));
        bigPanel.setMinimumSize(new Dimension(bigPanelWidth, bigPanelHeight));

        JPanel bottomPanel = new JPanel();
        bottomPanel.setBackground(Color.white);


        JMenuBar menuBar = null;
        try {
            menuBar = createMenuBar();
        } catch (Exception e) {
            e.printStackTrace();
        }
        assert menuBar != null;
        contentPane.add(menuBar, BorderLayout.NORTH);

        GridBagConstraints toolbarConstraints = new GridBagConstraints();
        toolbarConstraints.anchor = GridBagConstraints.LINE_START;
        toolbarConstraints.fill = GridBagConstraints.HORIZONTAL;
        toolbarConstraints.gridx = 0;
        toolbarConstraints.gridy = 0;
        toolbarConstraints.weightx = 0.1;

        // --- Chromosome panel ---
        JPanel chrSelectionPanel = new JPanel();
        toolbarPanel.add(chrSelectionPanel, toolbarConstraints);

        chrSelectionPanel.setBorder(LineBorder.createGrayLineBorder());

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

        chrSelectionPanel.setMinimumSize(new Dimension(200, 70));
        chrSelectionPanel.setPreferredSize(new Dimension(210, 70));

        //======== Display Option Panel ========
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
                safeDisplayOptionComboBoxActionPerformed(e);
            }
        });
        displayOptionButtonPanel.add(displayOptionComboBox);
        displayOptionPanel.add(displayOptionButtonPanel, BorderLayout.CENTER);
        displayOptionPanel.setMinimumSize(new Dimension(140, 70));
        displayOptionPanel.setPreferredSize(new Dimension(140, 70));
        displayOptionPanel.setMaximumSize(new Dimension(140, 70));

        toolbarConstraints.gridx = 1;
        toolbarConstraints.weightx = 0.1;
        toolbarPanel.add(displayOptionPanel, toolbarConstraints);
        displayOptionComboBox.setEnabled(false);

        //======== Normalization Panel ========
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
                safeNormalizationComboBoxActionPerformed(e);
            }
        });
        normalizationButtonPanel.add(normalizationComboBox);
        normalizationPanel.add(normalizationButtonPanel, BorderLayout.CENTER);
        normalizationPanel.setPreferredSize(new Dimension(180, 70));
        normalizationPanel.setMinimumSize(new Dimension(140, 70));


        toolbarConstraints.gridx = 2;
        toolbarConstraints.weightx = 0.1;
        toolbarPanel.add(normalizationPanel, toolbarConstraints);
        normalizationComboBox.setEnabled(false);

        //======== Resolution Panel ========
        hiCPanel = new JPanel();
        hiCPanel.setBackground(Color.white);
        hiCPanel.setLayout(new HiCLayout());
        bigPanel.add(hiCPanel,BorderLayout.CENTER);

        JPanel wrapGapPanel = new JPanel();
        wrapGapPanel.setBackground(Color.white);
        wrapGapPanel.setMaximumSize(new Dimension(5, 5));
        wrapGapPanel.setMinimumSize(new Dimension(5, 5));
        wrapGapPanel.setPreferredSize(new Dimension(5, 5));
        wrapGapPanel.setBorder(LineBorder.createBlackLineBorder());
        bigPanel.add(wrapGapPanel, BorderLayout.EAST);



        // splitPanel.insertPane(hiCPanel, 0);
        // splitPanel.setBackground(Color.white);

        //---- rulerPanel2 ----
        JPanel topPanel = new JPanel();
        topPanel.setBackground(Color.green);
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
        //Dimension screenDimension = Toolkit.getDefaultToolkit().getScreenSize();
        //int panelSize = screenDimension.height - 210;

        int panelWidth = screenSize.width - getWidth() - 300;
        int panelHeight = screenSize.height - taskBarSize - getHeight();



        System.err.println("Window W: " + panelWidth + " H" + panelHeight);

        JPanel wrapHeatmapPanel = new JPanel(new BorderLayout());
        wrapHeatmapPanel.setMaximumSize(new Dimension(panelWidth, panelHeight));
        wrapHeatmapPanel.setMinimumSize(new Dimension(panelWidth, panelHeight));
        wrapHeatmapPanel.setPreferredSize(new Dimension(panelWidth, panelHeight));
        wrapHeatmapPanel.setBackground(Color.BLUE);
        wrapHeatmapPanel.setVisible(true);

        heatmapPanel = new HeatmapPanel(this, hic);
        heatmapPanel.setMaximumSize(new Dimension(panelWidth-5, panelHeight-5));
        heatmapPanel.setMinimumSize(new Dimension(panelWidth-5, panelHeight-5));
        heatmapPanel.setPreferredSize(new Dimension(panelWidth-5, panelHeight-5));
        heatmapPanel.setBackground(Color.white);

        wrapHeatmapPanel.add(heatmapPanel, BorderLayout.CENTER);

        //hiCPanel.setMaximumSize(new Dimension(panelWidth, panelHeight));
        //hiCPanel.setMinimumSize(new Dimension(panelWidth, panelHeight));
        //hiCPanel.setPreferredSize(new Dimension(panelWidth, panelHeight));

        hiCPanel.add(wrapHeatmapPanel, BorderLayout.CENTER);

        //======== Resolution Slider Panel ========

        // Resolution  panel
        resolutionSlider = new ResolutionControl(hic, this, heatmapPanel);
        resolutionSlider.setPreferredSize(new Dimension(200, 70));
        resolutionSlider.setMinimumSize(new Dimension(150, 70));

        toolbarConstraints.gridx = 3;
        toolbarConstraints.weightx = 0.1;
        toolbarPanel.add(resolutionSlider, toolbarConstraints);

        //======== Color Range Panel ========

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
        colorRangeSlider.setDisplayToBlank(true);

        //---- colorRangeLabel ----
        colorRangeLabel = new JLabel("Color Range");
        MouseListener l = new MouseAdapter() {
            Font original;

            @Override
            public void mouseEntered(MouseEvent e) {
                if(colorRangeSlider.isEnabled()) {
                    original = e.getComponent().getFont();
                    Map attributes = original.getAttributes();
                    attributes.put(TextAttribute.UNDERLINE, TextAttribute.UNDERLINE_ON);
                    e.getComponent().setFont(original.deriveFont(attributes));
                }
            }


            @Override
            public void mouseExited(MouseEvent e) {
                //if (colorRangeSlider.isEnabled()) {
                e.getComponent().setFont(original);
                //}
            }

        };
        colorRangeLabel.addMouseListener(l);

        colorRangeLabel.setHorizontalAlignment(SwingConstants.CENTER);
        colorRangeLabel.setToolTipText("Range of color scale in counts per mega-base squared.");
        colorRangeLabel.setHorizontalTextPosition(SwingConstants.CENTER);

        colorRangeLabel.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                if (e.isPopupTrigger() && colorRangeSlider.isEnabled()) {
                    setColorRangeSliderVisible(false);
                    ColorRangeDialog rangeDialog = new ColorRangeDialog(MainWindow.this, colorRangeSlider, colorRangeScaleFactor, hic.getDisplayOption() == MatrixType.OBSERVED);
                    rangeDialog.setVisible(true);
                }
            }

            @Override
            public void mouseClicked(MouseEvent e) {
                //No double click here...
                if (e.getClickCount() == 1) {
                    if (colorRangeSlider.isEnabled()) {
                        setColorRangeSliderVisible(false);
                        ColorRangeDialog rangeDialog = new ColorRangeDialog(MainWindow.this, colorRangeSlider, colorRangeScaleFactor, hic.getDisplayOption() == MatrixType.OBSERVED);
                        rangeDialog.setVisible(true);
                    }}
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
        resetRegularColorRangeSlider();

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
                if (hic.getDisplayOption() == MatrixType.OE || hic.getDisplayOption() == MatrixType.RATIO) {
                    colorRangeSlider.setMinimum(-colorRangeSlider.getMaximum());
                    colorRangeSlider.setLowerValue(-colorRangeSlider.getUpperValue());
                }
                colorRangeSliderUpdateToolTip();
            }
        });

        minusButton = new JideButton();
        minusButton.setIcon(new ImageIcon(getClass().getResource("/images/zoom-minus.png")));
        minusButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                //Set limit to maximum range:
                int newMax = colorRangeSlider.getMaximum() / 2;
                if (newMax > 0) {
                    colorRangeSlider.setMaximum(newMax);
                    if (hic.getDisplayOption() == MatrixType.OE || hic.getDisplayOption() == MatrixType.RATIO) {
                        colorRangeSlider.setMinimum(-newMax);
                        colorRangeSlider.setLowerValue(-colorRangeSlider.getUpperValue());
                    }
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
        toolbarConstraints.gridx = 4;
        toolbarConstraints.weightx = 0.5;
        toolbarPanel.add(colorRangePanel, toolbarConstraints);

        //======== Goto Panel ========
        JPanel goPanel = new JPanel();
        goPanel.setBackground(new Color(238, 238, 238));
        goPanel.setBorder(LineBorder.createGrayLineBorder());
        goPanel.setLayout(new BorderLayout());
        JPanel goLabelPanel = new JPanel();
        goLabelPanel.setBackground(new Color(204, 204, 204));
        goLabelPanel.setLayout(new BorderLayout());

        JLabel goLabel = new JLabel("Goto");
        goLabel.setHorizontalAlignment(SwingConstants.CENTER);
        goLabelPanel.add(goLabel, BorderLayout.CENTER);
        goPanel.add(goLabelPanel, BorderLayout.PAGE_START);

        JPanel goButtonPanel = new JPanel();
        goButtonPanel.setBackground(new Color(238, 238, 238));
        goButtonPanel.setLayout(new BoxLayout(goButtonPanel, BoxLayout.X_AXIS));


        positionChrTop = new JTextField();
        positionChrTop.setFont(new Font("Arial", Font.ITALIC, 10));
        positionChrTop.setEnabled(false);
        positionChrTop.addActionListener(new ActionListener() {

            public void actionPerformed(ActionEvent e) {

                parsePositionText();

            }
        });

        //positionChrTop.setPreferredSize(new Dimension(10, 10));


        positionChrLeft = new JTextField();
        positionChrLeft.setFont(new Font("Arial", Font.ITALIC, 10));
        positionChrLeft.setEnabled(false);
        positionChrLeft.addActionListener(new ActionListener() {

            public void actionPerformed(ActionEvent e) {

                parsePositionText();

            }
        });

        //positionChrLeft.setPreferredSize(new Dimension(10, 10));
        JPanel goPositionPanel = new JPanel();
        goPositionPanel.setLayout(new BorderLayout());


        goPositionPanel.add(positionChrTop, BorderLayout.PAGE_START);
        goPositionPanel.add(positionChrLeft, BorderLayout.PAGE_END);

        goButtonPanel.add(goPositionPanel, BorderLayout.PAGE_START);

        goButton = new JideButton();
        goButton.setEnabled(false);
        goButton.setIcon(new ImageIcon(getClass().getResource("/toolbarButtonGraphics/general/Refresh24.gif")));
        goButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                parsePositionText();
            }
        });

        goButtonPanel.add(goButton);
        goPanel.add(goButtonPanel);


        //goPanel.setBackground(Color.white);
        //goPanel.setBorder(LineBorder.createBlackLineBorder());
        goPanel.setMinimumSize(new Dimension(100, 70));
        goPanel.setPreferredSize(new Dimension(120, 70));
        goPanel.setMaximumSize(new Dimension(200, 70));

        toolbarConstraints.gridx = 5;
        toolbarConstraints.weightx = 0.25;
        toolbarPanel.add(goPanel, toolbarConstraints);
        // not sure this is working
        //toolbarPanel.setPreferredSize(new Dimension(panelHeight,100));
        toolbarPanel.setEnabled(false);


        //======== Right side panel ========

        JPanel rightSidePanel = new JPanel(new BorderLayout());//(new BorderLayout());
        rightSidePanel.setBackground(Color.white);
        rightSidePanel.setPreferredSize(new Dimension(210, 1000));
        rightSidePanel.setMaximumSize(new Dimension(10000, 10000));

        //======== Bird's view mini map ========

        JPanel thumbPanel = new JPanel();
        thumbPanel.setLayout(new BorderLayout());

        //---- thumbnailPanel ----
        thumbnailPanel = new ThumbnailPanel(this, hic);
        thumbnailPanel.setBackground(Color.white);
        thumbnailPanel.setMaximumSize(new Dimension(210, 210));
        thumbnailPanel.setMinimumSize(new Dimension(210, 210));
        thumbnailPanel.setPreferredSize(new Dimension(210, 210));

//        JPanel gapPanel = new JPanel();
//        gapPanel.setMaximumSize(new Dimension(1, 1));
//        rightSidePanel.add(gapPanel,BorderLayout.WEST);
        thumbPanel.add(thumbnailPanel, BorderLayout.CENTER);
        thumbPanel.setBackground(Color.white);
        rightSidePanel.add(thumbPanel, BorderLayout.NORTH);

        //========= mouse hover text ======
        JPanel tooltipPanel = new JPanel(new BorderLayout());
        tooltipPanel.setBackground(Color.white);
        tooltipPanel.setPreferredSize(new Dimension(210, 490));
        mouseHoverTextPanel = new JEditorPane();
        mouseHoverTextPanel.setEditable(false);
        mouseHoverTextPanel.setContentType("text/html");
        mouseHoverTextPanel.setFont(new Font("sans-serif", 0, 20));

        mouseHoverTextPanel.setBackground(Color.white);
        mouseHoverTextPanel.setBorder(null);
        int mouseTextY = rightSidePanel.getBounds().y + rightSidePanel.getBounds().height;

        Dimension prefSize = new Dimension(210, 490);
        mouseHoverTextPanel.setPreferredSize(prefSize);

        JScrollPane tooltipScroller = new JScrollPane(mouseHoverTextPanel);
        tooltipScroller.setBackground(Color.white);
        tooltipScroller.setBorder(null);

        tooltipPanel.setPreferredSize(new Dimension(210, 500));
        tooltipPanel.add(tooltipScroller);
        tooltipPanel.setBounds(new Rectangle(new Point(0, mouseTextY), prefSize));
        tooltipPanel.setBackground(Color.white);
        tooltipPanel.setBorder(null);

        rightSidePanel.add(tooltipPanel, BorderLayout.CENTER);
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
        mainPanel.add(bigPanel, BorderLayout.CENTER);
        mainPanel.add(rightSidePanel, BorderLayout.EAST);
        //mainPanel.add(bottomPanel, BorderLayout.SOUTH);

        // splitPanel.insertPane(rightSidePanel, 1);
        // hiCPanel.add(rightSidePanel, BorderLayout.EAST);

        initializeGlassPaneListening();

        initProperties();
    }

    public void initProperties(){
        try {
            String url = System.getProperty("jnlp.loadMenu");
            if (url == null) {
                url = "http://hicfiles.tc4ga.com/juicebox.properties";
            }
            InputStream is = ParsingUtils.openInputStream(url);
            properties = new Properties();
            if (is != null) {
                properties.load(is);
            }
        } catch (Exception error) {
            JOptionPane.showMessageDialog(this, "Can't find properties file for loading list", "Error", JOptionPane.ERROR_MESSAGE);
        }
    }


    public void setPositionChrLeft(String newPositionDate) {
        positionChrLeft.setText(newPositionDate);
    }


    public void setPositionChrTop(String newPositionDate) {
        positionChrTop.setText(newPositionDate);
    }


    public void parsePositionText() {
        //Expected format: <chr>:<start>-<end>:<resolution>

        String delimiters = "\\s+|:\\s*|\\-\\s*";
        String dashDelimiters = "\\s+|\\-\\s*";
        java.lang.Integer outBinSize = 0;
        Long outBinLeft = 0L;
        Long outBinTop = 0L;
        Long topStart = 0L;
        Long topEnd = 0L;
        Long leftStart = 0L;
        Long leftEnd = 0L;

        String[] leftChrTokens = positionChrLeft.getText().split(delimiters);
        String[] topChrTokens = positionChrTop.getText().split(delimiters);
        String[] leftDashChrTokens = positionChrLeft.getText().split(dashDelimiters);
        String[] topDashChrTokens = positionChrTop.getText().split(dashDelimiters);

        String resolutionUnits = "BP";
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
            positionChrTop.setBackground(Color.yellow);
            return;
        }
        try {
            TopChrInt = Integer.parseInt(TopChrName);
            //todo - replace with actual chromosome range
            if (TopChrInt > 22) {
                positionChrTop.setBackground(Color.yellow);
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
                positionChrTop.setBackground(Color.yellow);
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
            positionChrLeft.setBackground(Color.yellow);
            return;
        }
        try {
            LeftChrInt = Integer.parseInt(LeftChrName);

            //todo - replace with actual chromosome range
            if (LeftChrInt > 22) {
                positionChrLeft.setBackground(Color.yellow);
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
                positionChrLeft.setBackground(Color.yellow);
                return;
            }
        }

        //Read positions:
        if (topChrTokens.length > 2 && topDashChrTokens.length > 1) {
            //Make sure values are numerical:
            try {
                Long.parseLong(topChrTokens[1].replaceAll(",", ""));
            } catch (Exception e) {
                positionChrTop.setBackground(Color.yellow);
                return;
            }
            try {
                Long.parseLong(topChrTokens[2].replaceAll(",", ""));
            } catch (Exception e) {
                positionChrLeft.setBackground(Color.yellow);
                return;
            }
            topStart = Long.min(Long.valueOf(topChrTokens[1].replaceAll(",", "")), Long.valueOf(topChrTokens[2].replaceAll(",", "")));
            topEnd = Long.max(Long.valueOf(topChrTokens[1].replaceAll(",", "")), Long.valueOf(topChrTokens[2].replaceAll(",", "")));
            outBinTop = topStart + ((topEnd - topStart) / 2);

        } else if (topChrTokens.length > 1) {
            outBinTop = Long.valueOf(topChrTokens[1].replaceAll(",", ""));
        }


        if (leftChrTokens.length > 2 && leftDashChrTokens.length > 1) {
            leftStart = Long.min(Long.valueOf(leftChrTokens[1].replaceAll(",", "")), Long.valueOf(leftChrTokens[2].replaceAll(",", "")));
            leftEnd = Long.max(Long.valueOf(leftChrTokens[1].replaceAll(",", "")), Long.valueOf(leftChrTokens[2].replaceAll(",", "")));
            outBinLeft = leftStart + ((leftEnd - leftStart) / 2);
        } else if (topChrTokens.length > 1) {
            //Make sure values are numerical: TODO parsing topChr but later getting value of leftChr - is this correct? seems like a typo
            try {
                Long.parseLong(topChrTokens[1].replaceAll(",", ""));
            } catch (Exception e) {
                positionChrTop.setBackground(Color.yellow);
                return;
            }
            outBinLeft = Long.valueOf(leftChrTokens[1].replaceAll(",", ""));
        }

        //Read resolution:
        if (topChrTokens.length > 3  || (topDashChrTokens.length == 1 && topChrTokens.length > 2)) {
            if(topDashChrTokens.length == 1)
            {
                outBinSize = hic.validteBinSize(topChrTokens[2].toLowerCase());
                if(outBinSize != null && topChrTokens[2].toLowerCase().contains("f")){
                    resolutionUnits = "FRAG";
                }
                else{
                    positionChrTop.setBackground(Color.yellow);
                    return;
                }
            }
            else if (topChrTokens.length > 3)
            {
                outBinSize = hic.validteBinSize(topChrTokens[3].toLowerCase());
                if(outBinSize != null && topChrTokens[3].toLowerCase().contains("f")){
                    resolutionUnits = "FRAG";
                }
                else if (outBinSize == null)
                {
                    positionChrTop.setBackground(Color.yellow);
                    return;
                }
            }
        } else if (leftChrTokens.length > 3  || (leftDashChrTokens.length == 1  && leftChrTokens.length > 2)) {
            if(leftDashChrTokens.length == 1)
            {
                outBinSize = hic.validteBinSize(leftChrTokens[2].toLowerCase());
                if(outBinSize != null && leftChrTokens[2].toLowerCase().contains("f")){
                    resolutionUnits = "FRAG";
                }
                else if (outBinSize == null)
                {
                    positionChrLeft.setBackground(Color.yellow);
                    return;
                }
            }
            else if (leftChrTokens.length > 3)
            {
                outBinSize = hic.validteBinSize(leftChrTokens[3].toLowerCase());
                if(outBinSize != null && leftChrTokens[3].toLowerCase().contains("f")){
                    resolutionUnits = "FRAG";
                }
                else{
                    positionChrLeft.setBackground(Color.yellow);
                    return;
                }
            }
        } else if (hic.getZoom().getBinSize() != 0) {
            outBinSize = hic.validteBinSize(String.valueOf(hic.getZoom().getBinSize()));
            if(outBinSize != null) {
                resolutionUnits = hic.getZoom().getUnit().toString();
            }
        }

        positionChrTop.setBackground(Color.white);
        positionChrLeft.setBackground(Color.white);

        if(outBinSize == null) {
            // If bin size is not valid, set to max bin size:
            outBinSize = 250000;
        }

        hic.setState(TopChrName, LeftChrName, resolutionUnits, outBinSize, 0, 0, hic.getScaleFactor());
        if (outBinTop > 0 && outBinLeft > 0) {
            hic.centerBP(Math.round(outBinTop), Math.round(outBinLeft));
        }

        //We might end with ALL->All view, make sure normalization state is updates acordingly...
        setNormalizationDisplayState();
    }


    private void colorRangeSliderUpdateToolTip() {
        if (hic.getDisplayOption() == MatrixType.OBSERVED ||
                hic.getDisplayOption() == MatrixType.CONTROL ||
                hic.getDisplayOption() == MatrixType.OE || hic.getDisplayOption() == MatrixType.RATIO) {

            int iMin = colorRangeSlider.getMinimum();
            int lValue = colorRangeSlider.getLowerValue();
            int uValue = colorRangeSlider.getUpperValue();
            int iMax = colorRangeSlider.getMaximum();

            /*
            colorRangeSlider.setToolTipText("<html>Range: " + (int) (iMin / colorRangeScaleFactor) + " "

                    + (int) (iMax / colorRangeScaleFactor) + "<br>Showing: " +
                    (int) (lValue / colorRangeScaleFactor) + " "
                    + (int) (uValue / colorRangeScaleFactor)
                    + "</html>");
            */

            Font f = FontManager.getFont(8);

            Hashtable<Integer, JLabel> labelTable = new Hashtable<Integer, JLabel>();


            if (hic.getDisplayOption() == MatrixType.OE || hic.getDisplayOption() == MatrixType.RATIO) {
                colorRangeSlider.setToolTipText("Log Enrichment Values");
            } else {
                colorRangeSlider.setToolTipText("Observed Counts");
            }

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

    public void setColorRangeSliderVisible(boolean state)
    {
        plusButton.setEnabled(state);
        minusButton.setEnabled(state);
        colorRangeSlider.setEnabled(state);
        if(state){
            colorRangeLabel.setForeground(Color.BLUE);
        }
        else {
            colorRangeLabel.setForeground(Color.BLACK);
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

        recentMapMenu = new RecentMenu("Open Recent", recentMapListMaxItems, recentMapEntityNode) {

            private static final long serialVersionUID = 3412L;

            public void onSelectPosition(String mapPath) {
                String delimiter = "@@";
                String[] temp;
                temp = mapPath.split(delimiter);
                initProperties();
                loadFromRecentActionPerformed((temp[1]), (temp[0]), false);
            }
        };
        recentMapMenu.setMnemonic('R');


        fileMenu.add(recentMapMenu);



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

        JMenuItem creditsMenu = new JMenuItem();
        creditsMenu.setText("About");
        creditsMenu.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                ImageIcon icon = new ImageIcon(getClass().getResource("/images/juicebox.png"));
                JLabel iconLabel = new JLabel(icon);
                JPanel iconPanel = new JPanel(new GridBagLayout());
                iconPanel.add(iconLabel);

                JPanel textPanel= new JPanel(new GridLayout(0, 1));
                textPanel.add(new JLabel("<html><center>" +
                        "<h2 style=\"margin-bottom:30px;\" class=\"header\">" +
                        "Juicebox: Visualization software for Hi-C data" +
                        "</h2>" +
                        "</center>" +
                        "<p>" +
                        "Juicebox is Aiden Lab's software for visualizing data from proximity ligation experiments, such as Hi-C, 5C, and Chia-PET.<br>" +
                        "Juicebox was created by Neva Durand, Jim Robinson, and Erez Aiden. Ongoing development work is carried out by Neva Durand,<br>" +
                        "Muhammad Shamim, and Ido Machol. <br><br>" +
                        "Copyright © 2014. Aiden Lab" +
                        "<br><br>" +
                        "If you use Juicebox in your research, please cite:<br><br>" +
                        "<strong>Suhas S.P. Rao*, Miriam H. Huntley*, Neva C. Durand, Elena K. Stamenova, Ivan D. Bochkov, James T. Robinson,<br>"+
                        "Adrian L. Sanborn, Ido Machol, Arina D. Omer, Eric S. Lander, Erez Lieberman Aiden.<br>" +
                        "\"A 3D Map of the Human Genome at Kilobase Resolution Reveals Principles of Chromatin Looping.\" <em>Cell</em> 159, 2014.</strong>" +
                        "</p></html>"));

                JPanel mainPanel = new JPanel(new BorderLayout());
                mainPanel.add(textPanel);
                mainPanel.add(iconPanel, BorderLayout.WEST);

                JOptionPane.showMessageDialog(null, mainPanel, "About", JOptionPane.PLAIN_MESSAGE);//INFORMATION_MESSAGE
            }
        });
        fileMenu.add(creditsMenu);

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

            private static final long serialVersionUID = 1234L;

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

    public String getToolTip() {
        return mouseHoverTextPanel.getText();
    }

    public boolean isTooltipAllowedToUpdated() {
        return tooltipAllowedToUpdated;
    }

    public void toggleToolTipUpdates(boolean tooltipAllowedToUpdated) {
        this.tooltipAllowedToUpdated = tooltipAllowedToUpdated;
    }
}


