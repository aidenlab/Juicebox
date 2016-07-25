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

package juicebox.mapcolorui;

import com.jidesoft.swing.JideButton;
import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.data.MatrixZoomData;
import juicebox.gui.SuperAdapter;
import juicebox.windowui.HiCZoom;
import org.broad.igv.ui.FontManager;
import org.broad.igv.ui.util.MessageUtils;

import javax.swing.*;
import javax.swing.border.LineBorder;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.font.TextAttribute;
import java.util.Dictionary;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class ResolutionControl extends JPanel {
    private static final long serialVersionUID = -5982918928089196379L;
    private final ImageIcon lockOpenIcon;
    private final ImageIcon lockIcon;
    private final HiC hic;
    private final HeatmapPanel heatmapPanel;
    private final JideButton lockButton;
    private final JLabel resolutionLabel;
    private final Map<Integer, HiCZoom> idxZoomMap = new ConcurrentHashMap<Integer, HiCZoom>();
    private final Map<Integer, String> bpLabelMap;
    private final HiCZoom pearsonZoom = new HiCZoom(HiC.Unit.BP, 500000);
    public HiC.Unit unit = HiC.Unit.BP;
    private boolean resolutionLocked = false;
    private JSlider resolutionSlider;
    private int lastValue = 0;

    {
        bpLabelMap = new Hashtable<Integer, String>();
        for (int i = 0; i < HiCGlobals.bpBinSizes.length; i++) {
            bpLabelMap.put(HiCGlobals.bpBinSizes[i], HiCGlobals.bpBinSizeNames[i]);
        }

        // 1kb not in global list
        bpLabelMap.put(1000, "1 KB");
    }

    public ResolutionControl(final SuperAdapter superAdapter) {

        this.hic = superAdapter.getHiC();
        this.heatmapPanel = superAdapter.getHeatmapPanel();

        this.setBorder(LineBorder.createGrayLineBorder());
        this.setLayout(new BorderLayout());

        resolutionLabel = new JLabel(getUnitLabel());
        resolutionLabel.setHorizontalAlignment(SwingConstants.CENTER);
        resolutionLabel.setBackground(HiCGlobals.backgroundColor);

        JPanel resolutionLabelPanel = new JPanel();
        resolutionLabelPanel.setBackground(HiCGlobals.backgroundColor);
        resolutionLabelPanel.setLayout(new BorderLayout());
        resolutionLabelPanel.add(resolutionLabel, BorderLayout.CENTER);

        // TODO not working
        // supposed to underline "resolution text" but why? is this an important gui issue?
        resolutionLabelPanel.addMouseListener(new MouseAdapter() {
            private Font original;

            @SuppressWarnings({"unchecked", "rawtypes"})
            @Override
            public void mouseEntered(MouseEvent e) {
                original = e.getComponent().getFont();
                if (resolutionSlider.isEnabled()) {
                    original = e.getComponent().getFont();
                    Map attributes = original.getAttributes();
                    attributes.put(TextAttribute.UNDERLINE, TextAttribute.UNDERLINE_ON);
                    e.getComponent().setFont(original.deriveFont(attributes));
                }
            }

            @Override
            public void mouseExited(MouseEvent e) {
                e.getComponent().setFont(original);
            }

        });


        resolutionLabelPanel.addMouseListener(new MouseAdapter() {

            @Override
            public void mousePressed(MouseEvent e) {
                if (e.isPopupTrigger() && resolutionSlider.isEnabled() && hic != null && hic.getDataset() != null) {
                    if (hic.getDataset().hasFrags()) {
                        processClick();
                    }
                }
            }

            @Override
            public void mouseClicked(MouseEvent e) {
                //No double click here...
                if (e.getClickCount() == 1 && resolutionSlider.isEnabled() && hic != null && hic.getDataset() != null) {
                    if (hic.getDataset().hasFrags()) {
                        processClick();
                    }
                }
            }

            private void processClick() {
                unit = (unit == HiC.Unit.FRAG ? HiC.Unit.BP : HiC.Unit.FRAG);
                resolutionLabel.setText(getUnitLabel());
                Runnable runnable = new Runnable() {
                    public void run() {
                        reset();
                        superAdapter.refresh(); // necessary to correct BP/FRAG switching all red box
                    }
                };
                superAdapter.executeLongRunningTask(runnable, "Resolution switched");
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
                final MatrixZoomData zd;
                try {
                    zd = hic.getZd();
                } catch (Exception ex) {
                    return;
                }
                if (hic == null || hic.getMatrix() == null || zd == null || resolutionSlider.getValueIsAdjusting())
                    return;
                final ChangeEvent eF = e;
                Runnable runnable = new Runnable() {
                    public void run() {
                        unsafeStateChanged(eF, zd);
                    }
                };
                superAdapter.executeLongRunningTask(runnable, "Resolution slider change");
            }

            private void unsafeStateChanged(ChangeEvent e, MatrixZoomData zd) {

                int idx = resolutionSlider.getValue();

                final HiCZoom zoom = idxZoomMap.get(idx);
                if (zoom == null) return;

                if (zoom.getBinSize() == hic.getXContext().getZoom().getBinSize() &&
                        zoom.getUnit() == hic.getXContext().getZoom().getUnit()) return;

                if (zoom.getBinSize() < HiCGlobals.MAX_PEARSON_ZOOM && hic.isInPearsonsMode()) {
                    MessageUtils.showMessage("Pearson's matrix is not available at this resolution,\n" +
                            "please use a resolution lower than 500 KB.");
                    setZoom(pearsonZoom);
                    return;
                }

                if (hic.getXContext() != null) {

                    double scaledXWidth = heatmapPanel.getWidth() / hic.getScaleFactor();
                    double scaledYHeight = heatmapPanel.getHeight() / hic.getScaleFactor();
                    double centerBinX = hic.getXContext().getBinOrigin() + scaledXWidth / 2;
                    double centerBinY = hic.getYContext().getBinOrigin() + scaledYHeight / 2;
                    int xGenome = zd.getXGridAxis().getGenomicMid(centerBinX);
                    int yGenome = zd.getYGridAxis().getGenomicMid(centerBinY);

                    // this to center zooming when there is lots of whitespace in the margins
                    try {
                        if (scaledXWidth > hic.getZd().getXGridAxis().getBinCount()) {
                            xGenome = hic.getXContext().getChrLength() / 2;
                        }
                    } catch (Exception ee) {
                    }

                    try {
                        if (scaledYHeight > hic.getZd().getYGridAxis().getBinCount()) {
                            yGenome = hic.getYContext().getChrLength() / 2;
                        }
                    } catch (Exception ee) {
                    }

                    if (zd == null) {
                        hic.unsafeActuallySetZoomAndLocation("", "", zoom, 0, 0, -1, true, HiC.ZoomCallType.STANDARD,
                                true);
                    } else {

                        if (hic.unsafeActuallySetZoomAndLocation("", "", zoom, xGenome, yGenome, -1, true,
                                HiC.ZoomCallType.STANDARD, true)) {
                            lastValue = resolutionSlider.getValue();
                        } else {
                            resolutionSlider.setValue(lastValue);
                        }
                    }
                }
            }

        });
        setEnabled(false);
    }

    private String getUnitLabel() {
        return unit == HiC.Unit.FRAG ? "Resolution (Frag)" : "Resolution (BP)";
    }

    public void setEnabled(boolean enabled) {
        resolutionSlider.setEnabled(enabled);
        lockButton.setEnabled(enabled);
    }

    /**
     * Called when a new dataset is loaded, or when units are switched bp<->frag
     */
    public void reset() {
        if (hic == null || hic.getDataset() == null) return;

        if (heatmapPanel != null) heatmapPanel.reset();

        int currentIdx = resolutionSlider.getValue();

        List<HiCZoom> binSizes = unit == HiC.Unit.BP ?
                hic.getDataset().getBpZooms() : hic.getDataset().getFragZooms();

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


    private String sizeToLabel(int binSize) {

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
        if (unit != newZoom.getUnit()) {
            unit = newZoom.getUnit();
            reset();
        }
        resolutionLabel.setText(getUnitLabel());
        resolutionLabel.setForeground(Color.BLUE);
        for (Map.Entry<Integer, HiCZoom> entry : idxZoomMap.entrySet()) {
            if (entry.getValue().equals(newZoom)) {
                resolutionSlider.setValue(entry.getKey());
            }
        }
    }

    public boolean isResolutionLocked() {
        return resolutionLocked;
    }
}
