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

package juicebox.mapcolorui;

import com.jidesoft.swing.JideButton;
import juicebox.HiC;
import juicebox.MainWindow;
import juicebox.windowui.HiCZoom;
import org.broad.igv.ui.FontManager;

import javax.swing.*;
import javax.swing.border.LineBorder;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.font.TextAttribute;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Map;

public class ResolutionControl extends JPanel {
    static final long serialVersionUID = 42L;
    private final ImageIcon lockOpenIcon;
    private final ImageIcon lockIcon;
    private final HiC hic;
    private final HeatmapPanel heatmapPanel;
    private final JideButton lockButton;
    private final JLabel resolutionLabel;
    private final Map<Integer, HiCZoom> idxZoomMap = new HashMap<Integer, HiCZoom>();
    private final Map<Integer, String> bpLabelMap;
    public HiC.Unit unit = HiC.Unit.BP;

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
        bpLabelMap.put(1000, "1 KB");
    }

    private boolean resolutionLocked = false;
    private JSlider resolutionSlider;
    private int lastValue = 0;

    public ResolutionControl(final HiC hic, final MainWindow mainWindow, final HeatmapPanel heatmapPanel) {

        this.hic = hic;
        this.heatmapPanel = heatmapPanel;

        this.setBorder(LineBorder.createGrayLineBorder());
        this.setLayout(new BorderLayout());

        resolutionLabel = new JLabel(getUnitLabel());
        resolutionLabel.setHorizontalAlignment(SwingConstants.CENTER);
        resolutionLabel.setBackground(new Color(204, 204, 204));
//        resolutionLabel.addMouseListener(new MouseAdapter() {
//            Font original;
//
//            @Override
//            public void mouseEntered(MouseEvent e) {
//                if (resolutionLabel.isEnabled()) {
//                    original = e.getComponent().getFont();
//                    Map attributes = original.getAttributes();
//                    attributes.put(TextAttribute.UNDERLINE, TextAttribute.UNDERLINE_ON);
//                    e.getComponent().setFont(original.deriveFont(attributes));
//                }
//            }
//
//            @Override
//            public void mouseExited(MouseEvent e) {
//                e.getComponent().setFont(original);
//            }
//        });
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
                        mainWindow.refresh();
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
                if (hic == null || hic.getMatrix() == null || hic.getZd() == null || resolutionSlider.getValueIsAdjusting()) return;
                final ChangeEvent eF = e;
                Runnable runnable = new Runnable() {
                    public void run() {
                        unsafeStateChanged(eF);
                    }
                };
                mainWindow.executeLongRunningTask(runnable, "Resolution slider change");
                //runnable.run();
            }

            private void unsafeStateChanged(ChangeEvent e){



                    int idx = resolutionSlider.getValue();

                    final HiCZoom zoom = idxZoomMap.get(idx);
                    if (zoom == null) return;

                    if (hic.getXContext() != null) {

                        hic.setScaleFactor(1.0);
                        hic.setScaleFactor(1.0);

                        double centerBinX = hic.getXContext().getBinOrigin() + (heatmapPanel.getWidth() / (2 * hic.getScaleFactor()));
                        double centerBinY = hic.getYContext().getBinOrigin() + (heatmapPanel.getHeight() / (2 * hic.getScaleFactor()));
                        final int xGenome = hic.getZd().getXGridAxis().getGenomicMid(centerBinX);
                        final int yGenome = hic.getZd().getYGridAxis().getGenomicMid(centerBinY);

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

        java.util.List<HiCZoom> binSizes =
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
