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

package juicebox.track;

import juicebox.HiC;
import juicebox.MainWindow;
import juicebox.data.Dataset;
import juicebox.encode.EncodeFileBrowser;
import juicebox.encode.EncodeFileRecord;
import org.apache.log4j.Logger;
import org.broad.igv.track.AttributeManager;
import org.broad.igv.ui.util.MessageUtils;
import org.broad.igv.util.ResourceLocator;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.io.IOException;
import java.util.*;
import java.util.List;

/**
 * @author jrobinso
 *         Date: 3/13/14
 *         Time: 9:57 AM
 */
public class LoadEncodeAction extends AbstractAction {

    private static final Logger log = Logger.getLogger(LoadEncodeAction.class);
    private static final long serialVersionUID = 3033491284874081821L;
    private static final Map<String, Color> colors;

    static {
        colors = new HashMap<String, Color>();
        colors.put("H3K27AC", new Color(200, 0, 0));
        colors.put("H3K27ME3", new Color(200, 0, 0));
        colors.put("H3K36ME3", new Color(0, 0, 150));
        colors.put("H3K4ME1", new Color(0, 150, 0));
        colors.put("H3K4ME2", new Color(0, 150, 0));
        colors.put("H3K4ME3", new Color(0, 150, 0));
        colors.put("H3K9AC", new Color(100, 0, 0));
        colors.put("H3K9ME1", new Color(100, 0, 0));
    }

    private final MainWindow mainWindow;
    private final HiC hic;
    private String genome;
    private HashSet<ResourceLocator> loadedLocators;

    public LoadEncodeAction(String s, MainWindow mainWindow, HiC hic) {
        super(s);
        this.mainWindow = mainWindow;
        this.hic = hic;
        this.genome = null;

    }

    public void checkEncodeBoxes(String track) {
        try {
            Dataset ds = hic.getDataset();
            genome = ds.getGenomeId();
            EncodeFileBrowser encodeFileBrowser = EncodeFileBrowser.getInstance(genome);
            assert encodeFileBrowser != null;
            encodeFileBrowser.checkEncodeTracks(track);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    @Override
    public void actionPerformed(ActionEvent e) {
        if (hic.getDataset() == null || hic.getDataset().getGenomeId() == null) {
            JOptionPane.showMessageDialog(mainWindow, "File must be loaded to load annotations", "Error", JOptionPane.ERROR_MESSAGE);
            return;
        }

        genome = hic.getDataset().getGenomeId();

        hic.setEncodeAction(this);
        String[] visibleAttributes = {"dataType", "cell", "antibody", "lab"};
        try {
            EncodeFileBrowser browser = EncodeFileBrowser.getInstance(genome);

            if (browser == null) {
                MessageUtils.showMessage("Encode tracks are not available for " + genome);
                return;
            }

            browser.setVisible(true);
            if (browser.isCanceled()) return;

            List<EncodeFileRecord> records = browser.getSelectedRecords();

            safeLoadENCODETracks(records, visibleAttributes);

        } catch (IOException exc) {
            log.error("Error opening Encode browser", exc);
        }
    }

    private void safeLoadENCODETracks(final List<EncodeFileRecord> records, final String[] visibleAttributes) {

        Runnable runnable = new Runnable() {
            @Override
            public void run() {
                unsafeLoadENCODETracks(records, visibleAttributes);
            }
        };
        mainWindow.executeLongRunningTask(runnable, "safe load encode tracks");
    }

    private void unsafeLoadENCODETracks(List<EncodeFileRecord> records, String[] visibleAttributes) {
        if (records.size() > 0) {
            if (loadedLocators == null) {
                loadedLocators = new HashSet<ResourceLocator>();
            }

            List<ResourceLocator> locators = null;
            for (EncodeFileRecord record : records) {
                ResourceLocator rl = new ResourceLocator(record.getPath());
                rl.setName(record.getTrackName());

                final String antibody = record.getAttributeValue("antibody");
                if (antibody != null) {
                    rl.setColor(colors.get(antibody.toUpperCase()));
                }

                for (String name : visibleAttributes) {
                    String value = record.getAttributeValue(name);
                    if (value != null) {
                        AttributeManager.getInstance().addAttribute(rl.getName(), name, value);
                    }
                }
                if (!loadedLocators.contains(rl)) {
                    if (locators == null) {
                        locators = new ArrayList<ResourceLocator>();
                    }

                    locators.add(rl);
                    loadedLocators.add(rl);
                }

            }
            if (locators != null) {
                hic.unsafeLoadHostedTracks(locators);
            }
        }
    }

    public void remove(ResourceLocator locator) {
        try {
            EncodeFileBrowser browser = EncodeFileBrowser.getInstance(genome);
            assert browser != null;
            browser.remove(locator);
            loadedLocators.remove(locator);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


}
