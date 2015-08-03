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

package juicebox.state;

import juicebox.HiC;
import juicebox.MainWindow;
import juicebox.track.HiCTrack;
import juicebox.track.LoadAction;
import juicebox.track.LoadEncodeAction;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.MatrixType;
import juicebox.windowui.NormalizationType;

import javax.swing.*;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * Created by Zulkifl on 6/10/2015.
 */
public class LoadStateFromXMLFile {

    public static void reloadSelectedState(String mapPath, MainWindow mainWindow, HiC hic) {
        hic.clearTracksForReloadState(); // TODO - should only remove necessary ones
        try {
            loadSavedStatePreliminaryStep(XMLFileParser.parseXML(mapPath), hic, mainWindow);
        } catch (IOException e) {
            e.printStackTrace();
        }
        mainWindow.refresh();
    }

    private static void loadSavedStatePreliminaryStep(String[] infoForReload, HiC hic, MainWindow mainWindow) throws IOException {
        String result = "OK";
        String[] initialInfo = new String[5]; //hicURL,xChr,yChr,unitSize
        double[] doubleInfo = new double[7]; //xOrigin, yOrigin, ScaleFactor, minColorVal, lowerColorVal, upperColorVal, maxColorVal
        String[] trackURLsAndNames = new String[2];
        System.out.println("Executing: " + Arrays.toString(infoForReload));
        if (infoForReload.length > 0) {
            //int fileSize = infoForReload.length;
            if (infoForReload.length > 14) {
                try {
                    // TODO cleanup extraction of data
                    initialInfo[0] = infoForReload[1]; //HiC Map Name
                    initialInfo[1] = infoForReload[2]; //hicURL
                    initialInfo[2] = infoForReload[3]; //xChr
                    initialInfo[3] = infoForReload[4]; //yChr
                    initialInfo[4] = infoForReload[5]; //unitSize
                    int binSize = Integer.parseInt(infoForReload[6]); //binSize
                    doubleInfo[0] = Double.parseDouble(infoForReload[7]); //xOrigin
                    doubleInfo[1] = Double.parseDouble(infoForReload[8]); //yOrigin
                    doubleInfo[2] = Double.parseDouble(infoForReload[9]); //ScaleFactor
                    MatrixType displayOption = MatrixType.valueOf(infoForReload[10].toUpperCase());
                    NormalizationType normType = NormalizationType.valueOf(infoForReload[11].toUpperCase());
                    doubleInfo[3] = Double.parseDouble(infoForReload[12]); //minColorVal
                    doubleInfo[4] = Double.parseDouble(infoForReload[13]); //lowerColorVal
                    doubleInfo[5] = Double.parseDouble(infoForReload[14]); //upperColorVal
                    doubleInfo[6] = Double.parseDouble(infoForReload[15]); //maxColorVal
                    trackURLsAndNames[0] = (infoForReload[16]); //trackURLs
                    trackURLsAndNames[1] = (infoForReload[17]); //trackNames

                    safeLoadStateFromXML(hic, mainWindow, initialInfo, binSize, doubleInfo, displayOption, normType, trackURLsAndNames);
                } catch (NumberFormatException nfe) {
                    JOptionPane.showMessageDialog(MainWindow.getInstance(), "Error:\n" + nfe.getMessage(), "Error",
                            JOptionPane.ERROR_MESSAGE);
                }
            } else {
                throw new IOException("Not enough parameters");
            }
        } else {
            throw new IOException("Unknown command string");
        }
    }

    private static void safeLoadStateFromXML(final HiC hic, final MainWindow mainWindow, final String[] initialInfo,
                                             final int binSize, final double[] doubleInfo, final MatrixType displaySelection,
                                             final NormalizationType normSelection, final String[] tracks) {
        Runnable runnable = new Runnable() {
            public void run() {
                try {
                    unsafeLoadStateFromXML(mainWindow, hic, initialInfo, binSize, doubleInfo,
                            displaySelection, normSelection, tracks);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        };
        mainWindow.executeLongRunningTask(runnable, "Loading a saved state from XML");
    }

    private static void unsafeLoadStateFromXML(MainWindow mainWindow, HiC hic, String[] initialInfo, int binSize, double[] doubleInfo,
                                               MatrixType displaySelection, NormalizationType normSelection,
                                               String[] tracks) {
        String mapNames = initialInfo[0];
        String mapURLs = initialInfo[1];
        String chrXName = initialInfo[2];
        String chrYName = initialInfo[3];
        String unitName = initialInfo[4];
        double xOrigin = doubleInfo[0];
        double yOrigin = doubleInfo[1];
        double scalefactor = doubleInfo[2];
        double minColor = doubleInfo[3];
        double lowColor = doubleInfo[4];
        double upColor = doubleInfo[5];
        double maxColor = doubleInfo[6];

        // only do this if not identical to current file
        List<String> urls = Arrays.asList(mapURLs.split("\\@\\@"));
        mainWindow.unsafeLoadWithTitleFix(urls, false, mapNames);

        hic.setLocation(chrXName, chrYName, "BP", binSize, xOrigin, yOrigin, scalefactor);

        HiCZoom newZoom = new HiCZoom(HiC.Unit.valueOf(unitName), binSize);
        hic.setZoom(newZoom, xOrigin, yOrigin);
        mainWindow.updateZoom(newZoom);

        mainWindow.setDisplayBox(displaySelection.ordinal());
        mainWindow.setNormalizationBox(normSelection.ordinal());
        mainWindow.updateColorSlider(minColor, lowColor, upColor, maxColor);
        mainWindow.enableAllOptionsButtons();


        LoadEncodeAction loadEncodeAction = mainWindow.getEncodeAction();
        LoadAction loadAction = mainWindow.getTrackLoadAction();

        // TODO - do not erase previous tracks, rather check if some may already be loaded
        try {
            if (tracks.length > 0 && !tracks[1].contains("none")) {
                String[] trackURLs = tracks[0].split("\\,");
                String[] trackNames = tracks[1].split("\\,");
                for (int i = 0; i < trackURLs.length; i++) {
                    String currentTrack = trackURLs[i].trim();
                    if (!currentTrack.isEmpty()) {
                        if (currentTrack.equals("Eigenvector")) {
                            hic.loadEigenvectorTrack();
                        } else if (currentTrack.toLowerCase().contains("coverage") || currentTrack.toLowerCase().contains("balanced")
                                || currentTrack.equals("Loaded")) {
                            hic.loadCoverageTrack(NormalizationType.enumValueFromString(currentTrack));
                        } else if (currentTrack.contains("peaks") || currentTrack.contains("blocks") || currentTrack.contains("superloop")) {
                            hic.getResourceTree().checkTrackBoxesForReloadState(currentTrack.trim());
                            hic.loadLoopList(currentTrack);
                        } else if (currentTrack.contains("goldenPath") || currentTrack.toLowerCase().contains("ensemble")) {
                            hic.loadTrack(currentTrack);
                            loadEncodeAction.checkEncodeBoxes(trackNames[i].trim());
                        } else {
                            hic.loadTrack(currentTrack);
                            loadAction.checkBoxesForReload(trackNames[i].trim());
                        }
                        //renaming
                    }
                }
                for (HiCTrack loadedTrack : hic.getLoadedTracks()) {
                    for (int i = 0; i < trackNames.length; i++) {
                        if (trackURLs[i].contains(loadedTrack.getName())) {
                            loadedTrack.setName(trackNames[i].trim());
                        }
                    }
                }

            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        mainWindow.updateTrackPanel();

    }

}
