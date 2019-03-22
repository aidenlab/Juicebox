/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2019 Broad Institute, Aiden Lab
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
import juicebox.gui.SuperAdapter;
import juicebox.track.*;
import juicebox.windowui.MatrixType;
import org.broad.igv.renderer.DataRange;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * Created by Zulkifl on 6/10/2015.
 */
public class LoadStateFromXMLFile {

    public static void reloadSelectedState(SuperAdapter superAdapter, String mapPath) {
        superAdapter.getHiC().clearTracksForReloadState(); // TODO - should only remove necessary ones
        try {
            loadSavedStatePreliminaryStep(XMLFileParser.parseXML(mapPath), superAdapter, superAdapter.getHiC());
        } catch (IOException e) {
            e.printStackTrace();
        }
        superAdapter.refresh();
    }

    private static void loadSavedStatePreliminaryStep(String[] infoForReload, SuperAdapter superAdapter, HiC hic) throws IOException {
        String result = "OK";
        String[] initialInfo = new String[6]; //hicURL,, controlURL,xChr,yChr,unitSize
        double[] doubleInfo = new double[8]; //xOrigin, yOrigin, ScaleFactor, minColorVal, lowerColorVal, upperColorVal, maxColorVal, colorScaleFactor
        String[] trackURLsAndNamesAndConfigInfo = new String[3];
        //System.out.println("Executing: " + Arrays.toString(infoForReload)); //TODO for reload
        if (infoForReload.length > 0) {
            //int fileSize = infoForReload.length;
            if (infoForReload.length > 15) {
                try {
                    // TODO cleanup extraction of data
                    initialInfo[0] = infoForReload[1]; //HiC Map Name
                    initialInfo[1] = infoForReload[2]; //hicURL
                    initialInfo[2] = infoForReload[3]; //controlURL
                    initialInfo[3] = infoForReload[4]; //xChr
                    initialInfo[4] = infoForReload[5]; //yChr
                    initialInfo[5] = infoForReload[6]; //unitSize
                    int binSize = Integer.parseInt(infoForReload[7]); //binSize
                    doubleInfo[0] = Double.parseDouble(infoForReload[8]); //xOrigin
                    doubleInfo[1] = Double.parseDouble(infoForReload[9]); //yOrigin
                    doubleInfo[2] = Double.parseDouble(infoForReload[10]); //ScaleFactor
                    MatrixType displayOption = MatrixType.valueOf(infoForReload[11].toUpperCase());
                    String normType = infoForReload[12].toUpperCase();
                    doubleInfo[3] = Double.parseDouble(infoForReload[13]); //minColorVal
                    doubleInfo[4] = Double.parseDouble(infoForReload[14]); //lowerColorVal
                    doubleInfo[5] = Double.parseDouble(infoForReload[15]); //upperColorVal
                    doubleInfo[6] = Double.parseDouble(infoForReload[16]); //maxColorVal
                    doubleInfo[7] = Double.parseDouble(infoForReload[17]); //colorScaleFactor
                    trackURLsAndNamesAndConfigInfo[0] = (infoForReload[18]); //trackURLs
                    trackURLsAndNamesAndConfigInfo[1] = (infoForReload[19]); //trackNames
                    trackURLsAndNamesAndConfigInfo[2] = (infoForReload[20]); //trackConfigInfo
                    //Increase XMLFileParser::infoForReload when adding more elements.
                    safeLoadStateFromXML(superAdapter, hic, initialInfo, binSize, doubleInfo, displayOption, normType, trackURLsAndNamesAndConfigInfo);
                } catch (NumberFormatException nfe) {
                    JOptionPane.showMessageDialog(superAdapter.getMainWindow(), "Error:\n" + nfe.getMessage(), "Error",
                            JOptionPane.ERROR_MESSAGE);
                }
            } else {
                throw new IOException("Not enough parameters");
            }
        } else {
            throw new IOException("Unknown command string");
        }
    }

    private static void safeLoadStateFromXML(final SuperAdapter superAdapter, final HiC hic, final String[] initialInfo,
                                             final int binSize, final double[] doubleInfo, final MatrixType displaySelection,
                                             final String normSelection, final String[] tracks) {
        Runnable runnable = new Runnable() {
            public void run() {
                try {
                    unsafeLoadStateFromXML(superAdapter, hic, initialInfo, binSize, doubleInfo,
                            displaySelection, normSelection, tracks);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        };
        superAdapter.executeLongRunningTask(runnable, "Loading a saved state from XML");
    }

    private static void unsafeLoadStateFromXML(SuperAdapter superAdapter, HiC hic, String[] initialInfo, int binSize, double[] doubleInfo,
                                               MatrixType displaySelection, String normSelection,
                                               String[] tracks) {

        superAdapter.resetControlMap(); //TODO test


        String mapNames = initialInfo[0];
        String mapURLs = initialInfo[1];
        String controlURLs = initialInfo[2];
        String chrXName = initialInfo[3];
        String chrYName = initialInfo[4];
        String unitName = initialInfo[5];
        double xOrigin = doubleInfo[0];
        double yOrigin = doubleInfo[1];
        double scalefactor = doubleInfo[2];
        double minColor = doubleInfo[3];
        double lowColor = doubleInfo[4];
        double upColor = doubleInfo[5];
        double maxColor = doubleInfo[6];
        double colorScaleFactor = doubleInfo[7];

        // TODO only do this if not identical to current file
        String[] temp = mapNames.split("\\(control=");
        String mainMapNames = temp[0];

        List<String> urls = Arrays.asList(mapURLs.split("\\#\\#"));
        superAdapter.unsafeLoadWithTitleFix(urls, false, mainMapNames, true);

        if (!controlURLs.contains("null") && temp.length > 1) {
            String ctrlMapNames = temp[1].substring(0, temp[1].length() - 1);
            List<String> ctrlURLs = Arrays.asList(controlURLs.split("\\#\\#"));
            superAdapter.unsafeLoadWithTitleFix(ctrlURLs, true, ctrlMapNames, true);
        }

        hic.unsafeSetLocation(chrXName, chrYName, unitName, binSize, xOrigin, yOrigin, scalefactor, HiC.ZoomCallType.DIRECT, true);
        superAdapter.getMainViewPanel().setDisplayBox(displaySelection.ordinal());
        // todo should not be implemented with ordinals; implement with proper lookup
        // leaving unimplemented at this time
        //superAdapter.getMainViewPanel().setNormalizationBox(normSelection.ordinal());
        superAdapter.getMainViewPanel().setNormalizationEnabledForReload();
        //todo: Check that color scale value is not 0!
        superAdapter.getMainViewPanel().updateColorSlider(hic, minColor / colorScaleFactor, lowColor / colorScaleFactor, upColor / colorScaleFactor, maxColor / colorScaleFactor);
        superAdapter.setEnableForAllElements(true);

        LoadEncodeAction loadEncodeAction = superAdapter.getEncodeAction();

        LoadAction loadAction = superAdapter.getTrackLoadAction();

        // TODO - do not erase previous tracks, rather check if some may already be loaded
        try {
            if (tracks.length > 0 && !tracks[1].contains("none")) {
                String[] trackURLs = tracks[0].split("\\,");
                String[] trackNames = tracks[1].split("\\,");
                for (int i = 0; i < trackURLs.length; i++) {
                    String currentTrack = trackURLs[i].trim();
                    if (!currentTrack.isEmpty()) {
                        if (currentTrack.equals("Eigenvector")) {
                            loadAction.checkBoxesForReload(trackNames[i].trim());
                            hic.loadEigenvectorTrack();
                        } else if (currentTrack.toLowerCase().contains("coverage") || currentTrack.toLowerCase().contains("balanced")
                                || currentTrack.equals("Loaded")) {
                            loadAction.checkBoxesForReload(trackNames[i].trim());
                            hic.loadCoverageTrack(currentTrack);
                        } else if (currentTrack.contains("peaks") || currentTrack.contains("blocks") || currentTrack.contains("superloop")) {
                            hic.getResourceTree().checkTrackBoxesForReloadState(currentTrack.trim());
                            hic.loadLoopList(currentTrack);
                        } else if (currentTrack.contains("goldenPath") || currentTrack.toLowerCase().contains("ensemble")) {
                            hic.unsafeLoadTrack(currentTrack);
                            loadEncodeAction.checkEncodeBoxes(trackNames[i].trim());
                        } else {
                            hic.unsafeLoadTrack(currentTrack);
                            loadAction.checkBoxesForReload(trackNames[i].trim());
                        }
                    }
                }
                for (HiCTrack loadedTrack : hic.getLoadedTracks()) {
                    for (int i = 0; i < trackNames.length; i++) {
                        if (trackURLs[i].contains(loadedTrack.getName())) {
                            loadedTrack.setName(trackNames[i].trim());
                            if (!tracks[2].contains("none") && tracks[2].contains(trackNames[i].trim())) {

                                HiCDataSource source = new HiCCoverageDataSource(hic, hic.getObsNormalizationType(), false);
                                HiCDataTrack hiCDataTrack = new HiCDataTrack(hic, loadedTrack.getLocator(), source);

                                String[] configTrackInfo = tracks[2].split("\\*\\*");
                                for (String aConfigTrackInfo : configTrackInfo) {

                                    String[] configInfo = aConfigTrackInfo.split("\\,"); //todo check
                                    hiCDataTrack.setPosColor(new Color(Integer.parseInt(configInfo[1])));
                                    hiCDataTrack.setNegColor(new Color(Integer.parseInt(configInfo[2])));
                                    DataRange newDataRange = new DataRange(Float.parseFloat(configInfo[3]), Float.parseFloat(configInfo[4]));//min,max
                                    if (Boolean.parseBoolean(configInfo[5])) {
                                        newDataRange.setType(DataRange.Type.LOG);
                                    } else {
                                        newDataRange.setType(DataRange.Type.LINEAR);
                                    }
                                    hiCDataTrack.setDataRange(newDataRange);

                                }
                            }
                        }
                    }

                }

            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        superAdapter.updateTrackPanel();

    }

}
