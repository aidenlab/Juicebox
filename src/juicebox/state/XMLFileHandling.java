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

import juicebox.Context;
import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.data.Dataset;
import juicebox.gui.SuperAdapter;
import juicebox.track.HiCCoverageDataSource;
import juicebox.track.HiCDataSource;
import juicebox.track.HiCDataTrack;
import juicebox.track.HiCTrack;
import juicebox.track.feature.AnnotationLayerHandler;
import juicebox.track.feature.Feature2DList;
import juicebox.windowui.HiCZoom;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Zulkifl Gire on 7/10/2015.
 */
public class XMLFileHandling {
    public static void addNewStateToXML(String stateID, SuperAdapter superAdapter) {

        HiC hic = superAdapter.getHiC();
        Context xContext = hic.getXContext();
        Context yContext = hic.getYContext();
        HiCZoom zoom = hic.getZoom();
        Dataset dataset = hic.getDataset();

        String xChr = xContext.getChromosome().getName();
        String yChr = yContext.getChromosome().getName();
        String colorVals = superAdapter.getMainViewPanel().getColorRangeValues();
        double colorRangeScaleFactor = superAdapter.getMainViewPanel().getColorRangeScaleFactor();
        List<HiCTrack> currentTracks = hic.getLoadedTracks();
        StringBuilder currentTrack = new StringBuilder();
        StringBuilder currentTrackName = new StringBuilder();
        String configTrackInfo = "none";
        String controlFiles = SuperAdapter.currentlyLoadedControlFiles;
        if (controlFiles == null || controlFiles.length() < 1 || controlFiles.isEmpty()) {
            controlFiles = "null";
        }

        String mapNameAndURLs = superAdapter.getMainWindow().getTitle().replace(HiCGlobals.juiceboxTitle, "") + "@@" + SuperAdapter.currentlyLoadedMainFiles
                + "@@" + controlFiles;

        String textToWrite = stateID + "--currentState:$$" + mapNameAndURLs + "$$" + xChr + "$$" + yChr + "$$" + zoom.getUnit().toString() + "$$" +
                zoom.getBinSize() + "$$" + xContext.getBinOrigin() + "$$" + yContext.getBinOrigin() + "$$" +
                hic.getScaleFactor() + "$$" + hic.getDisplayOption().name() + "$$" + hic.getObsNormalizationType().getLabel()
                + "$$" + colorVals + "$$" + colorRangeScaleFactor;

        if (currentTracks != null && !currentTracks.isEmpty()) {
            for (HiCTrack track : currentTracks) {
                //System.out.println("trackLocator: "+track.getLocator()); for debugging
                //System.out.println("track name: " + track.getName());
                currentTrack.append(track.getLocator()).append(", ");
                currentTrackName.append(track.getName()).append(", ");
                track.getLocator().getColor();
                try {
                    HiCDataSource source = new HiCCoverageDataSource(hic, hic.getObsNormalizationType(), false);
                    HiCDataTrack hiCDataTrack = new HiCDataTrack(hic, track.getLocator(), source);

                    configTrackInfo = track.getName() + "," + hiCDataTrack.getPosColor().getRGB() + ","
                            + hiCDataTrack.getNegColor().getRGB() + "," + hiCDataTrack.getDataRange().getMinimum() + ","
                            + hiCDataTrack.getDataRange().getMaximum() + "," + hiCDataTrack.getDataRange().isLog() + "**";
                    //Name, PosColor, AltColor, Min, Max, isLogScale
                } catch (Exception e) {
                    // Expected for tracks that cannot be configured
                }
            }
            textToWrite += "$$" + currentTrack + "$$" + currentTrackName + "$$" + configTrackInfo;
        } else {
            currentTrack = new StringBuilder("none");
            currentTrackName = new StringBuilder("none");
            textToWrite += "$$" + currentTrack + "$$" + currentTrackName + "$$" + configTrackInfo;
        }

        // TODO this needs some major restructuring
        List<Feature2DList> visibleLoops = new ArrayList<>();
        for (AnnotationLayerHandler handler : superAdapter.getAllLayers()) {
            visibleLoops.add(handler.getAllVisibleLoops());
        }
        if (visibleLoops != null && !visibleLoops.isEmpty()) {
            try {
                textToWrite += "$$" + dataset.getPeaks().toString() + "$$" +
                        dataset.getBlocks().toString() + "$$" + dataset.getSuperLoops().toString();
            } catch (Exception ignored) {

            }
        }

        //("currentState,xChr,yChr,resolution,zoom level,xbin,ybin,scale factor,display selection,
        // normalization type,color range values, tracks")
        HiCGlobals.savedStatesList.add(textToWrite);
        XMLFileWriter.overwriteXMLFile();
    }
}