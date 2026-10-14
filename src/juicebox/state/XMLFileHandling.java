/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2021 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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
 *  FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.state;

import juicebox.Context;
import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.gui.SuperAdapter;
import juicebox.track.HiCDataTrack;
import juicebox.track.HiCTrack;
import juicebox.windowui.HiCZoom;

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

        String xChr = xContext.getChromosome().getName();
        String yChr = yContext.getChromosome().getName();
        String colorVals = superAdapter.getMainViewPanel().getColorRangeValues();
        double colorRangeScaleFactor = superAdapter.getMainViewPanel().getColorRangeScaleFactor();
        List<HiCTrack> currentTracks = hic.getLoadedTracks();
        StringBuilder currentTrack = new StringBuilder();
        StringBuilder currentTrackName = new StringBuilder();
        String configTrackInfo = serializeTrackConfig(currentTracks);
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
            }
        } else {
            currentTrack = new StringBuilder("none");
            currentTrackName = new StringBuilder("none");
        }
        textToWrite += "$$" + currentTrack + "$$" + currentTrackName + "$$" + configTrackInfo;

        textToWrite += "$$" + hic.getControlNormalizationType().getLabel();

        //("currentState,xChr,yChr,resolution,zoom level,xbin,ybin,scale factor,display selection,
        // normalization type,color range values, tracks")
        HiCGlobals.savedStatesList.add(textToWrite);
        XMLFileWriter.saveState(textToWrite);
    }

    static String serializeTrackConfig(List<HiCTrack> tracks) {
        StringBuilder configs = new StringBuilder();
        if (tracks != null) {
            for (HiCTrack track : tracks) {
                if (track instanceof HiCDataTrack) {
                    HiCDataTrack dataTrack = (HiCDataTrack) track;
                    if (dataTrack.getDataRange() == null) continue;
                    configs.append(track.getName()).append(',').append(dataTrack.getPosColor().getRGB()).append(',')
                            .append(dataTrack.getNegColor().getRGB()).append(',')
                            .append(dataTrack.getDataRange().getMinimum()).append(',')
                            .append(dataTrack.getDataRange().getMaximum()).append(',')
                            .append(dataTrack.getDataRange().isLog()).append("**");
                }
            }
        }
        return configs.length() == 0 ? "none" : configs.toString();
    }
}
