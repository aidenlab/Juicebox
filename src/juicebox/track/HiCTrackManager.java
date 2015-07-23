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

package juicebox.track;

import htsjdk.tribble.AbstractFeatureReader;
import htsjdk.tribble.FeatureCodec;
import juicebox.HiC;
import juicebox.MainWindow;
import juicebox.windowui.NormalizationType;
import org.apache.log4j.Logger;
import org.broad.igv.bbfile.BBFileReader;
import org.broad.igv.bigwig.BigWigDataSource;
import org.broad.igv.feature.genome.Genome;
import org.broad.igv.feature.genome.GenomeManager;
import org.broad.igv.feature.tribble.CodecFactory;
import org.broad.igv.track.DataTrack;
import org.broad.igv.track.FeatureCollectionSource;
import org.broad.igv.track.Track;
import org.broad.igv.track.TrackLoader;
import org.broad.igv.ui.util.MessageUtils;
import org.broad.igv.util.ResourceLocator;

import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author Jim Robinson
 * @since 5/8/12
 */
public class HiCTrackManager {

    private static final Logger log = Logger.getLogger(HiCTrackManager.class);

    //static String path = "http://www.broadinstitute.org/igvdata/hic/tracksMenu.xml";
    //static String path = "/Users/jrobinso/Documents/IGV/hg19_encode.xml";

    private final java.util.List<HiCTrack> loadedTracks = new ArrayList<HiCTrack>();
    private final List<HiCTrack> reloadTrackNames = new ArrayList<HiCTrack>();
    private final Map<NormalizationType, HiCTrack> coverageTracks = new HashMap<NormalizationType, HiCTrack>();
    private final MainWindow mainWindow;
    private final HiC hic;

    public HiCTrackManager(MainWindow mainWindow, HiC hic) {
        this.mainWindow = mainWindow;
        this.hic = hic;

    }

    public void loadTrack(final String path) {
        Runnable runnable = new Runnable() {
            public void run() {
                loadTrack(new ResourceLocator(path));
                mainWindow.updateTrackPanel();
            }
        };
        // mainWindow.executeLongRunningTask(runnable);
        runnable.run();
    }

    public void loadCoverageTrack(NormalizationType no) {

        if (coverageTracks.containsKey(no)) return; // Already loaded

        HiCDataSource source = new HiCCoverageDataSource(hic, no);
        ResourceLocator locator = new ResourceLocator(no.getLabel());
        HiCDataTrack track = new HiCDataTrack(hic, locator, source);
        coverageTracks.put(no, track);
        loadedTracks.add(track);
        mainWindow.updateTrackPanel();
    }

    public void safeTrackLoad(final List<ResourceLocator> locators) {
        Runnable runnable = new Runnable() {
            public void run() {
                unsafeLoad(locators);
            }
        };
        mainWindow.executeLongRunningTask(runnable, "Safe Track Load");
    }

    private void unsafeLoad(final List<ResourceLocator> locators) {
        for (ResourceLocator locator : locators) {
            try {
                loadTrack(locator);
            } catch (Exception e) {
                MessageUtils.showMessage("Could not load resource:<br>" + e.getMessage());
                System.out.println("Removing " + locator.getName());
                hic.removeTrack(locator);

                if (locator.getType() != null && locator.getType().equals("loop")) {
                    try {
                        hic.setLoopsInvisible(locator.getPath());
                    } catch (Exception e2) {
                        log.error("Error while making loops invisible ", e2);
                        MessageUtils.showMessage("Error while removing loops: " + e2.getMessage());
                    }
                }
            }
        }
        mainWindow.updateTrackPanel();
    }

    public void add(HiCTrack track) {
        loadedTracks.add(track);
        mainWindow.updateTrackPanel();
    }

    private void loadTrack(final ResourceLocator locator) {

        Genome genome = loadGenome();
        String path = locator.getPath();
        String pathLC = path.toLowerCase();
        int index = path.lastIndexOf('.');
        String extension = path.substring(index).toLowerCase();
        // The below code is meant to solve problems recognizing the proper file type.  The IGV code looks for
        // the location "type" in order to read the file properly
        if (!extension.equals(".gz")) {
            locator.setType(extension);

        }
        else {
            // setting type to be the extension before the .gz
            int index2 = path.substring(0,index).lastIndexOf('.');
            String str = path.substring(0,index).substring(index2);
            // special exception for refGene.txt.gz
            if (!str.equals(".txt")) locator.setType(str);
        }

        if (pathLC.endsWith(".wig")  ||
                pathLC.endsWith(".wig.gz")) {
            HiCWigAdapter da = new HiCWigAdapter(hic, path);
            HiCDataTrack hicTrack = new HiCDataTrack(hic, locator, da);
            loadedTracks.add(hicTrack);
        } else if (pathLC.endsWith(".tdf") || pathLC.endsWith(".bigwig") || pathLC.endsWith(".bw")
                || pathLC.endsWith(".bedgraph") || pathLC.endsWith(".bedgraph.gz")) {
            List<Track> tracks = (new TrackLoader()).load(locator, genome);

            for (Track t : tracks) {
                HiCDataAdapter da = new HiCIGVDataAdapter(hic, (DataTrack) t);
                HiCDataTrack hicTrack = new HiCDataTrack(hic, locator, da);
                loadedTracks.add(hicTrack);
            }
       } else if (pathLC.endsWith(".bb")) {
            try {
                BigWigDataSource src = new BigWigDataSource(new BBFileReader(locator.getPath()), genome);
                HiCFeatureTrack track = new HiCFeatureTrack(hic, locator, src);
                track.setName(locator.getTrackName());
                loadedTracks.add(track);
            } catch (Exception e) {
                log.error("Error loading track: " + locator.getPath(), e);
                JOptionPane.showMessageDialog(mainWindow, "Error loading track. " + e.getMessage());
            }
        } else {
            FeatureCodec<?, ?> codec = CodecFactory.getCodec(locator, genome);
            if (codec != null) {
                AbstractFeatureReader<?, ?> bfs = AbstractFeatureReader.getFeatureReader(locator.getPath(), codec, false);

                try {
                    htsjdk.tribble.CloseableTribbleIterator<?> iter = bfs.iterator(); // CloseableTribbleIterator extends java.lang.Iterator
                    FeatureCollectionSource src = new FeatureCollectionSource(iter, genome);
                    HiCFeatureTrack track = new HiCFeatureTrack(hic, locator, src);
                    track.setName(locator.getTrackName());
                    loadedTracks.add(track);
                } catch (Exception e) {
                    log.error("Error loading track: " + path, e);
                    JOptionPane.showMessageDialog(mainWindow, "Error loading track. " + e.getMessage());
                }
                //Object header = bfs.getHeader();
                //TrackProperties trackProperties = getTrackProperties(header);
            } else {
                log.error("Error loading track: " + path);
                System.out.println("path: "+path);//DEBUGGING
                File file = new File(path);
                JOptionPane.showMessageDialog(mainWindow, "Error loading " + file.getName() + ".\n Does not appear to be a track file.");
                hic.removeTrack(new HiCFeatureTrack(hic, locator, null));
            }
        }

    }

    public void removeTrack(HiCTrack track) {
        loadedTracks.remove(track);

        NormalizationType key = null;
        for (Map.Entry<NormalizationType, HiCTrack> entry : coverageTracks.entrySet()) {
            if (entry.getValue() == track) {
                key = entry.getKey();
            }
        }

        if (key != null) {
            coverageTracks.remove(key);
        }
        mainWindow.updateTrackPanel();
    }


    public void removeTrack(ResourceLocator locator) {
        HiCTrack track = null;
        for (HiCTrack tmp : loadedTracks) {
            if (tmp.getLocator().equals(locator)) {
                track = tmp;
                break;
            }
        }
        removeTrack(track);
    }


    public List<HiCTrack> getLoadedTracks() {
        return loadedTracks;
    }

    public Map<NormalizationType, HiCTrack> getCoverageTracks() {
        return coverageTracks;
    }

    public void clearTracks() {
        loadedTracks.clear();
    }

    public List<HiCTrack> getReloadTracks(List<HiCTrack> reloadTracks){
        for (HiCTrack reloadTrack : reloadTracks)
            reloadTrackNames.add(reloadTrack);
        return reloadTrackNames;
    }

    public List<HiCTrack> getReloadTrackNames(){
        return reloadTrackNames;
    }


    private Genome loadGenome() {
        String genomePath;
        Genome genome = GenomeManager.getInstance().getCurrentGenome();
        if (genome == null) {
            if (hic.getDataset() != null) {
                if (hic.getDataset().getGenomeId().equals("assembly")) {
                    genomePath = "http://hicfiles.s3.amazonaws.com/internal/assembly.genome";
                }
                else if (hic.getDataset().getGenomeId().equals("dMel")) {
                    genomePath = "http://igvdata.broadinstitute.org/genomes/dmel_r5.22.genome";
                }
                else {
                    genomePath = "http://igvdata.broadinstitute.org/genomes/" + hic.getDataset().getGenomeId() + ".genome";
                }
            } else {
                genomePath = "http://igvdata.broadinstitute.org/genomes/hg19.genome";
            }

            try {
                genome = GenomeManager.getInstance().loadGenome(genomePath, null);
            } catch (IOException e) {
                log.error("Error loading genome: " + genomePath, e);
            }

        }
        return genome;
    }

}
