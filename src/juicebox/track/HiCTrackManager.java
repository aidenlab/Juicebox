/*
 * Copyright (c) 2007-2012 The Broad Institute, Inc.
 * SOFTWARE COPYRIGHT NOTICE
 * This software and its documentation are the copyright of the Broad Institute, Inc. All rights are reserved.
 *
 * This software is supplied without any warranty or guaranteed support whatsoever. The Broad Institute is not
 * responsible for its use, misuse, or functionality.
 *
 * This software is licensed under the terms of the GNU Lesser General Public License (LGPL), Version 2.1 which is
 * available at http://www.opensource.org/licenses/lgpl-2.1.php.
 */

package juicebox.track;

import htsjdk.tribble.AbstractFeatureReader;
import htsjdk.tribble.FeatureCodec;
import juicebox.HiC;
import juicebox.MainWindow;
import juicebox.NormalizationType;
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
import org.broad.igv.util.ResourceLocator;

import javax.swing.JOptionPane;
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

        if(coverageTracks.containsKey(no)) return; // Already loaded

        HiCDataSource source = new HiCCoverageDataSource(hic, no);
        ResourceLocator locator = new ResourceLocator(no.toString());
        HiCDataTrack track = new HiCDataTrack(hic, locator, source);
        coverageTracks.put(no, track);
        loadedTracks.add(track);
        mainWindow.updateTrackPanel();
    }


    public void load(final List<ResourceLocator> locators) {

        Runnable runnable = new Runnable() {
            public void run() {
                for (ResourceLocator locator : locators) {
                    loadTrack(locator);
                }

                mainWindow.updateTrackPanel();
            }
        };
        mainWindow.executeLongRunningTask(runnable);
    }

    public void add(HiCTrack track) {
        loadedTracks.add(track);
        mainWindow.updateTrackPanel();
    }

    private void loadTrack(final ResourceLocator locator) {

        Genome genome = loadGenome();
        String path = locator.getPath();
        String pathLC = path.toLowerCase();
        // genome = GenomeManager.getInstance().getCurrentGenome();

        if (pathLC.endsWith(".wig") || pathLC.endsWith(".bedgraph") ||
                pathLC.endsWith(".wig.gz") || pathLC.endsWith(".bedgraph.gz")) {
            HiCWigAdapter da = new HiCWigAdapter(hic, path);
            HiCDataTrack hicTrack = new HiCDataTrack(hic, locator, da);
            loadedTracks.add(hicTrack);
        } else if (pathLC.endsWith(".tdf") || pathLC.endsWith(".bigwig")) {
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
            }  catch (Exception e) {
                log.error("Error loading track: " + locator.getPath(), e);
                JOptionPane.showMessageDialog(mainWindow, "Error loading track. " + e.getMessage());
            }
        } else {
            FeatureCodec<?, ?> codec = CodecFactory.getCodec(locator, genome);
            if (codec != null) {
                AbstractFeatureReader bfs = AbstractFeatureReader.getFeatureReader(locator.getPath(), codec, false);

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
            }
            else {
                log.error("Error loading track: " + path);
                File file = new File(path);
                JOptionPane.showMessageDialog(mainWindow, "Error loading " + file.getName() +".\n Does not appear to be a track file.");
                hic.removeTrack(new HiCFeatureTrack(hic, locator, null));
            }
        }

    }

    public void removeTrack(HiCTrack track) {
        loadedTracks.remove(track);

        NormalizationType key = null;
        for(Map.Entry<NormalizationType, HiCTrack> entry : coverageTracks.entrySet()) {
              if(entry.getValue() == track) {
                  key = entry.getKey();
              }
        }

        if(key != null) {
            coverageTracks.remove(key);
        }

    }

    public void removeTrack(ResourceLocator locator) {
        HiCTrack track = null;
        for (HiCTrack tmp: loadedTracks){
            if (tmp.getLocator().equals(locator)) {
                track = tmp;
                break;
            }
        }
        loadedTracks.remove(track);

        NormalizationType key = null;
        for(Map.Entry<NormalizationType, HiCTrack> entry : coverageTracks.entrySet()) {
            if(entry.getValue() == track) {
                key = entry.getKey();
            }
        }

        if(key != null) {
            coverageTracks.remove(key);
        }

    }



    public List<HiCTrack> getLoadedTracks() {
        return loadedTracks;
    }

    public void clearTracks() {
        loadedTracks.clear();
    }

    private Genome loadGenome() {
        String genomePath;
        Genome genome = GenomeManager.getInstance().getCurrentGenome();
        if (genome == null) {
            if (hic.getDataset() != null) {
                // TODO this shouldn't be accessing broad anymore
                genomePath = "http://igvdata.broadinstitute.org/genomes/" + hic.getDataset().getGenomeId() + ".genome";
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
