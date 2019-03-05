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

package juicebox.track;

import juicebox.HiC;
import juicebox.gui.SuperAdapter;
import juicebox.tools.dev.Private;
import juicebox.windowui.NormalizationType;
import org.broad.igv.bbfile.BBFileReader;
import org.broad.igv.bigwig.BigWigDataSource;
import org.broad.igv.feature.genome.Genome;
import org.broad.igv.feature.genome.GenomeManager;
import org.broad.igv.feature.tribble.FeatureFileHeader;
import org.broad.igv.feature.tribble.TribbleIndexNotFoundException;
import org.broad.igv.track.*;
import org.broad.igv.util.ResourceLocator;

import javax.swing.*;
import java.io.IOException;
import java.util.*;

/**
 * @author Jim Robinson
 * @since 5/8/12
 */
public class HiCTrackManager {

    //static String path = "http://www.broadinstitute.org/igvdata/hic/tracksMenu.xml";
    //static String path = "/Users/jrobinso/Documents/IGV/hg19_encode.xml";

    private final List<HiCTrack> loadedTracks = new ArrayList<>();
    private final Map<NormalizationType, HiCTrack> coverageTracks = new HashMap<>();
    private final Map<NormalizationType, HiCTrack> controlCoverageTracks = new HashMap<>();
    private final SuperAdapter superAdapter;
    private final HiC hic;

    public HiCTrackManager(SuperAdapter superAdapter, HiC hic) {
        this.superAdapter = superAdapter;
        this.hic = hic;
        org.broad.igv.Globals.setSuppressMessages(true);

    }

    public void unsafeLoadTrackDirectPath(final String path) {
        unsafeLoadTrackDirect(new ResourceLocator(path));
        superAdapter.updateTrackPanel();
    }

    public void loadCoverageTrack(NormalizationType no, boolean isControl) {
        if (isControl) {
            if (controlCoverageTracks.containsKey(no)) return; // Already loaded
            HiCDataSource source = new HiCCoverageDataSource(hic, no, isControl);
            ResourceLocator locator = new ResourceLocator(no.getDescription());
            HiCDataTrack track = new HiCDataTrack(hic, locator, source);
            controlCoverageTracks.put(no, track);
            loadedTracks.add(track);
            superAdapter.updateTrackPanel();
        } else {
            if (coverageTracks.containsKey(no)) return; // Already loaded
            HiCDataSource source = new HiCCoverageDataSource(hic, no, isControl);
            ResourceLocator locator = new ResourceLocator(no.getDescription());
            HiCDataTrack track = new HiCDataTrack(hic, locator, source);
            coverageTracks.put(no, track);
            loadedTracks.add(track);
            superAdapter.updateTrackPanel();
        }
    }

    public void unsafeTrackLoad(final List<ResourceLocator> locators) {
        for (ResourceLocator locator : locators) {
            try {
                unsafeLoadTrackDirect(locator);
            } catch (Exception e) {
                SuperAdapter.showMessageDialog("Could not load resource:<br>" + e.getMessage());
                System.out.println("Removing " + locator.getName());
                hic.removeTrack(locator);
            }
        }
        superAdapter.updateTrackPanel();
    }

    public void add(HiCTrack track) {
        loadedTracks.add(track);
        superAdapter.updateTrackPanel();
    }

    private void unsafeLoadTrackDirect(final ResourceLocator locator) {

        Genome genome = loadGenome();
        String path = locator.getPath();
        String pathLC = path.toLowerCase();
        int index = path.lastIndexOf('.');

        if (index < 0) {
            SuperAdapter.showMessageDialog("File is missing extension");
            return;
        }

        String extension = path.substring(index).toLowerCase();
        // The below code is meant to solve problems recognizing the proper file type.  The IGV code looks for
        // the location "type" in order to read the file properly
        if (extension.equals(".gz")) {
            // setting type to be the extension before the .gz
            int index2 = path.substring(0, index).lastIndexOf('.');
            String str = path.substring(0, index).substring(index2);
            // special exception for refGene.txt.gz
            if (!str.equals(".txt")) {
                locator.setType(str);
            }
        } else {
            if (extension.equals(".txt") || extension.equals(".zip")) {
                SuperAdapter.showMessageDialog(".txt files are not a currently supported 1D track. " +
                        "If you are trying to use refGene, make sure it is in the .txt.gz format. " +
                        "If you are trying to load loops/domains, use the 2D Annotations panel.");
                return;
            } else {
                locator.setType(extension);
            }
        }

        if (pathLC.endsWith(".wig") ||
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
                System.err.println("Error loading track: " + locator.getPath() + " " + e.getLocalizedMessage());
                JOptionPane.showMessageDialog(superAdapter.getMainWindow(), "Error loading track. " + e.getMessage());
            }
        } else {
            List<HiCTrack> tracks = new ArrayList<>();
            try {
                loadTribbleFile(locator, tracks, genome);

                loadedTracks.addAll(tracks);
            } catch (Exception e) {
                System.err.println("Error loading track: " + locator.getPath() + " " + e.getLocalizedMessage());
                JOptionPane.showMessageDialog(superAdapter.getMainWindow(), "Error loading track. " + e.getMessage());
            }
           /* FeatureCodec<?, ?> codec = CodecFactory.getCodec(locator, genome);
            if (codec != null) {
                 AbstractFeatureReader<?, ?> bfs = AbstractFeatureReader.getFeatureReader(locator.getPath(), codec, false);

                try {
                  //  htsjdk.tribble.CloseableTribbleIterator<?> iter = bfs.iterator(); // CloseableTribbleIterator extends java.lang.Iterator
                    FeatureCollectionSource src = new FeatureCollectionSource(iter, genome);
                    HiCFeatureTrack track = new HiCFeatureTrack(hic, locator, src);
                    track.setName(locator.getTrackName());
                    loadedTracks.add(track);
                } catch (Exception e) {
                    System.err.println("Error loading track: " + path, e);
                    JOptionPane.showMessageDialog(superAdapter.getMainWindow(), "Error loading track. " + e.getMessage());
                }
                //Object header = bfs.getHeader();
                //TrackProperties trackProperties = getTrackProperties(header);
            } else {
                System.err.println("Error loading track: " + path);
                System.out.println("path: " + path);//DEBUGGING
                File file = new File(path);
                JOptionPane.showMessageDialog(superAdapter.getMainWindow(), "Error loading " + file.getName() + ".\n Does not appear to be a track file.");
                hic.removeTrack(new HiCFeatureTrack(hic, locator, null));
            }  */
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

        key = null;
        for (Map.Entry<NormalizationType, HiCTrack> entry : controlCoverageTracks.entrySet()) {
            if (entry.getValue() == track) {
                key = entry.getKey();
            }
        }

        if (key != null) {
            controlCoverageTracks.remove(key);
        }
        superAdapter.updateTrackPanel();
    }

    public void removeTrack(ResourceLocator locator) {
        List<HiCTrack> tracks = new ArrayList<>();
        for (HiCTrack tmp : loadedTracks) {
            if (tmp.getLocator().equals(locator)) {
                tracks.add(tmp);
                // for coverage tracks, can have more than one, so don't break
            }
        }
        for (HiCTrack track : tracks) {
            removeTrack(track);
        }
    }


    public void moveTrackUp(HiCTrack track) {
        int currentIdx = loadedTracks.indexOf(track);
        if (currentIdx != 0) {
            Collections.swap(loadedTracks, currentIdx, currentIdx - 1);
            superAdapter.updateTrackPanel();
        }
    }

    public void moveTrackDown(HiCTrack track) {
        int currentIdx = loadedTracks.indexOf(track);
        if (currentIdx != loadedTracks.size() - 1) {
            Collections.swap(loadedTracks, currentIdx, currentIdx + 1);
            superAdapter.updateTrackPanel();
        }
    }

    public List<HiCTrack> getLoadedTracks() {
        return loadedTracks;
    }

    public void clearTracks() {
        loadedTracks.clear();
        coverageTracks.clear();
        controlCoverageTracks.clear();
    }

    /* TODO @zgire, is this old code that can be deleted?

    public Map<NormalizationType, HiCTrack> getCoverageTracks() {
        return coverageTracks;
    }

    public List<HiCTrack> getReloadTracks(List<HiCTrack> reloadTracks) {
        for (HiCTrack reloadTrack : reloadTracks)
            reloadTrackNames.add(reloadTrack);
        return reloadTrackNames;
    }

    public List<HiCTrack> getReloadTrackNames() {
        return reloadTrackNames;
    }
    */

    private Genome loadGenome() {
        String genomePath;
        Genome genome = GenomeManager.getInstance().getCurrentGenome();
        if (genome == null) {
            if (hic.getDataset() != null) {
                if (Private.assessGenomeID(hic.getDataset().getGenomeId())) {
                    genomePath = Private.getGenome();
                } else if (hic.getDataset().getGenomeId().equals("dMel")) {
                    genomePath = "http://igvdata.broadinstitute.org/genomes/dmel_r5.22.genome";
                } else {
                    genomePath = "http://igvdata.broadinstitute.org/genomes/" + hic.getDataset().getGenomeId() + ".genome";
                }
            } else {
                genomePath = "http://igvdata.broadinstitute.org/genomes/hg19.genome";
            }

            try {
                genome = GenomeManager.getInstance().loadGenome(genomePath, null);
            } catch (IOException e) {
                System.err.println("Error loading genome: " + genomePath + " " + e.getLocalizedMessage());
            }

        }
        /**
         * TODO potential fix for ASSEMBLY vs assembly @sa501428
         List<Chromosome> cleanedChromosomes = new ArrayList<>();
         for(String name : genome.getAllChromosomeNames()){
         Chromosome chr = genome.getChromosome(name);
         //cleanedChromosomes.add(chr);
         cleanedChromosomes.add(new Chromosome(chr.getIndex(), ChromosomeHandler.cleanUpName(name), chr.getLength()));
         }


         Genome finalGenome = new Genome(genome.getId(), cleanedChromosomes);
         */

        return genome;
    }

    /**
     * Load the input file as a feature file.
     * Taken from IGV, but needed to be separate because our tracks are different.
     *
     * @param locator
     * @param newTracks
     */
    private void loadTribbleFile(ResourceLocator locator, List<HiCTrack> newTracks, Genome genome)
            throws IOException, TribbleIndexNotFoundException {

        TribbleFeatureSource tribbleFeatureSource = TribbleFeatureSource.getFeatureSource(locator, genome);
        FeatureSource<?> src = GFFFeatureSource.isGFF(locator.getPath()) ?
                new GFFFeatureSource(tribbleFeatureSource) : tribbleFeatureSource;

        // Create feature source and track
        HiCFeatureTrack t = new HiCFeatureTrack(hic, locator, src);
        t.setName(locator.getTrackName());
        //t.setRendererClass(BasicTribbleRenderer.class);

        // Set track properties from header
        Object header = tribbleFeatureSource.getHeader();
        if (header instanceof FeatureFileHeader) {
            FeatureFileHeader ffh = (FeatureFileHeader) header;
          /*  if (ffh.getTrackType() != null) {
                t.setTrackType(ffh.getTrackType());
            }
            if (ffh.getTrackProperties() != null) {
                t.setProperties(ffh.getTrackProperties());
            }*/

            if (ffh.getTrackType() == TrackType.REPMASK) {
                t.setHeight(15);
            }
        }
       /* if (locator.getPath().contains(".narrowPeak") || locator.getPath().contains(".broadPeak") || locator.getPath().contains(".gappedPeak")) {
            t.setUseScore(true);
        }*/
        newTracks.add(t);
    }

}
