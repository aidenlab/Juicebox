/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2018 Broad Institute, Aiden Lab
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

package juicebox.tools.dev;

import juicebox.data.ChromosomeHandler;
import juicebox.data.HiCFileTools;
import juicebox.data.anchor.MotifAnchor;
import juicebox.data.anchor.MotifAnchorParser;
import juicebox.data.anchor.MotifAnchorTools;
import juicebox.data.feature.FeatureFilter;
import juicebox.data.feature.FeatureFunction;
import juicebox.data.feature.GenomeWideList;
import juicebox.tools.clt.CommandLineParserForJuicer;
import juicebox.tools.clt.JuicerCLT;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;

public class TriplesAPA extends JuicerCLT {
    public TriplesAPA() {
        super("ignore_for_now");
    }

    private static void removeAnchorsThatAreTooCloseTogether(GenomeWideList<MotifAnchor> anchors, final int filterSize) {
        MotifAnchorTools.mergeAndExpandSmallAnchors(anchors, filterSize);
        anchors.filterLists(new FeatureFilter<MotifAnchor>() {
            @Override
            public List<MotifAnchor> filter(String chr, List<MotifAnchor> featureList) {
                List<MotifAnchor> newAnchors = new ArrayList<>();
                for (MotifAnchor anchor : featureList) {
                    if (anchor.getWidth() == filterSize) {
                        newAnchors.add(anchor);
                    }
                }
                return newAnchors;
            }
        });
    }

    private static GenomeWideList<IntraChromTriple> readInIntraChromTriple(ChromosomeHandler handler, String triplesPath) {
        GenomeWideList<IntraChromTriple> triples = new GenomeWideList<>(handler);
        try (BufferedReader br = new BufferedReader(new FileReader(triplesPath))) {
            for (String line; (line = br.readLine()) != null; ) {
                IntraChromTriple triple = IntraChromTriple.parse(line, handler);
                if (triple != null) {
                    triples.addFeature(triple.getKey(), triple);
                }
            }
        } catch (Exception ignored) {
            ignored.printStackTrace();
        }
        return triples;
    }

    @Override
    protected void readJuicerArguments(String[] args, CommandLineParserForJuicer juicerParser) {

    }

    @Override
    public void run() {
        ChromosomeHandler handler = HiCFileTools.loadChromosomes("hg19");
        GenomeWideList<MotifAnchor> anchors = MotifAnchorParser.loadFromBEDFile(handler,
                "/Users/muhammadsaadshamim/Desktop/goodell_2/GrandCanyons.bed");

        //anchors = MotifAnchorParser.loadFromBEDFile(handler, "/Users/muhammadsaadshamim/Desktop/goodell_2/ComboSubClique.1.2.6.7.bed");

        GenomeWideList<IntraChromTriple> triples = readInIntraChromTriple(handler, "/Users/muhammadsaadshamim/Desktop/goodell_2/dimension_3");

        /*
        for(int tolerance = 0; tolerance <= 100; tolerance +=5) {
            GenomeWideList<IntraChromTriple> triples = readInIntraChromTriple(handler, "/Users/muhammadsaadshamim/Desktop/goodell_2/dimension_3");
            dedupTriples(triples, tolerance);
            System.out.println(tolerance+"\t"+triples.size());
        }
        */


        int interval = 300000;
        int numIntervals = 7;
        int filterSize = interval * numIntervals;
        removeAnchorsThatAreTooCloseTogether(anchors, filterSize);

        int translation = 000000;
        int tolerance = 20;

        translateAnchors(anchors, translation);


        System.out.println("width " + filterSize);
        System.out.println("trans " + translation);

        System.out.println("num anchors " + anchors.size());
        System.out.println("Pre dedup num triples " + triples.size());

        dedupTriples(triples, tolerance);
        System.out.println("Tol " + tolerance);
        System.out.println("Post dedup num triples " + triples.size());

        //threeDimHandling(triples, anchors, interval, numIntervals);

        threeDimSearching(triples, interval);

    }

    private void threeDimSearching(GenomeWideList<IntraChromTriple> triples, final int interval) {


        triples.processLists(new FeatureFunction<IntraChromTriple>() {
            @Override
            public void process(String chr, List<IntraChromTriple> tripleList) {
                final Map<String, Integer> plot3D = new HashMap<>();
                Collections.sort(tripleList);

                for (IntraChromTriple triple : tripleList) {
                    int diff_1 = (triple.getX1()) / interval;
                    int diff_2 = (triple.getX2()) / interval;
                    int diff_3 = (triple.getX3()) / interval;
                    if (Math.abs(diff_1 - diff_2) > 10 &&
                            Math.abs(diff_3 - diff_2) > 10 &&
                            Math.abs(diff_1 - diff_3) > 10) {
                        String key = makeKey(diff_1, diff_2, diff_3);
                        if (plot3D.containsKey(key)) {
                            plot3D.put(key, plot3D.get(key) + 1);
                        } else {
                            plot3D.put(key, 1);
                        }
                    }
                }
                //System.out.println( chr+" - "+internalCount);

                if (!plot3D.isEmpty()) {
                    int max = Collections.max(plot3D.values());
                    int min = Collections.min(plot3D.values());

                    int limit = Math.max((2 * max) / 3, min * 2);
                    System.out.println(chr + " max/min/limit vals: " + max + "/" + min + "/" + limit);
                    for (String key : plot3D.keySet()) {
                        int val = plot3D.get(key);
                        if (val > limit) {
                            String[] indices = key.split("\t");
                            int y1 = Integer.parseInt(indices[0]) * interval;
                            int y2 = Integer.parseInt(indices[1]) * interval;
                            int y3 = Integer.parseInt(indices[2]) * interval;
                            System.out.println(y1 + "," + y2 + "," + y3 + "\t" + val);
                        }
                    }
                }
            }
        });

    }

    private void threeDimHandling(GenomeWideList<IntraChromTriple> triples, final GenomeWideList<MotifAnchor> anchors, final int interval, int numInterval) {
        final int[] count = new int[1];
        count[0] = 0;

        final Map<String, Integer> plot3D = new HashMap<>();
        final int midNumPoint = numInterval / 2;

        triples.processLists(new FeatureFunction<IntraChromTriple>() {
            @Override
            public void process(String chr, List<IntraChromTriple> tripleList) {
                List<MotifAnchor> canyons = anchors.getFeatures(chr);
                Collections.sort(tripleList);
                Collections.sort(canyons);

                for (IntraChromTriple triple : tripleList) {

                    int i0 = -1, j0 = -1, k0 = -1;

                    for (int i = 0; i < canyons.size(); i++) {
                        if (canyons.get(i).contains(triple.getX1())) {
                            i0 = i;
                            break;
                        }
                    }

                    if (i0 > -1) {
                        for (int j = i0 + 1; j < canyons.size(); j++) {
                            if (canyons.get(j).contains(triple.getX2())) {
                                j0 = j;
                                break;
                            }
                        }
                        if (j0 > -1) {
                            for (int k = j0 + 1; k < canyons.size(); k++) {
                                if (canyons.get(k).contains(triple.getX3())) {
                                    k0 = k;
                                    break;
                                }
                            }
                            if (k0 > -1) {
                                int diff_1 = (triple.getX1() - canyons.get(i0).getX1()) / interval - midNumPoint;
                                int diff_2 = (triple.getX2() - canyons.get(j0).getX1()) / interval - midNumPoint;
                                int diff_3 = (triple.getX3() - canyons.get(k0).getX1()) / interval - midNumPoint;
                                String key = makeKey(diff_1, diff_2, diff_3);
                                if (plot3D.containsKey(key)) {
                                    plot3D.put(key, plot3D.get(key) + 1);
                                } else {
                                    plot3D.put(key, 1);
                                }
                                count[0]++;
                            }
                        }
                    }
                }
                //System.out.println( chr+" - "+internalCount);

            }
        });
        System.out.println("count " + count[0]);
        int max = Collections.max(plot3D.values());
        System.out.println("max val " + max);
        for (String key : plot3D.keySet()) {
            System.out.println(key + "\t" + plot3D.get(key));
        }
    }

    private String makeKey(int diff_1, int diff_2, int diff_3) {
        return "" + diff_1 + "\t" + diff_2 + "\t" + diff_3;
    }

    private void dedupTriples(GenomeWideList<IntraChromTriple> anchors, final int tolerance) {
        anchors.filterLists(new FeatureFilter<IntraChromTriple>() {
            @Override
            public List<IntraChromTriple> filter(String chr, List<IntraChromTriple> triplesList) {

                Collections.sort(triplesList);
                IntraChromTriple first = triplesList.get(0);
                TripleCentroid centroid = first.toTripleCentroid();

                List<IntraChromTriple> uniqueTriples = new ArrayList<>();

                for (IntraChromTriple triple : triplesList) {
                    if (centroid.hasOverlapWithTriple(triple, tolerance)) {
                        centroid.consumeDuplicate(triple);
                    } else {
                        uniqueTriples.add(centroid.toIntraChromTriple());
                        centroid = triple.toTripleCentroid();
                    }
                }
                uniqueTriples.add(centroid.toIntraChromTriple());

                return uniqueTriples;
            }
        });
    }

    private void translateAnchors(GenomeWideList<MotifAnchor> anchors, final int translation) {
        if (translation != 0) {
            anchors.filterLists(new FeatureFilter<MotifAnchor>() {
                @Override
                public List<MotifAnchor> filter(String chr, List<MotifAnchor> featureList) {

                    List<MotifAnchor> translated = new ArrayList<>();
                    for (MotifAnchor anchor : featureList) {
                        MotifAnchor anchorT = new MotifAnchor(anchor.getChr(), anchor.getX1() + translation, anchor.getX2() + translation);
                        if (anchorT.getX1() > 0) {
                            translated.add(anchorT);
                        }
                    }

                    return translated;
                }
            });
        }
    }
}
