/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2017 Broad Institute, Aiden Lab
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
import juicebox.data.feature.GenomeWideList;
import juicebox.tools.clt.juicer.MotifFinder;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by muhammadsaadshamim on 1/19/16.
 * <p/>
 * Except for superloops, we don't observe long-range loops. Why not?
 * <p/>
 * The first possibility is that long-range loops do not form, either because:
 * <p/>
 * a) there is some mechanism that creates a hard cap on the length of loops,
 * such as the processivity of the excom, or
 * <p/>
 * b) given a convergent pair A/B separated by >2Mb,
 * there are too many competing ctcf motifs in between.
 * <p/>
 * Alternatively, loops do form between pairs of convergent CTCF sites that are far apart,
 * but those loops are too weak for us to see in our maps.
 * <p/>
 * A simple way to probe this is to do APA. Bin pairs of convergent loop anchors by 1d distance,
 * and then do APA on the pairs in each bin. You should get a strong apa score at 300kb.
 * what about 3mb? 30mb?
 */
class APAvsDistance {

    public static void main() {

        GenomeWideList<MotifAnchor> motifs = MotifAnchorParser.loadMotifsFromGenomeID("hg19", null);
        ChromosomeHandler handler = HiCFileTools.loadChromosomes("hg19");

        // read in all smc3, rad21, ctcf tracks and intersect them
        List<String> bedFiles = new ArrayList<>();

        File folder = new File("/users/name" + "directoryPath");
        File[] listOfFiles = folder.listFiles();

        for (File file : listOfFiles != null ? listOfFiles : new File[0]) {
            if (file.isFile()) {
                String path = file.getAbsolutePath();
                if (path.endsWith(".bed")) {
                    bedFiles.add(path);
                }
            }
        }

        GenomeWideList<MotifAnchor> proteins = MotifFinder.getIntersectionOfBEDFiles(handler, bedFiles);

        // preservative intersection of these protein list with motif list


        // extract positive anchors and negative anchors

        // create loops from all possible valid intersections

        // Feature2DList

        // bin loops by distance between loci


        // calculate APA score for each bin_list



        // plot APA score vs binned distance

    }

}
