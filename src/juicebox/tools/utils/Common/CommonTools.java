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

package juicebox.tools.utils.Common;

import juicebox.data.Dataset;
import juicebox.tools.chrom.sizes.ChromosomeSizes;
import juicebox.track.Feature2D;
import juicebox.windowui.HiCZoom;
import org.broad.igv.feature.Chromosome;

import java.io.*;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;


public class CommonTools {
    /**
     * Load chromosomes from given ID or file name.
     *
     * @param idOrFile Genome ID or file name where chromosome lengths written
     * @return Chromosome lengths
     * @throws java.io.IOException if chromosome length file not found
     */
    public static List<Chromosome> loadChromosomes(String idOrFile) throws IOException {

        InputStream is = null;

        try {
            // Note: to get this to work, had to edit Intellij settings
            // so that "?*.sizes" are considered sources to be copied to class path
            is = ChromosomeSizes.class.getResourceAsStream(idOrFile + ".chrom.sizes");

            if (is == null) {
                // Not an ID,  see if its a file
                File file = new File(idOrFile);

                if (file.exists()) {
                    is = new FileInputStream(file);
                } else {
                    throw new FileNotFoundException("Could not find chromosome sizes file for: " + idOrFile);
                }

            }

            List<Chromosome> chromosomes = new ArrayList<Chromosome>();
            chromosomes.add(0, null);   // Index 0 reserved for "whole genome" pseudo-chromosome

            Pattern pattern = Pattern.compile("\t");
            BufferedReader reader = new BufferedReader(new InputStreamReader(is));
            String nextLine;
            long genomeLength = 0;
            int idx = 1;

            while ((nextLine = reader.readLine()) != null) {
                String[] tokens = pattern.split(nextLine);
                if (tokens.length == 2) {
                    String name = tokens[0];
                    int length = Integer.parseInt(tokens[1]);
                    genomeLength += length;
                    chromosomes.add(idx, new Chromosome(idx, name, length));
                    idx++;
                } else {
                    System.out.println("Skipping " + nextLine);
                }
            }

            // Add the "pseudo-chromosome" All, representing the whole genome.  Units are in kilo-bases
            chromosomes.set(0, new Chromosome(0, "All", (int) (genomeLength / 1000)));


            return chromosomes;
        } finally {
            if (is != null) is.close();
        }
    }

    public static HiCZoom getZoomLevel(Dataset ds, int resolution) {
        List<HiCZoom> resolutions = ds.getBpZooms();
        HiCZoom zoom = resolutions.get(0);
        int currentDistance = Math.abs(zoom.getBinSize() - resolution);
        // Loop through resolutions
        for (HiCZoom subZoom : resolutions) {
            int newDistance = Math.abs(subZoom.getBinSize() - resolution);
            if (newDistance < currentDistance) {
                currentDistance = newDistance;
                zoom = subZoom;
            }
        }
        System.out.println("Adjusting resolution to " + zoom.getBinSize());
        return zoom;
    }

    /**
     * Set intersection
     *
     * http://stackoverflow.com/questions/7574311/efficiently-compute-intersection-of-two-sets-in-java
     *
     * @param set1
     * @param set2
     * @return
     */
    public static Set<Chromosome> getSetIntersection (Set<Chromosome> set1, Set<Chromosome> set2) {
        boolean set1IsLarger = set1.size() > set2.size();
        Set<Chromosome> cloneSet = new HashSet<Chromosome>(set1IsLarger ? set2 : set1);
        cloneSet.retainAll(set1IsLarger ? set1 : set2);
        return cloneSet;
    }

    public static Set<Chromosome> stringToChromosomes(Set<String> chromosomesSpecified,
                                                      List<Chromosome> referenceChromosomes) {

        Set<Chromosome> chrKeys = new HashSet<Chromosome>(referenceChromosomes);
        Set<String> strKeys = new HashSet<String>(chromosomesSpecified);
        Set<Chromosome> convertedChromosomes = new HashSet<Chromosome>();

        // filter down loops by uniqueness, then size, and save the totals at each stage
        for (Chromosome chrKey : chrKeys) {
            for (String strKey : strKeys) {
                if (equivalentChromosome(strKey, chrKey)) {
                    convertedChromosomes.add(chrKey);
                    strKeys.remove(strKey);
                    break;
                }
            }
        }

        return new HashSet<Chromosome>(convertedChromosomes);
    }

    /**
     * Evaluates whether the same chromosome is being referenced by the token
     *
     * @param token
     * @param chr
     * @return
     */
    public static boolean equivalentChromosome(String token, Chromosome chr) {
        String token2 = token.toLowerCase().replaceAll("chr", "");
        String chrName = chr.getName().toLowerCase().replaceAll("chr", "");
        return token2.equals(chrName);
    }

    /**
     *
     * @param fileName
     * @return
     */
    public static PrintWriter openWriter(String fileName) {
        try {
            File file = new File(fileName);
            PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(file)), true);
            return out;
        } catch (IOException e) {
            System.out.println("I/O error opening file: "+fileName);
            System.exit(0);
        }
        return null;
    }
}
