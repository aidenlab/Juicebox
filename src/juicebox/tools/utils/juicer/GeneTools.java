/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2016 Broad Institute, Aiden Lab
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

package juicebox.tools.utils.juicer;

import juicebox.data.ChromosomeHandler;
import juicebox.data.GeneLocation;
import juicebox.data.anchor.MotifAnchor;
import juicebox.data.anchor.MotifAnchorParser;
import juicebox.data.feature.GenomeWideList;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by muhammadsaadshamim on 8/3/16.
 */
public class GeneTools {


    public static BufferedReader getStreamToGeneFile(String genomeID) throws MalformedURLException {
        String path = extractProperGeneFilePath(genomeID);
        try {
            return new BufferedReader(new FileReader(path));
        } catch (FileNotFoundException e) {
            System.err.println("Unable to read from " + path);
            System.exit(56);
        }
        return null;
    }

    public static String extractProperGeneFilePath(String genomeID) {
        String newURL = "http://hicfiles.s3.amazonaws.com/internal/" + genomeID + "_refGene.txt";
        try {
            return MotifAnchorParser.downloadFromUrl(new URL(newURL), "genes");
        } catch (IOException e) {
            System.err.println("Unable to download file from online; attempting to use direct file path");
        }
        return genomeID;
    }

    public static Map<String, GeneLocation> getLocationMap(BufferedReader reader) throws IOException {
        Map<String, GeneLocation> geneLocationHashMap = new HashMap<String, GeneLocation>();

        String nextLine;
        while ((nextLine = reader.readLine()) != null) {
            String[] values = nextLine.split(" ");
            GeneLocation location = new GeneLocation(values[2].trim(), Integer.valueOf(values[3].trim()));
            geneLocationHashMap.put(values[0].trim().toLowerCase(), location);
            geneLocationHashMap.put(values[1].trim().toLowerCase(), location);
        }
        return geneLocationHashMap;
    }

    public static GenomeWideList<MotifAnchor> parseGenome(String genomeID, ChromosomeHandler handler) throws Exception {
        BufferedReader reader = getStreamToGeneFile(genomeID);
        List<MotifAnchor> allGenes = extractAllGenes(reader, handler);
        return new GenomeWideList<MotifAnchor>(handler, allGenes);
    }

    private static List<MotifAnchor> extractAllGenes(BufferedReader reader, ChromosomeHandler handler)
            throws IOException {
        List<MotifAnchor> genes = new ArrayList<MotifAnchor>();

        String nextLine;
        while ((nextLine = reader.readLine()) != null) {
            String[] values = nextLine.split(" ");
            int chrIndex = handler.getChr(values[2]).getIndex();
            int position = Integer.valueOf(values[3].trim());
            String name = values[1].trim();
            MotifAnchor gene = new MotifAnchor(chrIndex, position - 1, position + 1, name);
            genes.add(gene);
        }

        return genes;
    }
}
