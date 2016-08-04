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

package juicebox.tools.dev;

import htsjdk.samtools.seekablestream.SeekableHTTPStream;
import juicebox.HiCGlobals;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by muhammadsaadshamim on 8/3/16.
 */
public class GeneTools {


    public static BufferedReader getStreamToGeneFile(String genomeID) throws MalformedURLException {
        String path = "http://hicfiles.s3.amazonaws.com/internal/" + genomeID + "_refGene.txt";
        SeekableHTTPStream stream = new SeekableHTTPStream(new URL(path));
        return new BufferedReader(new InputStreamReader(stream), HiCGlobals.bufferSize);
    }

    public static Map<String, GeneLocation> getLocationMap(BufferedReader reader) throws IOException {
        Map<String, GeneLocation> geneLocationHashMap = new HashMap<String, GeneLocation>();
        new HashMap<String, GeneLocation>();
        String nextLine;
        while ((nextLine = reader.readLine()) != null) {
            String[] values = nextLine.split(" ");
            GeneLocation location = new GeneLocation(values[2].trim(), Integer.valueOf(values[3].trim()));
            geneLocationHashMap.put(values[0].trim().toLowerCase(), location);
            geneLocationHashMap.put(values[1].trim().toLowerCase(), location);
        }
        return geneLocationHashMap;
    }
}
