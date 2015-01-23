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

package juicebox.tools.clt;

import juicebox.tools.FragmentCalculation;
import juicebox.tools.HiCTools;
import org.broad.igv.Globals;
import org.broad.igv.feature.LocusScore;
import org.broad.igv.track.WindowFunction;
import org.broad.igv.util.ParsingUtils;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Pattern;


public class BPToFragment extends JuiceboxCLT {

    private String fragFile, inputBedFile, outputFile;

    @Override
    public void readArguments(String[] args, HiCTools.CommandLineParser parser) throws IOException {
        //setUsage("juicebox bpToFrag <fragmentFile> <inputBedFile> <outputFile>");
        if (args.length != 4) {
            throw new IOException("1");
        }

        fragFile = args[1];
        inputBedFile = args[2];
        outputFile = args[3];
    }

    @Override
    public void run() throws IOException {
        bpToFrag(fragFile, inputBedFile, outputFile);
    }

    private static void bpToFrag(String fragmentFile, String inputFile, String outputDir) throws IOException {
        BufferedReader fragmentReader = null;
        Pattern pattern = Pattern.compile("\\s");
        Map<String, int[]> fragmentMap = new HashMap<String, int[]>();  // Map of chr -> site positions
        try {
            fragmentReader = new BufferedReader(new FileReader(fragmentFile));

            String nextLine;
            while ((nextLine = fragmentReader.readLine()) != null) {
                String[] tokens = pattern.split(nextLine);

                // A hack, could use IGV's genome alias definitions
                String chr = getChrAlias(tokens[0]);

                int[] sites = new int[tokens.length];
                sites[0] = 0;  // Convenient convention
                for (int i = 1; i < tokens.length; i++) {
                    sites[i] = Integer.parseInt(tokens[i]) - 1;
                }
                fragmentMap.put(chr, sites);
            }
        } finally {
            assert fragmentReader != null;
            fragmentReader.close();
        }

        // inputFile contains a list of files or URLs.
        BufferedReader reader = null;
        try {
            File dir = new File(outputDir);
            if (!dir.exists() || !dir.isDirectory()) {
                System.out.println("Output directory does not exist, or is not directory");
                System.exit(1);
            }
            reader = new BufferedReader(new FileReader(inputFile));
            String nextLine;
            while ((nextLine = reader.readLine()) != null) {
                String path = nextLine.trim();
                int lastSlashIdx = path.lastIndexOf("/");
                if (lastSlashIdx < 0) lastSlashIdx = path.lastIndexOf("\\");  // Windows convention
                String fn = lastSlashIdx < 0 ? path : path.substring(lastSlashIdx);

                File outputFile = new File(dir, fn + ".sites");
                annotateWithSites(fragmentMap, path, outputFile);

            }
        } finally {
            if (reader != null) reader.close();
        }
    }

    /**
     * Find fragments that overlap the input bed file.  Its assumed the bed file is sorted by start position, otherwise
     * an exception is thrown.  If a fragment overlaps 2 or more bed featuers it is only outputted once.
     *
     * @param fragmentMap
     * @param bedFile
     * @param outputBedFile
     * @throws java.io.IOException
     */

    private static void annotateWithSites(Map<String, int[]> fragmentMap, String bedFile, File outputBedFile) throws IOException {


        BufferedReader bedReader = null;
        PrintWriter bedWriter = null;
        try {

            bedReader = ParsingUtils.openBufferedReader(bedFile);
            bedWriter = new PrintWriter(new BufferedWriter(new FileWriter(outputBedFile)));

            String nextLine;
            while ((nextLine = bedReader.readLine()) != null) {
                if (nextLine.startsWith("track") || nextLine.startsWith("browser") || nextLine.startsWith("#"))
                    continue;

                BedLikeFeature feature = new BedLikeFeature(nextLine);

                String[] tokens = Globals.whitespacePattern.split(nextLine);
                String chr = tokens[0];
                int start = Integer.parseInt(tokens[1]);
                int end = Integer.parseInt(tokens[2]);

                int[] sites = fragmentMap.get(feature.getChr());
                if (sites == null) continue;

                int firstSite = FragmentCalculation.binarySearch(sites, feature.getStart());
                int lastSite = FragmentCalculation.binarySearch(sites, feature.getEnd());

                bedWriter.print(chr + "\t" + start + "\t" + end + "\t" + firstSite + "\t" + lastSite);
                for (int i = 3; i < tokens.length; i++) {
                    bedWriter.print("\t" + tokens[i]);
                }
                bedWriter.println();


            }
        } finally {
            if (bedReader != null) bedReader.close();
            if (bedWriter != null) bedWriter.close();
        }
    }

    private static String getChrAlias(String token) {
        if (token.equals("MT")) {
            return "chrM";
        } else if (!token.startsWith("chr")) {
            return "chr" + token;
        } else {
            return token;
        }
    }

    private static class BedLikeFeature implements LocusScore {

        final String chr;
        final String line;
        int start;
        int end;
        String name;

        BedLikeFeature(String line) {
            this.line = line;
            String[] tokens = Globals.whitespacePattern.split(line);
            this.chr = tokens[0];
            this.start = Integer.parseInt(tokens[1]);
            this.end = Integer.parseInt(tokens[2]);
            if (tokens.length > 3) {
                this.name = name; // TODO - is this supposed to be this.name = tokens[x]? otherwise a redundant line
            }

        }

        @Override
        public String getValueString(double position, WindowFunction windowFunction) {
            return line;
        }

        public String getChr() {
            return chr;
        }

        public int getStart() {
            return start;
        }

        public void setStart(int start) {
            this.start = start;
        }

        public int getEnd() {
            return end;
        }

        public void setEnd(int end) {
            this.end = end;
        }

        public float getScore() {
            return 0;
        }
    }
}
