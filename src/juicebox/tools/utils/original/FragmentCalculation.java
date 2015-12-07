/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2014 Broad Institute, Aiden Lab
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

package juicebox.tools.utils.original;

import juicebox.HiCGlobals;

import java.io.*;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * @author nchernia
 *         Date: 8/26/12
 */
public class FragmentCalculation {

    private Map<String, int[]> sitesMap = null;

    private FragmentCalculation(Map<String, int[]> sitesMap) {
        this.sitesMap = sitesMap;
    }

    private static FragmentCalculation readFragments(InputStream is) throws IOException {
        Pattern pattern = Pattern.compile("\\s");
        BufferedReader reader = new BufferedReader(new InputStreamReader(is), HiCGlobals.bufferSize);
        String nextLine;
        Map<String, int[]> sitesMap = new LinkedHashMap<String, int[]>();

        while ((nextLine = reader.readLine()) != null) {
            String[] tokens = pattern.split(nextLine);
            if (tokens.length > 1) {
                String key = tokens[0];
                int[] sites = new int[tokens.length - 1];
                for (int i = 1; i < tokens.length; i++) {
                    sites[i - 1] = Integer.parseInt(tokens[i]);
                }

                sitesMap.put(key, sites);
            } else {
                System.out.println("Skipping line: " + nextLine);
            }
        }

        return new FragmentCalculation(sitesMap);
    }

    public static FragmentCalculation readFragments(String filename) throws IOException {
        InputStream is = null;
        try {
            File file = new File(filename);
            is = new FileInputStream(file);
            return readFragments(is);
        } finally {
            assert is != null;
            is.close();
        }

    }

    /**
     * Return fragment that this position lies on.  Fragment 0 means position < sites[0].
     * Fragment 1 means position >= sites[0] and < sites[1].
     *
     * @param sites    The sorted array of fragment sites for the chromosome
     * @param position The position to search for within that array
     * @return The fragment location such that position >= sites[retVal-1] and position <  sites[retVal]
     */
    public static int binarySearch(int[] sites, int position) {
        int lo = 0;
        int hi = sites.length - 1;
        while (lo <= hi) {
            // Base case - found range
            int mid = lo + (hi - lo) / 2;

            if (position > sites[mid]) lo = mid + 1;
            else if (position < sites[mid]) hi = mid - 1;
            else return mid + 1;
        }
        return lo;
    }

    public int[] getSites(String chrName) {
        return sitesMap.get(chrName);
    }

    public int getNumberFragments(String chrName) {
        int[] sites = sitesMap.get(chrName);
        if (sites == null) // for "All"
            return 1;
        return sites.length;
    }

    public Map<String, int[]> getSitesMap() {
        return sitesMap;
    }
}
