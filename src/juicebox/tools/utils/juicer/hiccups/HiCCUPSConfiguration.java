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

package juicebox.tools.utils.juicer.hiccups;

import juicebox.data.Dataset;
import juicebox.data.HiCFileTools;
import juicebox.tools.clt.CommandLineParserForJuicer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by muhammadsaadshamim on 10/8/15.
 */
public class HiCCUPSConfiguration {
    final int windowWidth, peakWidth, clusterRadius;
    final double fdrThreshold;
    int resolution;

    public HiCCUPSConfiguration(int resolution, double fdrThreshold, int peakWidth, int windowWidth, int clusterRadius) {
        this.resolution = resolution;
        this.fdrThreshold = fdrThreshold;
        this.windowWidth = windowWidth;
        this.peakWidth = peakWidth;
        this.clusterRadius = clusterRadius;
    }

    public static List<HiCCUPSConfiguration> filterConfigurations(HiCCUPSConfiguration[] configurations, Dataset ds) {

        int[] resolutions = new int[configurations.length];
        for (int i = 0; i < configurations.length; i++) {
            resolutions[i] = configurations[i].resolution;
        }
        List<Integer> filteredResolutions = HiCFileTools.filterResolutions(ds, resolutions);

        // using map because duplicate resolutions will be removed while preserving order of respective configurations
        Map<Integer, HiCCUPSConfiguration> configurationMap = new HashMap<Integer, HiCCUPSConfiguration>();
        for (int i = 0; i < configurations.length; i++) {
            configurations[i].resolution = filteredResolutions.get(i);
            configurationMap.put(filteredResolutions.get(i), configurations[i]);
        }

        return new ArrayList<HiCCUPSConfiguration>(configurationMap.values());
    }

    /*
     * Reasonable Commands
     *
     * fdr = 0.10 for all resolutions
     * peak width = 1 for 25kb, 2 for 10kb, 4 for 5kb
     * window = 3 for 25kb, 5 for 10kb, 7 for 5kb
     *
     * cluster radius is 20kb for 5kb and 10kb res and 50kb for 25kb res
     * fdrsumthreshold is 0.02 for all resolutions
     * oeThreshold1 = 1.5 for all res
     * oeThreshold2 = 1.75 for all res
     * oeThreshold3 = 2 for all res
     *
     * published GM12878 looplist was only generated with 5kb and 10kb resolutions
     * same with published IMR90 looplist
     * published CH12 looplist only generated with 10kb
     */

    public static HiCCUPSConfiguration[] extractConfigurationsFromCommandLine(CommandLineParserForJuicer juicerParser) {
        int[] resolutions = HiCCUPSUtils.extractIntegerValues(juicerParser.getMultipleResolutionOptions(), -1, -1);
        double[] fdr = HiCCUPSUtils.extractFDRValues(juicerParser.getFDROptions(), resolutions.length, 0.1f); // becomes default 10
        int[] peaks = HiCCUPSUtils.extractIntegerValues(juicerParser.getPeakOptions(), resolutions.length, 2);
        int[] windows = HiCCUPSUtils.extractIntegerValues(juicerParser.getWindowOptions(), resolutions.length, 5);
        int[] radii = HiCCUPSUtils.extractIntegerValues(juicerParser.getClusterRadiusOptions(), resolutions.length, 20000);

        HiCCUPSConfiguration[] configurations = new HiCCUPSConfiguration[resolutions.length];
        for (int i = 0; i < resolutions.length; i++) {
            configurations[i] = new HiCCUPSConfiguration(resolutions[i], fdr[i], peaks[i], windows[i], radii[i]);
        }
        return configurations;
    }

    public int divisor() {
        return (windowWidth - peakWidth) * (windowWidth + peakWidth);
    }

    public int getResolution() {
        return resolution;
    }

    public int getClusterRadius() {
        return clusterRadius;
    }

    public int getWindowWidth() {
        return windowWidth;
    }

    public int getPeakWidth() {
        return peakWidth;
    }

    public double getFDRThreshold() {
        return fdrThreshold;
    }
}
