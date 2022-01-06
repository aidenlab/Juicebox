/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2021 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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
 *  FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.tools.utils.juicer.hiccups;

import juicebox.data.HiCFileTools;
import juicebox.tools.clt.CommandLineParserForJuicer;
import juicebox.tools.utils.common.ArrayTools;
import juicebox.windowui.HiCZoom;

import java.text.DecimalFormat;
import java.util.*;

/**
 * Created by muhammadsaadshamim on 10/8/15.
 */
public class HiCCUPSConfiguration {
    private final int resolution;
    private int windowWidth, peakWidth, clusterRadius;
    private double fdrThreshold;

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
    public HiCCUPSConfiguration(int resolution, double fdrThreshold, int peakWidth, int windowWidth, int clusterRadius) {
        this.resolution = resolution;
        this.fdrThreshold = fdrThreshold;
        this.windowWidth = windowWidth;
        this.peakWidth = peakWidth;
        this.clusterRadius = clusterRadius;
    }

    public static List<HiCCUPSConfiguration> extractConfigurationsFromCommandLine(CommandLineParserForJuicer juicerParser,
                                                                                  List<HiCZoom> availableZooms, int version) {
        List<String> resString = juicerParser.getMultipleResolutionOptions();
        if (resString == null) return null;
        int[] resolutions = ArrayTools.extractIntegers(resString);

        Map<Integer, HiCCUPSConfiguration> configurationMap = new HashMap<>();
        for (int res : resolutions) {
            if (res == 1000) {
                configurationMap.put(res, getHiCCUPS2ConfigFor1K());
            } else if (res == 2000) {
                configurationMap.put(res, getHiCCUPS2ConfigFor2K());
            } else if (res == 5000) {
                if (version == 1) {
                    configurationMap.put(res, getDefaultConfigFor5K());
                } else {
                    configurationMap.put(res, getHiCCUPS2ConfigFor5K());
                }
            } else if (res == 10000) {
                configurationMap.put(res, getDefaultConfigFor10K());
            } else if (res == 25000) {
                configurationMap.put(res, getDefaultConfigFor25K());
            } else {
                configurationMap.put(res, getDefaultBlankConfig(res));
            }
        }

        double[] fdr = HiCCUPSUtils.extractFDRValues(juicerParser.getFDROptions(), resolutions.length, 0.1f); // becomes default 10
        int[] peaks = HiCCUPSUtils.extractIntegerValues(juicerParser.getPeakOptions(), resolutions.length);
        int[] windows = HiCCUPSUtils.extractIntegerValues(juicerParser.getWindowOptions(), resolutions.length);
        int[] radii = HiCCUPSUtils.extractIntegerValues(juicerParser.getClusterRadiusOptions(), resolutions.length);

        for (int i = 0; i < resolutions.length; i++) {
            if (fdr != null) configurationMap.get(resolutions[i]).fdrThreshold = fdr[i];
            if (peaks != null) configurationMap.get(resolutions[i]).peakWidth = peaks[i];
            if (windows != null) configurationMap.get(resolutions[i]).windowWidth = windows[i];
            if (radii != null) configurationMap.get(resolutions[i]).clusterRadius = radii[i];
        }

        Set<Integer> filteredResolutions = new HashSet<>(HiCFileTools.filterResolutions(availableZooms, resolutions));
        for (int res : resolutions) {
            if (!filteredResolutions.contains(res)) {
                System.err.println("Resolution " + res + " not available.");
            }
        }

        List<HiCCUPSConfiguration> validConfigs = new ArrayList<>();
        for (int res : filteredResolutions) {
            if (configurationMap.containsKey(res)) {
                HiCCUPSConfiguration config = configurationMap.get(res);
                if (config.isValid()) {
                    validConfigs.add(config);
                } else {
                    System.out.println("Discarding invalid configuration: " + config);
                }
            }
        }

        if (validConfigs.size() > 0) {
            System.out.println("Using the following configurations for HiCCUPS:");
            for (HiCCUPSConfiguration config : validConfigs) {
                System.out.println(config);
            }
        } else {
            return null;
        }

        return validConfigs;
    }

    public static HiCCUPSConfiguration getHiCCUPS2ConfigFor1K() {
        return new HiCCUPSConfiguration(1000, 10, 5, 10, 3000);
    }
    public static HiCCUPSConfiguration getHiCCUPS2ConfigFor2K() {
        return new HiCCUPSConfiguration(2000, 10, 4, 10, 6000);
    }

    public static HiCCUPSConfiguration getHiCCUPS2ConfigFor5K() {
        return new HiCCUPSConfiguration(5000, 10, 4, 7, 10000);
    }

    public static HiCCUPSConfiguration getDefaultConfigFor5K() {
        return new HiCCUPSConfiguration(5000, 10, 4, 7, 20000);
    }

    public static HiCCUPSConfiguration getDefaultConfigFor10K() {
        return new HiCCUPSConfiguration(10000, 10, 2, 5, 20000);
    }

    public static HiCCUPSConfiguration getDefaultConfigFor25K() {
        return new HiCCUPSConfiguration(25000, 10, 1, 3, 50000);
    }

    private static HiCCUPSConfiguration getDefaultBlankConfig(int res) {
        return new HiCCUPSConfiguration(res, 10, -1, -1, -1);
    }

    public static List<HiCCUPSConfiguration> getDefaultSetOfConfigsForUsers() {
        List<HiCCUPSConfiguration> configurations = new ArrayList<>();
        configurations.add(HiCCUPSConfiguration.getDefaultConfigFor5K());
        configurations.add(HiCCUPSConfiguration.getDefaultConfigFor10K());
        configurations.add(HiCCUPSConfiguration.getDefaultConfigFor25K());
        System.out.println("Default settings for 5kb, 10kb, and 25kb being used");
        return configurations;
    }

    public static List<HiCCUPSConfiguration> getHiCCUPS2DefaultSetOfConfigsForUsers() {
        List<HiCCUPSConfiguration> configurations = new ArrayList<>();
        configurations.add(HiCCUPSConfiguration.getHiCCUPS2ConfigFor1K());
        configurations.add(HiCCUPSConfiguration.getHiCCUPS2ConfigFor2K());
        configurations.add(HiCCUPSConfiguration.getHiCCUPS2ConfigFor5K());
        configurations.add(HiCCUPSConfiguration.getDefaultConfigFor10K());
        System.out.println("Default HiCCUPS2 settings for 1kb, 2kb, 5kb, and 10kb are being used");
        return configurations;
    }

    private boolean isValid() {
        return resolution > 0 && windowWidth > 0 && peakWidth > 0 && clusterRadius > 0 && windowWidth > peakWidth;
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

    public String toString() {
        return "Config res: " + resolution + " peak: " + peakWidth + " window: " + windowWidth +
                " fdr: " + getFDRPercent() + " radius: " + clusterRadius;
    }

    private String getFDRPercent() {
        DecimalFormat format = new DecimalFormat("#.##");
        return format.format(100. * (1. / fdrThreshold)) + "%";
    }
}
