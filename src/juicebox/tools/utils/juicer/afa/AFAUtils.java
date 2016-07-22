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

package juicebox.tools.utils.juicer.afa;

import juicebox.data.HiCFileTools;
import juicebox.data.MatrixZoomData;
import juicebox.track.feature.Feature2D;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.RealMatrix;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by muhammadsaadshamim on 9/9/15.
 */
class AFAUtils {
    public static List<Feature2D> filterFeaturesByAttribute(List<Feature2D> features, String attribute) {
        List<Feature2D> filteredFeatures = new ArrayList<Feature2D>();

        for (Feature2D feature : features) {
            if (feature.containsAttributeValue(attribute)) {
                filteredFeatures.add(feature);
            }
        }

        return filteredFeatures;
    }

    /*
     * loading filtered looplists categorized by attributes
    {
        Feature2DList featureList = Feature2DParser.loadFeatures(files[1], chromosomes, true,
                new FeatureFilter() {
                    // Remove duplicates and filters by size
                    // also save internal metrics for these measures
                    @Override
                    public List<Feature2D> filter(String chr, List<Feature2D> features) {

                        List<Feature2D> uniqueFeatures = new ArrayList<Feature2D>(new HashSet<Feature2D>(features));

                        List<Feature2D> filteredUniqueFeatures;
                        if (attribute.length() > 0) {
                            filteredUniqueFeatures = AFAUtils.filterFeaturesByAttribute(uniqueFeatures, attribute);
                        } else {
                            System.out.println("No filtering by attribute");
                            filteredUniqueFeatures = uniqueFeatures;
                        }

                        filterMetrics.put(chr,
                                new Integer[]{filteredUniqueFeatures.size(), uniqueFeatures.size(), features.size()});

                        return filteredUniqueFeatures;
                    }
                }, false);
    }
    */

    public static RealMatrix extractLocalizedData(MatrixZoomData zd, Feature2D feature, int L, int resolution, int window,
                                                  NormalizationType norm, LocationType relativeLocation) throws IOException {

        int loopX, loopY;
        if (relativeLocation.equals(LocationType.TL)) {
            loopX = feature.getStart1();
            loopY = feature.getStart2();
        } else if (relativeLocation.equals(LocationType.BR)) {
            loopX = feature.getEnd1();
            loopY = feature.getEnd2();
        } else {//LocationType.CENTER
            loopX = feature.getMidPt1();
            loopY = feature.getMidPt2();
        }

        loopX /= resolution;
        loopY /= resolution;

        int binXStart = loopX - window;
        int binXEnd = loopX + (window + 1);
        int binYStart = loopY - window;
        int binYEnd = loopY + (window + 1);

        return HiCFileTools.extractLocalBoundedRegion(zd, binXStart, binXEnd, binYStart, binYEnd, L, L, norm);
    }
}
