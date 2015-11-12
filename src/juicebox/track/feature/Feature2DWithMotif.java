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

package juicebox.track.feature;

import juicebox.HiCGlobals;
import juicebox.track.anchor.MotifAnchor;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;


/**
 * Created by muhammadsaadshamim on 11/4/15.
 */
public class Feature2DWithMotif extends Feature2D {

    // true = +, false = -, null = NA
    private boolean strand1, strand2;
    // true - unique, false = inferred, null = NA
    private boolean unique1, unique2;
    private String sequence1, sequence2;
    private int motifStart1, motifEnd1, motifStart2, motifEnd2;

    public Feature2DWithMotif(String featureName, String chr1, int start1, int end1, String chr2, int start2, int end2,
                              Color c, Map<String, String> attributes) {
        super(featureName, chr1, start1, end1, chr2, start2, end2, c, attributes);
    }

    public Feature2DWithMotif(String featureName, String chr1, int start1, int end1, String chr2, int start2, int end2,
                              Color c, Map<String, String> attributes,
                              boolean strand1, boolean unique1, String sequence1, int motifStart1, int motifEnd1,
                              boolean strand2, boolean unique2, String sequence2, int motifStart2, int motifEnd2) {
        super(featureName, chr1, start1, end1, chr2, start2, end2, c, attributes);
        updateMotifData(strand1, unique1, sequence1, motifStart1, motifEnd1, true); // motif1
        updateMotifData(strand2, unique2, sequence2, motifStart2, motifEnd2, false); // motif2
    }

    public void updateMotifData(boolean strand, boolean unique, String sequence, int motifStart, int motifEnd,
                                boolean dataBelongsToAnchor1) {
        if (unique) {
            if (dataBelongsToAnchor1 && strand) {
                if (sequence1 == null) {
                    this.strand1 = strand;
                    this.unique1 = unique;
                    this.sequence1 = sequence;
                    this.motifStart1 = motifStart;
                    this.motifEnd1 = motifEnd;
                } else {
                    if (HiCGlobals.printVerboseComments) {
                        System.err.println("Not unique motif1 - error\n" + this + "\n" +
                                motifStart + "\t" + motifEnd + "\t" + sequence + "\t" + strand + "\t" + unique);
                    }
                    sequence1 = "null";
                }
            } else {
                if (sequence2 == null && !strand) {
                    this.strand2 = strand;
                    this.unique2 = unique;
                    this.motifStart2 = motifStart;
                    this.motifEnd2 = motifEnd;
                } else {
                    if (HiCGlobals.printVerboseComments) {
                        System.err.println("Not unique motif2 - error\n" + this + "\n" +
                                motifStart + "\t" + motifEnd + "\t" + sequence + "\t" + strand + "\t" + unique);
                    }
                    sequence2 = "null";
                }
            }
        } else {
            if (dataBelongsToAnchor1 && strand) {
                if (sequence1 == null) {
                    this.strand1 = strand;
                    this.unique1 = unique;
                    this.sequence1 = sequence;
                    this.motifStart1 = motifStart;
                    this.motifEnd1 = motifEnd;
                } else {
                    if (HiCGlobals.printVerboseComments) {
                        System.err.println("Not unique motif1 - error\n" + this + "\n" +
                                motifStart + "\t" + motifEnd + "\t" + sequence + "\t" + strand + "\t" + unique);
                    }
                    sequence1 = "null";
                }
            } else {
                if (sequence2 == null && !strand) {
                    this.strand2 = strand;
                    this.unique2 = unique;
                    this.motifStart2 = motifStart;
                    this.motifEnd2 = motifEnd;
                } else {
                    if (HiCGlobals.printVerboseComments) {
                        System.err.println("Not unique motif2 - error\n" + this + "\n" +
                                motifStart + "\t" + motifEnd + "\t" + sequence + "\t" + strand + "\t" + unique);
                    }
                    sequence2 = "null";
                }
            }
        }

    }

    @Override
    public String getOutputFileHeader() {
        return super.getOutputFileHeader() +
                "\tmotif_start1\tmotif_end1\tsequence_1\torientation_1\tuniqueness_1" +
                "\tmotif_start2\tmotif_end2\tsequence2\torientation_2\tuniqueness_2";
    }


    @Override
    public String toString() {
        String output = super.toString();

        if (sequence1 == null || sequence1.equals("null")) {
            output += "\tNA\tNA\tNA\tNA\tNA";
        } else {
            String orientation = strand1 ? "p" : "n";
            String uniqueness = unique1 ? "u" : "i";
            output += "\t" + motifStart1 + "\t" + motifEnd1 + "\t" + sequence1 + "\t" + orientation + "\t" + uniqueness;
        }

        if (sequence2 == null || sequence2.equals("null")) {
            output += "\tNA\tNA\tNA\tNA\tNA";
        } else {
            String orientation = strand2 ? "p" : "n";
            String uniqueness = unique2 ? "u" : "i";
            output += "\t" + motifStart2 + "\t" + motifEnd2 + "\t" + sequence2 + "\t" + orientation + "\t" + uniqueness;
        }

        return output;
    }

    public List<MotifAnchor> getAnchors(boolean onlyUninitializedFeatures) {
        java.util.List<Feature2DWithMotif> originalFeatures = new ArrayList<Feature2DWithMotif>();
        originalFeatures.add(this);

        java.util.List<MotifAnchor> anchors = new ArrayList<MotifAnchor>();
        if (isOnDiagonal()) {
            // loops should not be on diagonal
            // anchors.add(new MotifAnchor(chr1, start1, end1, originalFeatures, originalFeatures));
        } else {
            java.util.List<Feature2DWithMotif> emptyList = new ArrayList<Feature2DWithMotif>();

            if (onlyUninitializedFeatures) {
                if (sequence1 == null) {
                    anchors.add(new MotifAnchor(chr1, start1, end1, originalFeatures, emptyList));
                }
                if (sequence2 == null) {
                    anchors.add(new MotifAnchor(chr2, start2, end2, emptyList, originalFeatures));
                }
            } else {
                anchors.add(new MotifAnchor(chr1, start1, end1, originalFeatures, emptyList));
                anchors.add(new MotifAnchor(chr2, start2, end2, emptyList, originalFeatures));
            }
        }
        return anchors;
    }
}
