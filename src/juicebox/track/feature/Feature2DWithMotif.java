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
import juicebox.data.anchor.MotifAnchor;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;


/**
 * Created by muhammadsaadshamim on 11/4/15.
 */
public class Feature2DWithMotif extends Feature2D {

    public static int negReceived = 0, negWritten = 0, posNull = 0, posWritten = 0, negNull = 0;
    // true = +, false = -, null = NA
    private boolean strand1, strand2;
    // true - unique, false = inferred, null = NA
    private boolean unique1, unique2;
    private String sequence1, sequence2;
    private int motifStart1, motifEnd1, motifStart2, motifEnd2;
    private double score1, score2;
    private String MFS1 = "motif_start_1";
    private String MFE1 = "motif_end_1";
    private String MFSEQ1 = "sequence_1";
    private String MFO1 = "orientation_1";
    private String MFU1 = "uniqueness_1";
    private String MFS2 = "motif_start_2";
    private String MFE2 = "motif_end_2";
    private String MFSEQ2 = "sequence_2";
    private String MFO2 = "orientation_2";
    private String MFU2 = "uniqueness_2";
    public Feature2DWithMotif(String featureName, String chr1, int start1, int end1, String chr2, int start2, int end2,
                              Color c, Map<String, String> attributes) {
        super(featureName, chr1, start1, end1, chr2, start2, end2, c, attributes);

        searchAttributesForMotifInformation();

    }

    public void updateMotifData(boolean strand, boolean unique, String sequence, int motifStart, int motifEnd,
                                boolean dataBelongsToAnchor1, double score) {
        if (unique) {
            if (dataBelongsToAnchor1) {
                if (sequence1 == null) {//unique
                    this.strand1 = strand;
                    this.unique1 = unique;
                    this.sequence1 = sequence;
                    this.motifStart1 = motifStart;
                    this.motifEnd1 = motifEnd;
                    this.score1 = score;
                } else if (!(sequence.equals(sequence1) && motifStart1 == motifStart)) {//check equivalence for dups; otherwise not unique
                    if (HiCGlobals.printVerboseComments) {
                        System.err.println("Not unique motif1 - error\n" + this + "\n" +
                                motifStart + "\t" + motifEnd + "\t" + sequence + "\t" + strand + "\t" + unique);
                    }
                    sequence1 = "null";
                }
            } else {
                if (sequence2 == null) {//unique
                    negReceived++;
                    this.sequence2 = sequence;
                    this.strand2 = strand;
                    this.unique2 = unique;
                    this.motifStart2 = motifStart;
                    this.motifEnd2 = motifEnd;
                    this.score2 = score;
                } else if (!(sequence.equals(sequence2) && motifStart2 == motifStart)) {//check equivalence for dups; otherwise not unique
                    if (HiCGlobals.printVerboseComments) {
                        System.err.println("Not unique motif2 - error\n" + this + "\n" +
                                motifStart + "\t" + motifEnd + "\t" + sequence + "\t" + strand + "\t" + unique);
                    }
                    sequence2 = "null";
                }
            }
        } else {//inferred
            if (dataBelongsToAnchor1) {
                if (sequence1 == null || score > score1) {
                    this.strand1 = strand;
                    this.unique1 = unique;
                    this.sequence1 = sequence;
                    this.motifStart1 = motifStart;
                    this.motifEnd1 = motifEnd;
                    this.score1 = score;
                }
            } else {
                if (sequence2 == null || score > score2) {
                    this.strand2 = strand;
                    this.unique2 = unique;
                    this.motifStart2 = motifStart;
                    this.motifEnd2 = motifEnd;
                    this.score2 = score;
                }
            }
        }

    }

    private void searchAttributesForMotifInformation() {
        try {
            strand1 = getAttribute(MFO1).contains("p");
            strand2 = getAttribute(MFO2).contains("p");
            unique1 = getAttribute(MFU1).contains("u");
            unique2 = getAttribute(MFU2).contains("u");
            sequence1 = getAttribute(MFSEQ1);
            sequence2 = getAttribute(MFSEQ2);
            motifStart1 = Integer.parseInt(getAttribute(MFS1));
            motifEnd1 = Integer.parseInt(getAttribute(MFE1));
            motifStart2 = Integer.parseInt(getAttribute(MFS2));
            motifEnd2 = Integer.parseInt(getAttribute(MFE2));

            attributes.remove(MFS1);
            attributes.remove(MFE1);
            attributes.remove(MFSEQ1);
            attributes.remove(MFO1);
            attributes.remove(MFU1);

            attributes.remove(MFS2);
            attributes.remove(MFE2);
            attributes.remove(MFSEQ2);
            attributes.remove(MFO2);
            attributes.remove(MFU2);
        } catch (Exception e) {
            // attributes not present
        }

    }

    @Override
    public String getOutputFileHeader() {
        return super.getOutputFileHeader() + "\t" + MFS1 + "\t" + MFE1 + "\t" + MFSEQ1 + "\t" + MFO1 + "\t" + MFU1 + "\t" +
                MFS2 + "\t" + MFE2 + "\t" + MFSEQ2 + "\t" + MFO2 + "\t" + MFU2;
    }


    @Override
    public String toString() {
        String output = super.toString();

        if (sequence1 == null || sequence1.equals("null")) {
            posNull++;
            output += "\tNA\tNA\tNA\tNA\tNA";
        } else {
            posWritten++;
            String orientation = strand1 ? "p" : "n";
            String uniqueness = unique1 ? "u" : "i";
            output += "\t" + motifStart1 + "\t" + motifEnd1 + "\t" + sequence1 + "\t" + orientation + "\t" + uniqueness;
        }

        if (sequence2 == null || sequence2.equals("null")) {
            negNull++;
            output += "\tNA\tNA\tNA\tNA\tNA";
        } else {
            negWritten++;
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

            // always should be only uninitialized?
            if (onlyUninitializedFeatures) {
                if (sequence1 == null || sequence1.equals("null")) {
                    sequence1 = null;
                    anchors.add(new MotifAnchor(chr1, start1, end1, originalFeatures, emptyList));
                }
                if (sequence2 == null || sequence2.equals("null")) {
                    sequence2 = null;
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
