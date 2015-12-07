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

import juicebox.data.anchor.MotifAnchor;
import juicebox.tools.clt.juicer.CompareLists;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;


/**
 * Created by muhammadsaadshamim on 11/4/15.
 */
public class Feature2DWithMotif extends Feature2D {

    public static int negReceived = 0, negWritten = 0, posNull = 0, posWritten = 0, negNull = 0;
    public static boolean useSimpleOutput = false;
    public static boolean uniquenessCheckEnabled = true;
    public static boolean lenientEqualityEnabled = false;
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
    private int chr1Index, chr2Index;

    public Feature2DWithMotif(FeatureType featureName, String chr1, int chr1Index, int start1, int end1,
                              String chr2, int chr2Index, int start2, int end2, Color c, Map<String, String> attributes) {
        super(featureName, chr1, start1, end1, chr2, start2, end2, c, attributes);
        this.chr1Index = chr1Index;
        this.chr2Index = chr2Index;

        importAttributesForMotifInformation();

    }


    public void updateMotifData(boolean strand, boolean unique, String sequence, int motifStart, int motifEnd,
                                boolean dataBelongsToAnchor1, double score) {
        if (unique) {
            if (dataBelongsToAnchor1) {
                if (sequence1 == null || score > score1) {//unique
                    this.strand1 = strand;
                    this.unique1 = unique;
                    this.sequence1 = sequence;
                    this.motifStart1 = motifStart;
                    this.motifEnd1 = motifEnd;
                    this.score1 = score;
                }/* else if (!(sequence.equals(sequence1) && motifStart1 == motifStart)) {//check equivalence for dups; otherwise not unique
                    sequence1 = "null";
                }*/
            } else {
                if (sequence2 == null || score > score2) {//unique
                    negReceived++;
                    this.sequence2 = sequence;
                    this.strand2 = strand;
                    this.unique2 = unique;
                    this.motifStart2 = motifStart;
                    this.motifEnd2 = motifEnd;
                    this.score2 = score;
                }/* else if (!(sequence.equals(sequence2) && motifStart2 == motifStart)) {//check equivalence for dups; otherwise not unique
                    sequence2 = "null";
                }*/
            }
        } else {//inferred
            if (dataBelongsToAnchor1 && strand) {
                if (sequence1 == null) {
                    this.strand1 = strand;
                    this.unique1 = unique;
                    this.sequence1 = sequence;
                    this.motifStart1 = motifStart;
                    this.motifEnd1 = motifEnd;
                    this.score1 = score;
                } else if (!(sequence.equals(sequence1) && motifStart1 == motifStart)) {
                    sequence1 = "null";
                }
            } else if (!dataBelongsToAnchor1 && !strand) {
                if (sequence2 == null) {
                    this.strand2 = strand;
                    this.unique2 = unique;
                    this.sequence2 = sequence;
                    this.motifStart2 = motifStart;
                    this.motifEnd2 = motifEnd;
                    this.score2 = score;
                } else if (!(sequence.equals(sequence2) && motifStart2 == motifStart)) {
                    sequence2 = "null";
                }
            }
        }
    }

    private void importAttributesForMotifInformation() {
        try {
            boolean strand1 = getAttribute(MFO1).contains("p") || getAttribute(MFO1).contains("+");
            boolean unique1 = getAttribute(MFU1).contains("u");
            String sequence1 = getAttribute(MFSEQ1);
            int motifStart1 = Integer.parseInt(getAttribute(MFS1));
            int motifEnd1 = Integer.parseInt(getAttribute(MFE1));

            attributes.remove(MFO1);
            attributes.remove(MFU1);
            attributes.remove(MFS1);
            attributes.remove(MFE1);
            attributes.remove(MFSEQ1);

            // done last so that all info must be complete first
            // incomplete motifs will exit via catch before this point
            this.strand1 = strand1;
            this.unique1 = unique1;
            this.sequence1 = sequence1;
            this.motifStart1 = motifStart1;
            this.motifEnd1 = motifEnd1;
        } catch (Exception e) {
            // attributes not present
        }
        try {
            boolean strand2 = getAttribute(MFO2).contains("p") || getAttribute(MFO2).contains("+");
            boolean unique2 = getAttribute(MFU2).contains("u");
            String sequence2 = getAttribute(MFSEQ2);
            int motifStart2 = Integer.parseInt(getAttribute(MFS2));
            int motifEnd2 = Integer.parseInt(getAttribute(MFE2));

            attributes.remove(MFO2);
            attributes.remove(MFU2);
            attributes.remove(MFSEQ2);
            attributes.remove(MFS2);
            attributes.remove(MFE2);

            // done last so that all info must be complete first
            // incomplete motifs will exit via catch before this point
            this.strand2 = strand2;
            this.unique2 = unique2;
            this.sequence2 = sequence2;
            this.motifStart2 = motifStart2;
            this.motifEnd2 = motifEnd2;
        } catch (Exception e) {
            // attributes not present
        }
    }

    @Override
    public String getOutputFileHeader() {
        String additionalAttributes = "\t" + MFS1 + "\t" + MFE1 + "\t" + MFSEQ1 + "\t" + MFO1 + "\t" + MFU1 + "\t" +
                MFS2 + "\t" + MFE2 + "\t" + MFSEQ2 + "\t" + MFO2 + "\t" + MFU2;
        if (useSimpleOutput) {
            if (attributes.containsKey(CompareLists.PARENT_ATTRIBUTE)) {
                return genericHeader + "\t" + CompareLists.PARENT_ATTRIBUTE + additionalAttributes;
            } else {
                return genericHeader + additionalAttributes;
            }
        }
        return super.getOutputFileHeader() + additionalAttributes;
    }

    @Override
    public String toString() {
        String output = super.toString();
        if (useSimpleOutput) {
            output = simpleString();
            if (attributes.containsKey(CompareLists.PARENT_ATTRIBUTE)) {
                output += "\t" + attributes.get(CompareLists.PARENT_ATTRIBUTE);
            }
        }

        if (sequence1 == null) {
            posNull++;
            output += "\tNA\tNA\tNA\tNA\tNA";
        } else if (sequence1.equals("null")) {
            posNull++;
            output += "\tna\tna\tna\tna\tna";
        } else {
            posWritten++;
            String orientation = strand1 ? "+" : "-";
            String uniqueness = unique1 ? "u" : "i";
            output += "\t" + motifStart1 + "\t" + motifEnd1 + "\t" + sequence1 + "\t" + orientation + "\t" + uniqueness;
        }

        if (sequence2 == null) {
            negNull++;
            output += "\tNA\tNA\tNA\tNA\tNA";
        } else if (sequence2.equals("null")) {
            negNull++;
            output += "\tna\tna\tna\tna\tna";
        } else {
            negWritten++;
            String orientation = strand2 ? "+" : "-";
            String uniqueness = unique2 ? "u" : "i";
            output += "\t" + motifStart2 + "\t" + motifEnd2 + "\t" + sequence2 + "\t" + orientation + "\t" + uniqueness;
        }

        return output;
    }

    public List<MotifAnchor> getAnchors(boolean onlyUninitializedFeatures) {
        List<Feature2DWithMotif> originalFeatures = new ArrayList<Feature2DWithMotif>();
        originalFeatures.add(this);

        List<MotifAnchor> anchors = new ArrayList<MotifAnchor>();
        if (isOnDiagonal()) {
            // loops should not be on diagonal
            // anchors.add(new MotifAnchor(chr1, start1, end1, originalFeatures, originalFeatures));
        } else {
            List<Feature2DWithMotif> emptyList = new ArrayList<Feature2DWithMotif>();

            // always should be only uninitialized?
            if (onlyUninitializedFeatures) {
                if (sequence1 == null || sequence1.equals("null")) {
                    sequence1 = null;
                    anchors.add(new MotifAnchor(chr1Index, start1, end1, originalFeatures, emptyList));
                }
                if (sequence2 == null || sequence2.equals("null")) {
                    sequence2 = null;
                    anchors.add(new MotifAnchor(chr2Index, start2, end2, emptyList, originalFeatures));
                }
            } else {
                anchors.add(new MotifAnchor(chr1Index, start1, end1, originalFeatures, emptyList));
                anchors.add(new MotifAnchor(chr2Index, start2, end2, emptyList, originalFeatures));
            }
        }
        return anchors;
    }

    @Override
    public boolean equals(Object obj) {
        if (super.equals(obj)) {
            Feature2DWithMotif o = (Feature2DWithMotif) obj;
            try {
                if ((sequence1 == null && o.sequence1 == null) || sequence1.equals(o.sequence1)) {
                    if ((sequence2 == null && o.sequence2 == null) || sequence2.equals(o.sequence2)) {
                        if (unique1 == o.unique1 && unique2 == o.unique2) {
                            if (strand1 == o.strand1 && strand2 == o.strand2) {
                                return true;
                            }
                        }
                    }
                }
            } catch (Exception e) {
            }
            if (lenientEqualityEnabled) {
                // assuming this is B, obj is reference
                boolean motifsAreEqual = true;

                // reference has more data
                if (o.sequence1 != null && sequence1 == null) {
                    // only if reference also follows convergent rule
                    if (o.strand1) {
                        motifsAreEqual = false;
                    }
                }
                if (o.sequence2 != null && sequence2 == null) {
                    // only if reference also follows convergent rule
                    if (!o.strand2) {
                        motifsAreEqual = false;
                    }
                }

                // actually different data
                if (o.sequence1 != null && sequence1 != null && !sequence1.equals(o.sequence1)) {
                    // only if reference also follows convergent rule
                    if (o.strand1) {
                        motifsAreEqual = false;
                    }
                }
                if (o.sequence2 != null && sequence2 != null && !sequence2.equals(o.sequence2)) {
                    // only if reference also follows convergent rule
                    if (!o.strand2) {
                        motifsAreEqual = false;
                    }
                }

                /*
                if(o.sequence1 != null && sequence1 != null && sequence1.equals(o.sequence1)){
                    if(unique1 != o.unique1){
                        motifsAreEqual = false;
                    }
                }
                if(o.sequence2 != null && sequence2 != null && sequence2.equals(o.sequence2)){
                    if(unique2 != o.unique2){
                        motifsAreEqual = false;
                    }
                }
                */

                return motifsAreEqual;
            }
        }

        return false;
    }

    @Override
    public int hashCode() {
        int hash = super.hashCode();
        if (sequence1 != null) hash = 51 * hash + sequence1.hashCode();
        if (sequence2 != null) hash = 53 * hash + sequence2.hashCode();

        return hash;
    }

    public int getConvergenceStatus() {

        // ++, +- (convergent), -+ (divergent), --, other (incomplete)

        if (sequence1 != null && sequence2 != null) {
            if (unique1 && unique2) {
                if (strand1) {
                    if (strand2) {
                        return 0;
                    } else {
                        return 1;
                    }
                } else {
                    if (strand2) {
                        return 2;
                    } else {
                        return 3;
                    }
                }
            }
        }

        return 4;
    }
}
