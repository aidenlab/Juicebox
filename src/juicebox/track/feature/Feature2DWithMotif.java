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

package juicebox.track.feature;

import juicebox.data.ChromosomeHandler;
import juicebox.data.anchor.MotifAnchor;
import juicebox.tools.clt.juicer.CompareLists;

import java.awt.*;
import java.util.List;
import java.util.*;


/**
 * Created by muhammadsaadshamim on 11/4/15.
 */
public class Feature2DWithMotif extends Feature2D {

    public static boolean useSimpleOutput = false;
    public static boolean uniquenessCheckEnabled = true;
    public static boolean lenientEqualityEnabled = false;
    private static int negReceived = 0;
    private static int negWritten = 0;
    private static int posNull = 0;
    private static int posWritten = 0;
    private static int negNull = 0;
    private final String MFS1 = "motif_start_1";
    private final String MFE1 = "motif_end_1";
    private final String MFSEQ1 = "sequence_1";
    private final String MFO1 = "orientation_1";
    private final String MFU1 = "uniqueness_1";
    private final String MFS2 = "motif_start_2";
    private final String MFE2 = "motif_end_2";
    private final String MFSEQ2 = "sequence_2";
    private final String MFO2 = "orientation_2";
    private final String MFU2 = "uniqueness_2";
	
	private final String LEGACY_MFS1 = "motif_x1";
	private final String LEGACY_MFE1 = "motif_x2";
	private final String LEGACY_MFS2 = "motif_y1";
	private final String LEGACY_MFE2 = "motif_y2";
	
	// true = +, false = -, null = NA
	private boolean strand1, strand2;
	// true - unique, false = inferred, null = NA
	private boolean unique1, unique2;
	private String sequence1, sequence2;
	private long motifStart1, motifEnd1, motifStart2, motifEnd2;
	private double score1, score2;
	
	public Feature2DWithMotif(FeatureType featureType, String chr1, long start1, long end1,
							  String chr2, long start2, long end2, Color c, Map<String, String> attributes) {
		super(featureType, chr1, start1, end1, chr2, start2, end2, c, attributes);
		importAttributesForMotifInformation();
	}
	
	
	public void updateMotifData(boolean strand, boolean unique, String sequence, long motifStart, long motifEnd,
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
        String sequence1 = getAttribute(MFSEQ1);
        if (sequence1 != null && !sequence1.equalsIgnoreCase("null") && !sequence1.equalsIgnoreCase("na")) {
            boolean strand1 = getAttribute(MFO1).contains("p") || getAttribute(MFO1).contains("+");
            boolean unique1 = getAttribute(MFU1).contains("u");

            int motifStart1 = -1, motifEnd1 = -1;
            try {
                motifStart1 = Integer.parseInt(getAttribute(MFS1));
                motifEnd1 = Integer.parseInt(getAttribute(MFE1));

            } catch (Exception e) {
                try {
                    motifStart1 = Integer.parseInt(getAttribute(LEGACY_MFS1));
                    motifEnd1 = Integer.parseInt(getAttribute(LEGACY_MFE1));
                } catch (Exception ee) {
                    ee.printStackTrace();
                }
            }

            /*
            attributes.remove(MFO1);
            attributes.remove(MFU1);
            attributes.remove(MFS1);
            attributes.remove(MFE1);
            attributes.remove(MFSEQ1);
            */

            // done last so that all info must be complete first
            // incomplete motifs will exit via catch before this point
            this.strand1 = strand1;
            this.unique1 = unique1;
            this.sequence1 = sequence1;
            this.motifStart1 = motifStart1;
            this.motifEnd1 = motifEnd1;
        }

        String sequence2 = getAttribute(MFSEQ2);
        if (sequence2 != null && !sequence2.equalsIgnoreCase("null") && !sequence2.equalsIgnoreCase("na")) {
            boolean strand2 = getAttribute(MFO2).contains("p") || getAttribute(MFO2).contains("+");
            boolean unique2 = getAttribute(MFU2).contains("u");
            int motifStart2 = -1, motifEnd2 = -1;
            try {
                motifStart2 = Integer.parseInt(getAttribute(MFS2));
                motifEnd2 = Integer.parseInt(getAttribute(MFE2));
            } catch (Exception e) {
                try {
                    motifStart2 = Integer.parseInt(getAttribute(LEGACY_MFS2));
                    motifEnd2 = Integer.parseInt(getAttribute(LEGACY_MFE2));
                } catch (Exception ee) {
                    ee.printStackTrace();
                }
            }
            /*
            attributes.remove(MFO2);
            attributes.remove(MFU2);
            attributes.remove(MFSEQ2);
            attributes.remove(MFS2);
            attributes.remove(MFE2);
            */

            // done last so that all info must be complete first
            // incomplete motifs will exit via catch before this point
            this.strand2 = strand2;
            this.unique2 = unique2;
            this.sequence2 = sequence2;
            this.motifStart2 = motifStart2;
            this.motifEnd2 = motifEnd2;
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
            output = simpleStringWithColor();
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

    @Override
    public List<MotifAnchor> getAnchors(boolean onlyUninitializedFeatures, ChromosomeHandler handler) {
        List<Feature2D> originalFeatures = new ArrayList<>();
        originalFeatures.add(this);

        List<MotifAnchor> anchors = new ArrayList<>();
        if (isOnDiagonal()) {
            // loops should not be on diagonal
            // anchors.add(new MotifAnchor(chr1, start1, end1, originalFeatures, originalFeatures));
        } else {
            List<Feature2D> emptyList = new ArrayList<>();

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
            } catch (Exception ignored) {
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
		return Objects.hash(super.hashCode(), sequence1, sequence2);
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
            } else {
                return 4;
            }
        } else {
            return 5;
        }
    }

    @Override
    public Feature2D deepCopy() {
        Map<String, String> attrClone = new HashMap<>();
        for (String key : attributes.keySet()) {
            attrClone.put(key, attributes.get(key));
        }
        Feature2DWithMotif clone = new Feature2DWithMotif(featureType, getChr1(), start1, end1, getChr2(), start2, end2, getColor(), attrClone);
        clone.strand1 = strand1;
        clone.strand2 = strand2;
        clone.unique1 = unique1;
        clone.unique2 = unique2;
        clone.sequence1 = sequence1;
        clone.sequence2 = sequence2;
        clone.motifStart1 = motifStart1;
        clone.motifEnd1 = motifEnd1;
        clone.motifStart2 = motifStart2;
        clone.motifEnd2 = motifEnd2;
        clone.score1 = score1;
        clone.score2 = score2;
        return clone;
    }
}
