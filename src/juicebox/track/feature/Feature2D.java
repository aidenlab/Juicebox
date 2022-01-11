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

import juicebox.HiCGlobals;
import juicebox.assembly.AssemblyHeatmapHandler;
import juicebox.assembly.Scaffold;
import juicebox.data.ChromosomeHandler;
import juicebox.data.anchor.MotifAnchor;
import juicebox.tools.utils.juicer.arrowhead.ArrowheadScore;
import juicebox.tools.utils.juicer.hiccups.HiCCUPSUtils;

import java.awt.*;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.List;
import java.util.*;


/**
 * @author jrobinso, mshamim, mhoeger
 *         <p/>
 *         reflection only used for plotting, should not be used by CLTs
 */
public class Feature2D implements Comparable<Feature2D> {

    static final String genericHeader = "#chr1\tx1\tx2\tchr2\ty1\ty2\tname\tscore\tstrand1\tstrand2\tcolor";
	private static final String genericLegacyHeader = "#chr1\tx1\tx2\tchr2\ty1\ty2\tcolor";
	private static final String BEDPE_SPACER = "\t.\t.\t.\t.";
	private static final String[] categories = new String[]{"observed", "coordinate", "enriched", "expected", "fdr"};
	public static int tolerance = 0;
	public static boolean allowHiCCUPSOrdering = false;
	final FeatureType featureType;
	final Map<String, String> attributes;
	final String chr1;
	final String chr2;
	private final NumberFormat formatter = NumberFormat.getInstance();
	final long start1;
	final long start2;
	long end1;
	long end2;
	private boolean isSelected = false;
	private Feature2D reflection = null;
	private Color color, translucentColor;
	
	public Feature2D(FeatureType featureType, String chr1, long start1, long end1, String chr2, long start2, long end2, Color c,
					 Map<String, String> attributes) {
		this.featureType = featureType;
		this.chr1 = chr1;
		this.start1 = start1;
		this.end1 = end1;
		this.chr2 = chr2;
		this.start2 = start2;
		this.end2 = end2;
		this.color = (c == null ? Color.black : c);
		setTranslucentColor();
        this.attributes = attributes;
    }

    public static String getDefaultOutputFileHeader() {
        if (HiCGlobals.isLegacyOutputPrintingEnabled) {
            return genericLegacyHeader;
        } else {
            return genericHeader;
        }
    }

    public FeatureType getFeatureType() {
        return this.featureType;
    }

    private String getFeatureName() {
        if (featureType == FeatureType.PEAK) {
            return "Peak";
        }
        if (featureType == FeatureType.DOMAIN) {
            return "Contact Domain";
        }
        return "Feature";
    }

    public String getChr1() {
        return chr1;
    }

    public String getChr2() {
        return chr2;
    }
	
	public long getStart1() {
		return start1;
	}
	
	public long getStart2() {
		return start2;
	}
	
	public long getEnd1() {
		return end1;
	}

    public void setEnd1(int end1) {
        this.end1 = end1;
        if (reflection != null)
            reflection.end2 = end1;
    }
	
	public long getEnd2() {
		return end2;
	}

    public void setEnd2(int end2) {
        this.end2 = end2;
        if (reflection != null)
            reflection.end1 = end2;
    }
	
	public long getWidth1() {
		return end1 - start1;
	}
	
	public long getWidth2() {
		return end2 - start2;
	}
	
	public long getMidPt1() {
		return midPoint(start1, end1);
	}
	
	public long getMidPt2() {
		return midPoint(start2, end2);
	}
	
	private long midPoint(long start, long end) {
		return (long) (start + (end - start) / 2.0);
	}
	
	public Color getColor() {
		if (isSelected) {
            return Color.DARK_GRAY;
        } else {
			return color;
		}
	}


    public void setColor(Color color) {
        this.color = color;
        if (reflection != null)
            reflection.color = color;
        setTranslucentColor();
    }

    public Color getTranslucentColor() {
        if (isSelected) {
            return Color.DARK_GRAY;
        } else {
            return translucentColor;
        }
    }

    private void setTranslucentColor() {
        translucentColor = new Color(color.getRed(), color.getGreen(), color.getBlue(), 50);
        if (reflection != null)
            reflection.translucentColor = translucentColor;
    }

    public String tooltipText() {

        String scaledStart1 = formatter.format(start1 * HiCGlobals.hicMapScale + 1);
        String scaledStart2 = formatter.format(start2 * HiCGlobals.hicMapScale + 1);
        String scaledEnd1 = formatter.format(end1 * HiCGlobals.hicMapScale);
        String scaledEnd2 = formatter.format(end2 * HiCGlobals.hicMapScale);

        if (getFeatureType() == FeatureType.SCAFFOLD) {
            Scaffold scaffold = AssemblyHeatmapHandler.getSuperAdapter().getAssemblyStateTracker().getAssemblyHandler().getScaffoldFromFeature(this);
            scaledStart1 = formatter.format(scaffold.getCurrentStart() + 1);
            scaledStart2 = formatter.format(scaffold.getCurrentStart() + 1);
            scaledEnd1 = formatter.format(scaffold.getCurrentEnd());
            scaledEnd2 = formatter.format(scaffold.getCurrentEnd());
        }

        StringBuilder txt = new StringBuilder();
        txt.append("<span style='color:red; font-family: arial; font-size: 12pt;'>");
        txt.append(getFeatureName());
        txt.append("</span><br>");

        txt.append("<span style='font-family: arial; font-size: 12pt;color:" + HiCGlobals.topChromosomeColor + ";'>");
        txt.append(chr1).append(":").append(scaledStart1);
        if ((end1 - start1) > 1) {
            txt.append("-").append(scaledEnd1);
        }

        txt.append("</span><br>");

        txt.append("<span style='font-family: arial; font-size: 12pt;color:" + HiCGlobals.leftChromosomeColor + ";'>");
        txt.append(chr2).append(":").append(scaledStart2);
        if ((end2 - start2) > 1) {
            txt.append("-").append(scaledEnd2);
        }
        txt.append("</span>");
        DecimalFormat df = new DecimalFormat("#.##");

        if (HiCGlobals.allowSpacingBetweenFeatureText) {
            // organize attributes into categories. +1 is for the leftover category if no keywords present
            ArrayList<ArrayList<Map.Entry<String, String>>> sortedFeatureAttributes = new ArrayList<>();
            for (int i = 0; i < categories.length + 1; i++) {
                sortedFeatureAttributes.add(new ArrayList<>());
            }

            // sorting the entries, also filtering out f1-f5 flags
            for (Map.Entry<String, String> entry : attributes.entrySet()) {
                String tmpKey = entry.getKey();
                boolean categoryHasBeenAssigned = false;
                for (int i = 0; i < categories.length; i++) {
                    if (tmpKey.contains(categories[i])) {
                        sortedFeatureAttributes.get(i).add(entry);
                        categoryHasBeenAssigned = true;
                        break;
                    }
                }
                if (!categoryHasBeenAssigned) {
                    sortedFeatureAttributes.get(categories.length).add(entry);
                }
            }

            // append to tooltip text, but now each category is spaced apart
            for (ArrayList<Map.Entry<String, String>> attributeCategory : sortedFeatureAttributes) {
                if (attributeCategory.isEmpty())
                    continue;
                //sort attributes before printing
                Comparator<Map.Entry<String, String>> cmp = new Comparator<Map.Entry<String, String>>() {
                    @Override
                    public int compare(Map.Entry<String, String> o1, Map.Entry<String, String> o2) {
                        return o1.getKey().compareToIgnoreCase(o2.getKey());
                    }
                };
				attributeCategory.sort(cmp);
                for (Map.Entry<String, String> entry : attributeCategory) {
                    String tmpKey = entry.getKey();
                    txt.append("<br>");
                    txt.append("<span style='font-family: arial; font-size: 12pt;'>");
                    txt.append(tmpKey);
                    txt.append(" = <b>");
                    try {
                        txt.append(df.format(Double.valueOf(entry.getValue())));
                    } catch (Exception e) {
                        txt.append(entry.getValue()); // for text i.e. non-decimals
                    }
                    txt.append("</b>");
                    txt.append("</span>");
                }
                txt.append("<br>"); // the extra spacing between categories
            }
        } else {
            // simple text dump for plotting, no spacing or rearranging by category
            for (Map.Entry<String, String> entry : attributes.entrySet()) {
                String tmpKey = entry.getKey();
                if (!(tmpKey.equals("f1") || tmpKey.equals("f2") || tmpKey.equals("f3") || tmpKey.equals("f4") || tmpKey.equals("f5"))) {
                    txt.append("<br>");
                    txt.append("<span style='font-family: arial; font-size: 12pt;'>");
                    txt.append(tmpKey);
                    txt.append(" = <b>");
                    //System.out.println(entry.getValue());
                    try {
                        txt.append(df.format(Double.valueOf(entry.getValue())));
                    } catch (Exception e) {
                        txt.append(entry.getValue());
                    }
                    txt.append("</b>");
                    txt.append("</span>");
                }
            }
        }
        return txt.toString();
    }

    public String getOutputFileHeader() {
        StringBuilder output = new StringBuilder(getDefaultOutputFileHeader());

        ArrayList<String> keys = new ArrayList<>(attributes.keySet());
        Collections.sort(keys);

        for (String key : keys) {
            output.append("\t").append(key);
        }

        return output.toString();
    }

    private String simpleString() {
        return chr1 + "\t" + start1 + "\t" + end1 + "\t" + chr2 + "\t" + start2 + "\t" + end2;
    }

    private String justColorString() {
        return "\t" + color.getRed() + "," + color.getGreen() + "," + color.getBlue();
    }

    public String simpleStringWithColor() {
        if (HiCGlobals.isLegacyOutputPrintingEnabled) {
            return simpleString() + justColorString();
        } else if (this.containsAttributeKey("score")) {
            return simpleString() + "\t.\t" + this.attributes.get("score") + "\t.\t." + justColorString();
        } else {
            return simpleString() + BEDPE_SPACER + justColorString();
        }
    }

    @Override
    public String toString() {
        StringBuilder output = new StringBuilder(simpleStringWithColor());

        ArrayList<String> keys = new ArrayList<>(attributes.keySet());
        Collections.sort(keys);
        for (String key : keys) {
            output.append("\t").append(attributes.get(key));
        }

        return output.toString();
    }

    public ArrayList<String> getAttributeKeys() {
        ArrayList<String> keys = new ArrayList<>(attributes.keySet());
        Collections.sort(keys);
        return keys;
    }

    public String getAttribute(String key) {
        return attributes.get(key);
    }

    public void setAttribute(String key, String newVal) {
        attributes.put(key, newVal);
        // attribute directly shared between reflections
        if (reflection != null)
            reflection.attributes.put(key, newVal);
    }

    public float getFloatAttribute(String key) {
        return Float.parseFloat(attributes.get(key));
    }

    public void addIntAttribute(String key, int value) {
        attributes.put(key, "" + value);
    }

    public void addFloatAttribute(String key, Float value) {
        attributes.put(key, "" + value);
    }

    public void addStringAttribute(String key, String value) {
        attributes.put(key, value);
    }

    /**
     * @param otherFeature
     * @return
     */
    public boolean overlapsWith(Feature2D otherFeature) {
	
		float window1 = (otherFeature.getEnd1() - otherFeature.getStart1()) / 2;
		float window2 = (otherFeature.getEnd2() - otherFeature.getStart2()) / 2;
	
		long midOther1 = otherFeature.getMidPt1();
		long midOther2 = otherFeature.getMidPt2();
	
		return midOther1 >= (this.start1 - window1) && midOther1 <= (this.end1 + window1) && midOther2 >= (this.start2 - window2) && midOther2 <= (this.end2 + window2);
	}

    @Override
    public int compareTo(Feature2D o) {
        // highest observed point ordering needed for hiccups sorting
        if (allowHiCCUPSOrdering && attributes.containsKey(HiCCUPSUtils.OBSERVED)
                && o.attributes.containsKey(HiCCUPSUtils.OBSERVED)) {
            float val = Float.parseFloat(getAttribute(HiCCUPSUtils.OBSERVED)) - Float.parseFloat(o.getAttribute(HiCCUPSUtils.OBSERVED));
            if (val > 0) return 1;
            if (val < 0) return -1;
        }
		long[] comparisons = new long[]{chr1.compareTo(o.chr1), chr2.compareTo(o.chr2), start1 - o.start1,
				start2 - o.start2, end1 - o.end1, end2 - o.end2};
		for (long i : comparisons) {
			if (i != 0)
				if (i > 0) {
					return 1;
				} else {
					return -1;
				}
		}
        return 0;
    }

    public boolean isOnDiagonal() {
        return chr1.equals(chr2) && start1 == start2 && end1 == end2;
    }

    public Feature2D reflectionAcrossDiagonal() {
        if (reflection == null) {
            reflection = new Feature2D(featureType, chr2, start2, end2, chr1, start1, end1, color, attributes);
            reflection.reflection = this;
        }
        return reflection;
    }

    public boolean isInLowerLeft() {
        return chr1.equals(chr2) && start2 > start1;
    }

    public boolean isInUpperRight() {
        return chr1.equals(chr2) && start2 < start1;
    }

    public boolean doesNotContainAttributeKey(String attribute) {
        return !attributes.containsKey(attribute);
    }

    public boolean containsAttributeValue(String attribute) {
		return attributes.containsValue(attribute);
    }

    public boolean containsAttributeKey(String attribute) {
        return attributes.containsKey(attribute);
    }

    public String getLocationKey() {
        return start1 + "_" + start2;
    }

    public ArrowheadScore toArrowheadScore() {
		long[] indices = new long[]{start1, end1, start2, end2};
        return new ArrowheadScore(indices);
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        if (this == obj) {
            return true;
        }

        final Feature2D other = (Feature2D) obj;
        if (chr1.equals(other.chr1)) {
            if (chr2.equals(other.chr2)) {
                if (Math.abs(start1 - other.start1) <= tolerance) {
                    if (Math.abs(start2 - other.start2) <= tolerance) {
                        if (Math.abs(end1 - other.end1) <= tolerance) {
                            return Math.abs(end2 - other.end2) <= tolerance;
                        }
                    }
                }
            }
        }

        return false;
    }

    @Override
    public int hashCode() {
		return Objects.hash(chr1, end1, start1, chr2, end2, start2);
    }

    public void clearAttributes() {
        attributes.clear();
    }

    public List<MotifAnchor> getAnchors(boolean onlyUninitializedFeatures, ChromosomeHandler handler) {
        List<Feature2D> originalFeatures = new ArrayList<>();
        originalFeatures.add(this);

        List<MotifAnchor> anchors = new ArrayList<>();
        if (isOnDiagonal()) {
            // loops should not be on diagonal
            // anchors.add(new MotifAnchor(chr1, start1, end1, originalFeatures, originalFeatures));
        } else {
            List<Feature2D> emptyList = new ArrayList<>();
            anchors.add(new MotifAnchor(chr1, start1, end1, originalFeatures, emptyList));
            anchors.add(new MotifAnchor(chr2, start2, end2, emptyList, originalFeatures));
        }
        return anchors;
    }

    public Feature2D deepCopy() {
        Map<String, String> attrClone = new HashMap<>();
        for (String key : attributes.keySet()) {
            attrClone.put(key, attributes.get(key));
        }
        return new Feature2D(featureType, chr1, start1, end1, chr2, start2, end2, color, attrClone);
	}
	
	public void setSetIsSelectedColorUpdate(boolean setIsSelectedColorUpdate) {
        isSelected = setIsSelectedColorUpdate;
    }

    public Feature2DWithMotif toFeature2DWithMotif() {
        return new Feature2DWithMotif(featureType, chr1, start1, end1, chr2, start2, end2, color, attributes);
    }

    public boolean containsPoint(float x, float y) {
        return start1 <= x && x <= end1 && start2 <= y && y <= end2;
    }

    public boolean containsPoint(Point point) {
        return containsPoint(point.x, point.y);
    }

    public enum FeatureType {
        NONE, PEAK, DOMAIN, GENERIC, SCAFFOLD, SUPERSCAFFOLD, SELECTED_GROUP
    }
}
