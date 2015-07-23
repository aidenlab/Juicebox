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

import java.awt.*;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Map;

/**
 * chr1	x1	x2	chr2	y1	y2	color	observed	bl expected	donut expected	bl fdr	donut fdr
 *
 * @author jrobinso
 *         Date: 5/22/13
 *         Time: 8:51 AM
 *         <p/>
 *         Chr Chr pos pos observed expected1 expected2 fdr
 */
public class Feature2D {

    public static final String peak = "Peak";
    public static final String domain = "Contact domain";
    public static final String generic = "feature";
    private static final String genericHeader = "chr1\tx1\tx2\tchr2\ty1\ty2\tcolor";
    private static final String[] categories = new String[]{"observed", "coordinate", "enriched", "expected", "fdr"};
    private final NumberFormat formatter = NumberFormat.getInstance();
    private final String chr1;
    private final int start1;
    private final int end1;
    private final String chr2;
    private final int start2;
    private final int end2;
    private final String featureName;
    private final Map<String, String> attributes;
    private Color color;

    public Feature2D(String featureName, String chr1, int start1, int end1, String chr2, int start2, int end2, Color c,
                     Map<String, String> attributes) {
        this.featureName = featureName;
        this.chr1 = chr1;
        this.start1 = start1;
        this.end1 = end1;
        this.chr2 = chr2;
        this.start2 = start2;
        this.end2 = end2;
        this.color = (c == null ? Color.black : c);
        this.attributes = attributes;
    }

    public static String getDefaultOutputFileHeader() {
        return genericHeader;
    }

    public String getChr1() {
        return chr1;
    }

    public String getChr2() {
        return chr2;
    }

    public int getStart1() {
        return start1;
    }

    public int getStart2() {
        return start2;
    }

    public int getEnd1() {
        return end1;
    }

    public int getEnd2() {
        return end2;
    }

    public int getMidPt1() { return midPoint(start1, end1);}

    public int getMidPt2() { return midPoint(start2, end2);}

    private int midPoint(int start, int end) {
        return (int) (start + (end - start)/2.0);
    }

    public Color getColor() {
        return color;
    }

    public void setColor(Color color) {
        this.color = color;
    }

    public String tooltipText() {

        StringBuilder txt = new StringBuilder();
        txt.append("<span style='color:red; font-family: arial; font-size: 12pt;'>");
        txt.append(featureName);
        txt.append("</span><br>");

        txt.append("<span style='font-family: arial; font-size: 12pt;color:"+ HiCGlobals.topChromosomeColor+";'>");
        txt.append(chr1).append(":").append(formatter.format(start1 + 1));
        if ((end1 - start1) > 1) {
            txt.append("-").append(formatter.format(end1));
        }

        txt.append("</span><br>");

        txt.append("<span style='font-family: arial; font-size: 12pt;color:"+ HiCGlobals.leftChromosomeColor+";'>");
        txt.append(chr2).append(":").append(formatter.format(start2 + 1));
        if ((end2 - start2) > 1) {
            txt.append("-").append(formatter.format(end2));
        }
        txt.append("</span>");
        DecimalFormat df = new DecimalFormat("#.##");

        if(HiCGlobals.allowSpacingBetweenFeatureText) {
            // organize attributes into categories. +1 is for the leftover category if no keywords present
            ArrayList<ArrayList<Map.Entry<String, String>>> sortedFeatureAttributes = new ArrayList<ArrayList<Map.Entry<String, String>>>();
            for (int i = 0; i < categories.length + 1; i++) {
                sortedFeatureAttributes.add(new ArrayList<Map.Entry<String, String>>());
            }

            // sorting the entries, also filtering out f1-f5 flags
            for (Map.Entry<String, String> entry : attributes.entrySet()) {
                String tmpKey = entry.getKey();
                if (!(tmpKey.equals("f1") || tmpKey.equals("f2") || tmpKey.equals("f3") || tmpKey.equals("f4") || tmpKey.equals("f5"))) {
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
            }

            // append to tooltip text, but now each category is spaced apart
            for (ArrayList<Map.Entry<String, String>> attributeCategory : sortedFeatureAttributes) {
                if(attributeCategory.isEmpty())
                    continue;
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
        }
        else {
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
                    }
                    catch (Exception e){
                        txt.append(entry.getValue());
                    }
                    txt.append("</b>");
                    txt.append("</span>");
                }
            }
        }
        return txt.toString();
    }

    public String getOutputFileHeader(){
        String output = genericHeader;

        ArrayList<String> keys = new ArrayList<String>(attributes.keySet());
        Collections.sort(keys);

        for(String key : keys){
            output += "\t"+key;
        }

        return output;
    }

    @Override
    public String toString(){
        String output = chr1+"\t"+start1+"\t"+end1+"\t"+chr2+"\t"+start2+"\t"+end2;
        output += "\t"+color.getRed()+","+color.getGreen()+","+color.getBlue();

        ArrayList<String> keys = new ArrayList<String>(attributes.keySet());
        Collections.sort(keys);

        for(String key : keys){
            output += "\t"+attributes.get(key);
        }

        return output;
    }

    public ArrayList<String> getAttributeKeys(){
        ArrayList<String> keys = new ArrayList<String>(attributes.keySet());
        Collections.sort(keys);
        return keys;
    }

    public String getAttribute(String key) {
        return attributes.get(key);
    }

    public void setAttribute(String key, String newVal) {attributes.put(key, newVal);}

    public float getFloatAttribute(String key) {
        return Float.parseFloat(attributes.get(key));
    }

    public void addFeature(String key, String value) {
        attributes.put(key, value);
    }

    public boolean overlapsWith(Feature2D otherFeature){

        float window1 = (otherFeature.getEnd1() - otherFeature.getStart1()) / 2;
        float window2 = (otherFeature.getEnd2() - otherFeature.getStart2()) / 2;

        int midOther1 = otherFeature.getMidPt1();
        int midOther2 = otherFeature.getMidPt2();

        if (midOther1 >= (this.start1 - window1) && midOther1 <= (this.end1 + window1)){
            if (midOther2 >= (this.start2 - window2) && midOther2 <= (this.end2 + window2)){
                return true;
            }
        }
        return false;
    }

}
