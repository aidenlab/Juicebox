/*
 * Copyright (C) 2011-2014 Aiden Lab - All Rights Reserved
 * SOFTWARE COPYRIGHT NOTICE
 * This software and its documentation are the copyright of the Aiden Lab All rights are reserved.
 *
 * This software is supplied without any warranty or guaranteed support whatsoever. Aiden Lab is not responsible for its use, misuse, or functionality.
 *
 * This software is licensed under the terms of the GNU Lesser General Public License (LGPL),
 * Version 2.1 which is available at http://www.opensource.org/licenses/lgpl-2.1.php.
 */

package juicebox.track;

import java.awt.*;
import java.text.NumberFormat;
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

    private final NumberFormat formatter = NumberFormat.getInstance();

    private final String chr1;
    private final int start1;
    private final int end1;
    private final String chr2;
    private final int start2;
    private final int end2;
    private final Color color;
    private final Map<String, String> attributes;
    private final String featureName;


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

    public Color getColor() {
        return color;
    }

    public String tooltipText() {

        StringBuilder txt = new StringBuilder();

        txt.append("<font color='red'>");
        txt.append(featureName);
        txt.append(":</font><br>");

        txt.append(chr1).append(":").append(formatter.format(start1 + 1));
        if ((end1 - start1) > 1) {
            txt.append("-").append(formatter.format(end1));
        }
        txt.append("<br>");

        txt.append(chr2).append(":").append(formatter.format(start2 + 1));
        if ((end2 - start2) > 1) {
            txt.append("-").append(formatter.format(end2));
        }


        for (Map.Entry<String, String> entry : attributes.entrySet()) {
            txt.append("<br>");
            txt.append(entry.getKey());
            txt.append(" = ");
            txt.append(entry.getValue());
        }

        return txt.toString();
    }

}
