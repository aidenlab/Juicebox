/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2019 Broad Institute, Aiden Lab
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

// Created by Santiago Garcia Acosta July 4, 2018.

package juicebox.assembly;

import org.broad.igv.feature.Exon;
import org.broad.igv.feature.IGVFeature;
import org.broad.igv.feature.Strand;
import org.broad.igv.track.WindowFunction;
import org.broad.igv.util.collections.MultiMap;

import java.awt.*;
import java.util.List;

public class IGVFeatureCopy implements IGVFeature {
    private final IGVFeature origFeat;
    private final String type;
    private final String id;
    private final String description;
    private final String url;
    private final String name;
    private final String chr;
    private final String contig;
    private Strand strand;
    private final int length;
    private final MultiMap<String, String> attributes;
    private final float score;
    private List<Exon> exons;
    private Color color;
    private int start;
    private int end;
    public static boolean colorFeaturesChk = false;

    public IGVFeatureCopy(IGVFeature feature) {
        this.origFeat = feature;
        this.type = feature.getType();
        this.id = feature.getIdentifier();
        this.description = feature.getDescription();
        this.url = feature.getURL();
        this.color = feature.getColor();
        this.attributes = feature.getAttributes();
        this.name = feature.getName();
        this.chr = feature.getChr();
        this.score = feature.getScore();
        this.contig = feature.getContig();
        this.start = feature.getStart();
        this.end = feature.getEnd();
        this.length = feature.getLength();
        this.strand = feature.getStrand();
        this.exons = feature.getExons();
    }

    public void updateExons(List<Exon> newExons) {
      exons = newExons;
    }

    public void updateStrand(Strand curStrand, boolean inversionState, boolean isBed) {
      Strand newStrand;

      if (!inversionState) {
          newStrand = curStrand;
      }
      else {
        if (colorFeaturesChk && isBed && color != null) {
          // Update color to complement color
          int r = color.getRed();
          int g = color.getGreen();
          int b = color.getBlue();

          color = new Color(~r & 0xff, ~g & 0xff, ~b & 0xff);
        }

        // Update strand orientation
        if (curStrand == Strand.POSITIVE) {
            newStrand = Strand.NEGATIVE;
        }
        else if (curStrand == Strand.NEGATIVE) {
            newStrand = Strand.POSITIVE;
        }
        else {
            newStrand = strand;
        }
      }
      this.strand = newStrand;
    }

    public static void invertColorFeaturesChk() {
      colorFeaturesChk = !colorFeaturesChk;
    }

    public void setStart(int newStart) {
        this.start = newStart;
    }

    public void setEnd(int newEnd) {
        this.end = newEnd;
    }

    public int getStart() {
        return start;
    }

    public int getEnd() {
        return end;
    }

    public int getLength() {
        return java.lang.Math.abs(end - start);
    }

    public String getType() {
        return type;
    }

    public String getIdentifier() {
        return id;
    }

    public String getDescription() {
        return description;
    }

    public Strand getStrand() { return strand; }

    public MultiMap<String, String> getAttributes() {
        return attributes;
    }

    public boolean contains(IGVFeature var1) {
        return origFeat.contains(var1);
    }

    public boolean contains(double var1) {
        return origFeat.contains(var1);
    }

    public List<Exon> getExons() {
        return exons;
    }

    public Color getColor() {
        return color;
    }

    public String getURL() {
        return url;
    }

    public String getName() {
        return name;
    }

    public String getChr() {
        return chr;
    }

    public float getScore() {
        return score;
    }

    public String getValueString(double var1, WindowFunction var3) {
        return origFeat.getValueString(var1, var3);
    }

    public String getContig() {
        return contig;
    }
}
