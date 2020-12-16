/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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

package juicebox.track;

import htsjdk.tribble.Feature;
import juicebox.Context;
import juicebox.HiC;
import juicebox.assembly.IGVFeatureCopy;
import juicebox.assembly.OneDimAssemblyTrackLifter;
import juicebox.gui.SuperAdapter;
import org.broad.igv.feature.Exon;
import org.broad.igv.feature.FeatureUtils;
import org.broad.igv.feature.IGVFeature;
import org.broad.igv.feature.Strand;
import org.broad.igv.track.FeatureSource;
import org.broad.igv.ui.FontManager;
import org.broad.igv.util.BrowserLauncher;
import org.broad.igv.util.ResourceLocator;

import java.awt.*;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

/**
 * @author jrobinso
 *         Date: 12/1/12
 *         Time: 12:34 AM
 */
public class HiCFeatureTrack extends HiCTrack {

    private static final int BLOCK_HEIGHT = 14;
    private static final int ARROW_SPACING = 10;
    private final Font font;
    private final HiC hic;
    private final FeatureSource<?> featureSource;
    private String name;


    public HiCFeatureTrack(HiC hic, ResourceLocator locator, FeatureSource<?> featureSource) {
        super(locator);
        this.hic = hic;
        this.featureSource = featureSource;
        font = FontManager.getFont(6);
    }

    public static double getFractionalBin(int position, double scaleFactor, HiCGridAxis gridAxis) {
        double bin1 = gridAxis.getBinNumberForGenomicPosition(position);
        // Fractional bin (important for "super-zoom")
        if (scaleFactor > 1) {
            double bw1 = gridAxis.getGenomicEnd(bin1) - gridAxis.getGenomicStart(bin1);
            bin1 += (position - gridAxis.getGenomicStart(bin1)) / bw1;
        }
        return bin1;
    }

    @Override
    public void render(Graphics g, Context context, Rectangle rect, TrackPanel.Orientation orientation, HiCGridAxis gridAxis) {
        int height = orientation == TrackPanel.Orientation.X ? rect.height : rect.width;
        int width = orientation == TrackPanel.Orientation.X ? rect.width : rect.height;
        int y = orientation == TrackPanel.Orientation.X ? rect.y : rect.x;
        int x = orientation == TrackPanel.Orientation.X ? rect.x : rect.y;
    
        String chr = context.getChromosome().getName();
        double startBin = context.getBinOrigin();
        final double scaleFactor = hic.getScaleFactor();
        double endBin = startBin + (width / scaleFactor);
    
        // todo igv
        int gStart = (int) gridAxis.getGenomicStart(startBin);
        int gEnd = (int) gridAxis.getGenomicEnd(endBin);
    
        int fh = Math.min(height - 2, BLOCK_HEIGHT);
        int fy = y + (height - fh) / 2;
        int fCenter = y + height / 2;
    
        g.setFont(font);
        g.setColor(getPosColor());
    
        //Graphics strGraphics = g.create();
        g.setColor(new Color(0, 150, 0));

        Iterator<?> iter;

        if (SuperAdapter.assemblyModeCurrentlyActive) {
            // Update features according to current assembly status
            gStart = 0;
            gEnd = (int) context.getChrLength();
        }

        try {
            iter = featureSource.getFeatures(chr, gStart, gEnd);
            if (!iter.hasNext()) {
                // if empty probably because "chr" missing at start of chromosome
                // TODO mitochondrial genes may be an issue here?
                iter = featureSource.getFeatures("chr" + chr, gStart, gEnd);
            }
        } catch (IOException error) {
            System.err.println("Error getting feature source " + error);
            return;
        }

        //handles bed and gff files only for now
        if (SuperAdapter.assemblyModeCurrentlyActive && (getLocator().getPath().toLowerCase().endsWith(".bed") || getLocator().getPath().toLowerCase().endsWith(".gff"))) {
            // update features according to assembly status
            ArrayList<IGVFeature> iterItems = new ArrayList<>();

            while (iter.hasNext()) {
                IGVFeature feature = (IGVFeature) iter.next();
                iterItems.add(feature);
            }

            List<IGVFeatureCopy> newFeatureList = OneDimAssemblyTrackLifter.liftIGVFeatures(hic, context.getChromosome(), (int) startBin, (int) endBin + 1, gridAxis, iterItems, getLocator().getPath().toLowerCase().endsWith(".bed"));
            iter = newFeatureList.iterator();
        }

        while (iter.hasNext()) {
            IGVFeature feature = (IGVFeature) iter.next();

            final Color featureColor = feature.getColor();
            if (featureColor != null) {
                g.setColor(featureColor);
            }

            int startPoint = feature.getStart();
            int endPoint = feature.getEnd();

            double bin1 = getFractionalBin(startPoint, scaleFactor, gridAxis);
            double bin2 = getFractionalBin(endPoint, scaleFactor, gridAxis);

            if (bin2 < startBin) {
                continue;
            } else if (bin1 > endBin) {
                break;
            }

            int xPixelLeft = x + (int) ((bin1 - startBin) * scaleFactor);
            int xPixelRight = x + (int) ((bin2 - startBin) * scaleFactor);

            int fw = Math.max(1, xPixelRight - xPixelLeft);

            if (fw < 5 || feature.getExons() == null || feature.getExons().size() == 0) {
                g.fillRect(xPixelLeft, fy, fw, fh);

            } else {

                // intron
                g.drawLine(xPixelLeft, fCenter, xPixelRight, fCenter);

                // arrows
                if (fw > 20) {
                    if (feature.getStrand() == Strand.POSITIVE) {
                        for (int p = xPixelLeft + 5; p < xPixelLeft + fw; p += 10) {
                            g.drawLine(p - 2, fCenter - 2, p, fCenter);
                            g.drawLine(p - 2, fCenter + 2, p, fCenter);
                        }
                    } else if (feature.getStrand() == Strand.NEGATIVE) {
                        for (int p = xPixelLeft + fw - 5; p > xPixelLeft; p -= 10) {
                            g.drawLine(p + 2, fCenter - 2, p, fCenter);
                            g.drawLine(p + 2, fCenter + 2, p, fCenter);
                        }
                    }
                }

                for (Exon exon : feature.getExons()) {
                    bin1 = getFractionalBin(exon.getStart(), scaleFactor, gridAxis);
                    bin2 = getFractionalBin(exon.getEnd(), scaleFactor, gridAxis);

                    xPixelLeft = (int) ((bin1 - startBin) * scaleFactor);
                    fw = (int) ((bin2 - bin1 + 1) * scaleFactor);
                    g.fillRect(xPixelLeft, fy, fw, fh);
                }
            }
        }
    }
    
    
    @Override
    public String getToolTipText(int x, int y, TrackPanel.Orientation orientation) {

        Context context = orientation == TrackPanel.Orientation.X ? hic.getXContext() : hic.getYContext();
        StringBuilder txt = new StringBuilder();


        txt.append("<span style='color:red; font-family: arial; font-size: 12pt;'>");
        txt.append(getName());
        txt.append("</span>");

        IGVFeature f = getFeatureAtPixel(x, context, orientation);
        if (f != null) { // && (f.getEnd() > start && f.getStart() < end)) {
            txt.append("<span style='font-family: arial; font-size: 12pt;'><br>");
            txt.append(f.getDescription());
            txt.append("</span>");
        }
        return txt.toString();
    }

    private IGVFeature getFeatureAtPixel(int x, Context context, TrackPanel.Orientation orientation) {
    
        HiCGridAxis gridAxis;
        try {
            gridAxis = orientation == TrackPanel.Orientation.X ? hic.getZd().getXGridAxis() : hic.getZd().getYGridAxis();
        } catch (Exception e) {
            return null;
        }
    
        int binOrigin = (int) (context.getBinOrigin());
        int bin = binOrigin + (int) (x / hic.getScaleFactor());
    
        int start = (int) gridAxis.getGenomicStart(bin);
        int end = (int) gridAxis.getGenomicEnd(bin);
        int middle = (int) gridAxis.getGenomicMid(bin);
    
        String chr = context.getChromosome().getName();
    
        int b1 = Math.max(0, bin - 2);
        int b2 = bin + 2;
        int buffer = (int) ((gridAxis.getGenomicEnd(b2) - gridAxis.getGenomicStart(b1)) / 2);
    
        // The maximum length of all features in this collection. Used to insure we consider all features that
        // might overlap the position (feature are sorted by start position, but length is variable)
        int maxFeatureLength = 284000;  // TTN gene
        Iterator<?> iter;
        try {
            iter = featureSource.getFeatures(chr, start, end);
            if (!iter.hasNext()) {
                // if empty, probably because "chr" missing at start of chromosome
                // TODO mitochondrial genes may be an issue here?
                iter = featureSource.getFeatures("chr" + chr, start, end);
            }
        } catch (IOException error) {
            System.err.println("Error getting feature source " + error);
            return null;
        }
        List<Feature> allFeatures = new ArrayList<>();
        while (iter.hasNext()) {
            allFeatures.add((Feature) iter.next());
        }

        List<Feature> featuresAtMouse = FeatureUtils.getAllFeaturesAt(middle, maxFeatureLength, buffer, allFeatures);
        // Return the most specific (smallest);
        if (featuresAtMouse != null && featuresAtMouse.size() > 0) {
            featuresAtMouse.sort(new Comparator<Feature>() {
                @Override
                public int compare(Feature feature, Feature feature1) {
                    return ((feature.getEnd() - feature.getStart()) - (feature1.getEnd() - feature1.getStart()));
                }
            });
            return (IGVFeature) featuresAtMouse.get(0);
        }

        return null;
    }

    @Override
    public String getName() {
        return name;  //To change body of implemented methods use File | Settings | File Templates.
    }

    public void setName(String name) {
        this.name = name;
    }

    protected void drawStrandArrows(Strand strand, int startX, int endX, int startY, Graphics2D g2D) {

        // Don't draw strand arrows for very small regions

        int distance = endX - startX;
        if ((distance < 6)) {
            return;
        }


        int sz = strand.equals(Strand.POSITIVE) ? -3 : 3;

        final int asz = Math.abs(sz);

        for (int ii = startX + ARROW_SPACING / 2; ii < endX; ii += ARROW_SPACING) {

            g2D.drawLine(ii, startY, ii + sz, startY + asz);
            g2D.drawLine(ii, startY, ii + sz, startY - asz);
        }
    }

    public void mouseClicked(int x, int y, Context context, TrackPanel.Orientation orientation) {
        IGVFeature f = getFeatureAtPixel(x, context, orientation);
        String url = "";
        if (f != null) {
            try {
                url = "http://www.genecards.org/cgi-bin/carddisp.pl?gene=" + f.getName();
                BrowserLauncher.openURL(url);
            } catch (IOException e) {
                System.err.println("Error opening gene link: " + url + " " + e.getLocalizedMessage());
            }
        }
    }
}
