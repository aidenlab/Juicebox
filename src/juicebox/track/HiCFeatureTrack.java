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

import htsjdk.tribble.Feature;
import juicebox.Context;
import juicebox.HiC;
import org.apache.log4j.Logger;
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
import java.util.*;
import java.util.List;

/**
 * @author jrobinso
 *         Date: 12/1/12
 *         Time: 12:34 AM
 */
public class HiCFeatureTrack extends HiCTrack {

    static final int THIN_BLOCK_HEIGHT = 6;
    private static final Logger log = Logger.getLogger(HiCFeatureTrack.class);
    private static final int BLOCK_HEIGHT = 14;
    private static final int ARROW_SPACING = 10;
    private final Font font;
    private final HiC hic;
    private final FeatureSource<?> featureSource;
    private Color color = Color.blue.darker();
    private Color altColor = Color.blue.brighter();
    private String name;


    public HiCFeatureTrack(HiC hic, ResourceLocator locator, FeatureSource<?> featureSource) {
        super(locator);
        this.hic = hic;
        this.featureSource = featureSource;
        font = FontManager.getFont(6);
    }

    @Override
    public void render(Graphics2D g2d, Context context, Rectangle rect, TrackPanel.Orientation orientation, HiCGridAxis gridAxis) {

        int height = orientation == TrackPanel.Orientation.X ? rect.height : rect.width;
        int width = orientation == TrackPanel.Orientation.X ? rect.width : rect.height;
        int y = orientation == TrackPanel.Orientation.X ? rect.y : rect.x;
        int x = orientation == TrackPanel.Orientation.X ? rect.x : rect.y;

        String chr = context.getChromosome().getName();
        double startBin = context.getBinOrigin();
        final double scaleFactor = hic.getScaleFactor();
        double endBin = startBin + (width / scaleFactor);

        int gStart = gridAxis.getGenomicStart(startBin);
        int gEnd = gridAxis.getGenomicEnd(endBin);

        int fh = Math.min(height - 2, BLOCK_HEIGHT);
        int fy = y + (height - fh) / 2;
        int fCenter = y + height / 2;

        g2d.setFont(font);
        g2d.setColor(color);

        Graphics strGraphics = g2d.create();
        strGraphics.setColor(new Color(0, 150, 0));

        if (!chr.startsWith("chr")) chr = "chr" + chr;   // TODO - use alias

        Iterator<?> iter;
        try {
            iter = featureSource.getFeatures(chr, gStart, gEnd);
        } catch (IOException error) {
            log.error("Error getting feature source " + error);
            return;
        }
        while (iter.hasNext()) {

            IGVFeature feature = (IGVFeature) iter.next();
            final Color featureColor = feature.getColor();
            if (featureColor != null) {
                g2d.setColor(featureColor);
            }

            double bin1 = gridAxis.getBinNumberForGenomicPosition(feature.getStart());
            double bin2 = gridAxis.getBinNumberForGenomicPosition(feature.getEnd() - 1);

            if (bin2 < startBin) {
                continue;
            } else if (bin1 > endBin) {
                break;
            }

            // Fractional bin (important for "super-zoom")
            if (scaleFactor > 1) {
                double bw1 = gridAxis.getGenomicEnd(bin1) - gridAxis.getGenomicStart(bin1);
                bin1 += (feature.getStart() - gridAxis.getGenomicStart(bin1)) / bw1;

                double bw2 = gridAxis.getGenomicEnd(bin2) - gridAxis.getGenomicStart(bin2);
                bin2 += (feature.getEnd() - gridAxis.getGenomicStart(bin2)) / bw2;
            }

            int xPixelLeft = x + (int) ((bin1 - startBin) * scaleFactor);
            int xPixelRight = x + (int) ((bin2 - startBin) * scaleFactor);

            int fw = Math.max(1, xPixelRight - xPixelLeft);

            if (fw < 5 || feature.getExons() == null || feature.getExons().size() == 0) {
                g2d.fillRect(xPixelLeft, fy, fw, fh);
            } else {

                // intron
                g2d.drawLine(xPixelLeft, fCenter, xPixelRight, fCenter);

                // arrows
                if (fw > 20) {
                    if (feature.getStrand() == Strand.POSITIVE) {
                        for (int p = xPixelLeft + 5; p < xPixelLeft + fw; p += 10) {
                            g2d.drawLine(p - 2, fCenter - 2, p, fCenter);
                            g2d.drawLine(p - 2, fCenter + 2, p, fCenter);
                        }
                    } else if (feature.getStrand() == Strand.NEGATIVE) {
                        for (int p = xPixelLeft + fw - 5; p > xPixelLeft; p -= 10) {
                            g2d.drawLine(p + 2, fCenter - 2, p, fCenter);
                            g2d.drawLine(p + 2, fCenter + 2, p, fCenter);
                        }
                    }
                }

                // UTRS  -- Don't bother with this.  Bins are too coarse a resolution.
                // "Thick" start for UTRs.  It's UCSC's normenclature, not mine
//                int thickBinStart = bin1;
//                int thickBinEnd = bin2;
//                if (feature instanceof BasicFeature) {
//                    thickBinStart = gridAxis.getBinNumberForGenomicPosition(((BasicFeature) feature).getThickStart());
//                    thickBinEnd = gridAxis.getBinNumberForGenomicPosition(((BasicFeature) feature).getThickEnd());
//                }

                for (Exon exon : feature.getExons()) {

                    bin1 = gridAxis.getBinNumberForGenomicPosition(exon.getStart());
                    bin2 = gridAxis.getBinNumberForGenomicPosition(exon.getEnd() - 1);

                    // UTRS  -- Don't bother with this.  Bins are too coarse a resolution.
//                    if (feature instanceof BasicFeature) {
//                        if (thickBinStart > bin1) {
//                            int utrW = (int) ((Math.min(bin2, thickBinStart) - bin1 + 1) * context.getScaleFactor());
//                            g2d.fillRect(xPixelLeft, fUTRy, utrW, THIN_BLOCK_HEIGHT);
//                            if (thickBinStart >= bin2) continue;  // entire exon is UTR
//                            bin1 = thickBinStart;
//                        }
//
//                        if (thickBinEnd < bin2) {
//                            int tmp = Math.max(bin1, thickBinEnd);
//                            int utrW = (int) ((bin2 - tmp + 1) * context.getScaleFactor());
//                            g2d.fillRect(xPixelLeft, fUTRy, utrW, THIN_BLOCK_HEIGHT);
//                            if (thickBinEnd <= bin1) continue;  // entire exon is UTR
//                            bin2 = thickBinEnd;
//                        }
//                    }

                    xPixelLeft = (int) ((bin1 - startBin) * scaleFactor);
                    fw = (int) ((bin2 - bin1 + 1) * scaleFactor);
                    g2d.fillRect(xPixelLeft, fy, fw, fh);
                }
//
//                // Tag transcription start with green
//                if (feature instanceof BasicFeature) {
//                    int b;
//                    if (feature.getStrand() == Strand.POSITIVE) {
//                        b = gridAxis.getBinNumberForGenomicPosition(((BasicFeature) feature).getThickStart());
//                    } else {
//                        b = gridAxis.getBinNumberForGenomicPosition(((BasicFeature) feature).getThickEnd());
//                    }
//                    int pixel = (int) ((b - startBin) * context.getScaleFactor());
//                    int pw = (int) Math.max(1, context.getScaleFactor());
//                    strGraphics.fillRect(pixel, fy, pw, fh);
//                }


            }

            // Name
//            String name = feature.getName();
//            int sw = (int) fm.getStringBounds(name, g2d).getWidth();
//            int sx = (xPixelLeft + xPixelRight - sw) / 2;
//            if (sx > lastNameX) {
//                lastNameX = sx + sw;
//                g2d.drawString(name, sx, y + height - 5);
//            }

            strGraphics.dispose();
        }
    }

    @Override
    public Color getPosColor() {
        return color;  //To change body of implemented methods use File | Settings | File Templates.
    }

    @Override
    public String getToolTipText(int x, int y, TrackPanel.Orientation orientation) {

        Context context = orientation == TrackPanel.Orientation.X ? hic.getXContext() : hic.getYContext();

        IGVFeature f = getFeatureAtPixel(x, context, orientation);
        if (f != null) { // && (f.getEnd() > start && f.getStart() < end)) {
            return f.getDescription();
        } else {
            return null;
        }
    }

    private IGVFeature getFeatureAtPixel(int x, Context context, TrackPanel.Orientation orientation) {

        HiCGridAxis gridAxis;
        gridAxis = orientation == TrackPanel.Orientation.X ? hic.getZd().getXGridAxis() : hic.getZd().getYGridAxis();

        int binOrigin = (int) (context.getBinOrigin());
        int bin = binOrigin + (int) (x / hic.getScaleFactor());

        int start = gridAxis.getGenomicStart(bin);
        int end = gridAxis.getGenomicEnd(bin);
        int middle = gridAxis.getGenomicMid(bin);

        String chr = context.getChromosome().getName();
        if (!chr.startsWith("chr")) chr = "chr" + chr;   // TODO -- use genome aliax

        int b1 = Math.max(0, bin - 2);
        int b2 = bin + 2;
        int buffer = (gridAxis.getGenomicEnd(b2) - gridAxis.getGenomicStart(b1)) / 2;

        // The maximum length of all features in this collection. Used to insure we consider all features that
        // might overlap the position (feature are sorted by start position, but length is variable)
        int maxFeatureLength = 284000;  // TTN gene
        Iterator<?> iter;
        try {
            iter = featureSource.getFeatures(chr, start, end);
        } catch (IOException error) {
            log.error("Error getting feature source " + error);
            return null;
        }
        List<Feature> allFeatures = new ArrayList<Feature>();
        while (iter.hasNext()) {
            allFeatures.add((Feature) iter.next());
        }

        List<Feature> featuresAtMouse = FeatureUtils.getAllFeaturesAt(middle, maxFeatureLength, buffer, allFeatures);
        // Return the most specific (smallest);
        if (featuresAtMouse != null && featuresAtMouse.size() > 0) {
            Collections.sort(featuresAtMouse, new Comparator<Feature>() {
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
    public void setColor(Color selectedColor) {
        this.color = selectedColor;
    }

    @Override
    public void setAltColor(Color selectedColor) {
        this.altColor = selectedColor;
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
                log.error("Error opening gene link: " + url, e);
            }
        }
    }
}
