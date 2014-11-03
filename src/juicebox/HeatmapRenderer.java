package juicebox;

import org.apache.commons.math.stat.StatUtils;
import juicebox.data.*;
import juicebox.matrix.BasicMatrix;
import org.broad.igv.renderer.ColorScale;
import org.broad.igv.renderer.ContinuousColorScale;
import org.broad.igv.util.collections.DoubleArrayList;

import java.awt.*;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author jrobinso
 * @date Aug 11, 2010
 */
public class HeatmapRenderer {

    // TODO -- introduce a "model" in lieu of MainWindow pointer
    HiC hic;
    MainWindow mainWindow;

    private ContinuousColorScale observedColorScale;
    private ColorScale oeColorScale;
    private ColorScale pearsonColorScale;

    Map<String, ContinuousColorScale> observedColorScaleMap = new HashMap<String, ContinuousColorScale>();

    public HeatmapRenderer(MainWindow mainWindow, HiC hic) {
        this.mainWindow = mainWindow;
        this.hic = hic;

        oeColorScale = new OEColorScale();
        pearsonColorScale = new HiCColorScale();
    }

    public boolean render(int originX,
                          int originY,
                          int width,
                          int height,
                          final MatrixZoomData zd,
                          MainWindow.MatrixType displayOption,
                          Graphics2D g) {


        g.setRenderingHint(RenderingHints.KEY_COLOR_RENDERING, RenderingHints.VALUE_COLOR_RENDER_SPEED);
        g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_SPEED);

        int chr1 = zd.getChr1Idx();
        int chr2 = zd.getChr2Idx();
        int x = originX;
        int y = originY;

        boolean isWholeGenome = chr1 == 0 && chr2 == 0;
        boolean sameChr = (chr1 == chr2);

        if (sameChr) {
            // Data is transposable, transpose if necessary.  Convention is to use lower diagonal
            if (x > y) {
                x = originY;
                y = originX;
                int tmp = width;
                width = height;
                height = tmp;
            }
        }

        int maxX = x + width - 1;
        int maxY = y + height - 1;

        if (displayOption == MainWindow.MatrixType.PEARSON) {

            BasicMatrix bm = zd.getPearsons(hic.getDataset().getExpectedValues(hic.getZd().getZoom(), hic.getNormalizationType()));

            ((HiCColorScale) pearsonColorScale).setMin(bm.getLowerValue());
            ((HiCColorScale) pearsonColorScale).setMax(bm.getUpperValue());
            renderMatrix(bm, originX, originY, width, height, pearsonColorScale, g);

        } else {
            // Iterate through blocks overlapping visible region

            List<Block> blocks = zd.getNormalizedBlocksOverlapping(x, y, maxX, maxY, hic.getNormalizationType());
            if (blocks == null) {
                return false;
            }

            MatrixZoomData controlZD = hic.getControlZd();
            boolean hasControl = controlZD != null;
            Map<Integer, Block> controlBlocks = new HashMap<Integer, Block>();
            if (hasControl) {
                List<Block> ctrls = controlZD.getNormalizedBlocksOverlapping(x, y, maxX, maxY, hic.getNormalizationType());
                for (Block b : ctrls) {
                    controlBlocks.put(b.getNumber(), b);
                }
            }


            ColorScale cs = getColorScale(zd, displayOption, isWholeGenome, blocks);

            ExpectedValueFunction df = null;
            if (sameChr && (displayOption == MainWindow.MatrixType.OE || displayOption == MainWindow.MatrixType.EXPECTED)) {
                df = hic.getDataset().getExpectedValues(zd.getZoom(), hic.getNormalizationType());
            }

            double averageCount = zd.getAverageCount(); // Will get overwritten for intra-chr
            double ctrlAverageCount = controlZD == null ? 1 : controlZD.getAverageCount();
            for (Block b : blocks) {

                Collection<ContactRecord> recs = b.getContactRecords();
                if (recs != null) {

                    Map<String, ContactRecord> controlRecords = new HashMap<String, ContactRecord>();
                    if (hasControl) {
                        Block cb = controlBlocks.get(b.getNumber());
                        if (cb != null) {
                            for (ContactRecord ctrlRec : cb.getContactRecords()) {
                                controlRecords.put(ctrlRec.getKey(), ctrlRec);
                            }
                        }
                    }

                    for (ContactRecord rec : recs) {
                        double score = Double.NaN;
                        double expectedValue = averageCount; // Will get overriden for intra
                        if (displayOption == MainWindow.MatrixType.OE || displayOption == MainWindow.MatrixType.EXPECTED) {
                            double expected = 0;
                            if (chr1 == chr2) {
                                if (df != null) {
                                    int binX = rec.getBinX();
                                    int binY = rec.getBinY();
                                    int dist = Math.abs(binX - binY);
                                    expected = df.getExpectedValue(chr1, dist);
                                }
                            } else {
                                expected = (expectedValue > 0 ? expectedValue : 1);
                            }

                            if (displayOption == MainWindow.MatrixType.OE) {
                                score = rec.getCounts() / expected;
                            } else {
                                score = expected;
                            }
                        } else if (displayOption == MainWindow.MatrixType.CONTROL && hasControl) {
                            ContactRecord ctrlRecord = controlRecords.get(rec.getKey());
                            if (ctrlRecord != null) score = ctrlRecord.getCounts();
                        } else if (displayOption == MainWindow.MatrixType.RATIO && hasControl) {
                            ContactRecord ctrlRecord = controlRecords.get(rec.getKey());
                            if (ctrlRecord != null && ctrlRecord.getCounts() > 0) {
                                double num = rec.getCounts() / averageCount;
                                double den = ctrlRecord.getCounts() / ctrlAverageCount;
                                score = num / den;
                            }
                        } else {
                            score = rec.getCounts();
                        }
                        if (Double.isNaN(score)) continue;

                        Color color = cs.getColor((float) score);
                        g.setColor(color);
                        // TODO -- need to check right bounds before drawing

                        int px = rec.getBinX() - originX;
                        int py = rec.getBinY() - originY;
                        if (px > -1 && py > -1 && px <= width && py <= height) {
                            g.fillRect(px, py, MainWindow.BIN_PIXEL_WIDTH, MainWindow.BIN_PIXEL_WIDTH);
                        }

                        if (sameChr && (rec.getBinX() != rec.getBinY())) {
                            px = (rec.getBinY() - originX);
                            py = (rec.getBinX() - originY);
                            if (px > -1 && py > -1 && px <= width && py <= height) {
                                g.fillRect(px, py, MainWindow.BIN_PIXEL_WIDTH, MainWindow.BIN_PIXEL_WIDTH);
                            }
                        }
                    }
                }
            }
        }
        return true;
    }

    private ColorScale getColorScale(MatrixZoomData zd, MainWindow.MatrixType displayOption, boolean wholeGenome, List<Block> blocks) {
        ColorScale cs;
        if (displayOption == MainWindow.MatrixType.OBSERVED || displayOption == MainWindow.MatrixType.EXPECTED ||
                displayOption == MainWindow.MatrixType.CONTROL) {
            String key = zd.getKey() + displayOption;
            observedColorScale = observedColorScaleMap.get(key);
            if (observedColorScale == null) {
                double percentile = wholeGenome ? 99 : 95;
                float max = computePercentile(blocks, percentile);

                observedColorScale = new ContinuousColorScale(0, max, Color.white, Color.red);
                observedColorScaleMap.put(key, observedColorScale);
                mainWindow.updateColorSlider(0, 2 * max, max);
            }
            cs = observedColorScale;
        } else {
            cs = oeColorScale;
        }
        return cs;
    }

    private float computePercentile(List<Block> blocks, double p) {


        DoubleArrayList dal = new DoubleArrayList(10000);
        if (blocks != null) {
            for (Block b : blocks) {
                for (ContactRecord rec : b.getContactRecords()) {
                    // Filter diagonal
                    if (Math.abs(rec.getBinX() - rec.getBinY()) > 1) {
                        float val = rec.getCounts();  // view with average multiplied

                        dal.add(val);
                    }
                }
            }
        }
        return dal.size() == 0 ? 1 : (float) StatUtils.percentile(dal.toArray(), p);
    }


    /**
     * Render a dense matrix. Used for Pearsons correlation.  The bitmap is drawn at 1 data point
     * per pixel, scaling happens elsewhere.
     *
     * @param rm
     * @param originX    origin in pixels
     * @param originY    origin in pixels
     * @param colorScale
     * @param g
     */
    private void renderMatrix(BasicMatrix rm, int originX, int originY, int width, int height,
                              ColorScale colorScale, Graphics g) {


        int endX = Math.min(originX + width, rm.getColumnDimension());
        int endY = Math.min(originY + height, rm.getRowDimension());

        // TODO -- need to check bounds before drawing
        for (int row = originY; row < endY; row++) {
            for (int col = originX; col < endX; col++) {

                float score = rm.getEntry(row, col);
                Color color;
                if (Float.isNaN(score)) {
                    color = Color.gray;
                } else {
                    color = score == 0 ? Color.black : colorScale.getColor((float) score);
                }
                int px = (int) ((col - originX));
                int py = (int) ((row - originY));
                g.setColor(color);
                g.fillRect(px, py, MainWindow.BIN_PIXEL_WIDTH, MainWindow.BIN_PIXEL_WIDTH);
                // Assuming same chromosome
                if (col != row) {
                    px = (int) ((row - originX));
                    py = (int) ((col - originY));
                    g.fillRect(px, py, MainWindow.BIN_PIXEL_WIDTH, MainWindow.BIN_PIXEL_WIDTH);
                }
            }
        }
    }


    public void setObservedRange(double min, double max) {
        if (observedColorScale == null) {
            observedColorScale = new ContinuousColorScale(min, max, Color.white, Color.red);
        }
        observedColorScale.setNegEnd(min);
        observedColorScale.setPosEnd(max);
    }

    public void setOEMax(double max) {
        ((OEColorScale) oeColorScale).setMax(max);
    }

    public void reset() {
        observedColorScaleMap.clear();
    }
}
