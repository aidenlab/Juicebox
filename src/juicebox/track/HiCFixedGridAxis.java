package juicebox.track;

import org.broad.igv.Globals;

/**
 * @author jrobinso
 *         Date: 9/14/12
 *         Time: 8:54 AM
 */
public class HiCFixedGridAxis implements HiCGridAxis {

    private final int binCount;
    private final int binSize;
    private final int igvZoom;
    private final int[] sites;

    public HiCFixedGridAxis(int binCount, int binSize, int [] sites) {

        this.binCount = binCount;
        this.binSize = binSize;
        this.sites = sites;

        // Compute an approximate igv zoom level
        igvZoom = Math.max(0, (int) (Math.log(binCount / 700) / Globals.log2));

    }

    @Override
    public int getGenomicStart(double binNumber) {
        return (int) (binNumber * binSize);
    }

    @Override
    public int getGenomicEnd(double binNumber) {
        return (int) ((binNumber + 1) * binSize);
    }

    @Override
    public int getGenomicMid(double binNumber) {
        return (int) ((binNumber + 0.5) * binSize);
    }

    @Override
    public int getIGVZoom() {
        return igvZoom;
    }

    @Override
    public int getBinNumberForGenomicPosition(int genomicPosition) {
        return (int) (genomicPosition / ((double) binSize));
    }

    @Override
    public int getBinNumberForFragment(int fragment) {

        if (fragment < sites.length && fragment >= 0) {
            int genomicPosition = sites[fragment];
            return getBinNumberForGenomicPosition(genomicPosition);
        }
        throw new RuntimeException("Fragment: " + fragment + " is out of range");
    }

    @Override
    public int getBinCount() {
        return binCount;
    }

}
