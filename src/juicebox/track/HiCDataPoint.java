package juicebox.track;

import org.broad.igv.track.WindowFunction;

/**
 * @author jrobinso
 *         Date: 8/1/13
 *         Time: 6:45 PM
 */
public interface HiCDataPoint {

    double getBinNumber();

    int getGenomicStart();

    int getGenomicEnd();

    double getValue(WindowFunction windowFunction);

    double getWithInBins();

}
