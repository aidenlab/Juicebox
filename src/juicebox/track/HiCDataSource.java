package juicebox.track;

import org.broad.igv.feature.Chromosome;
import org.broad.igv.renderer.DataRange;
import org.broad.igv.track.WindowFunction;

import java.awt.*;
import java.util.Collection;

/**
 * @author jrobinso
 *         Date: 8/1/13
 *         Time: 7:51 PM
 */
public interface HiCDataSource {

    String getName();

    Color getColor();

    Color getAltColor();

    DataRange getDataRange();

    boolean isLog();

    void setDataRange(DataRange dataRange);

    void setName(String text);

    void setColor(Color selectedColor);

    void setAltColor(Color selectedColor);

    HiCDataPoint[] getData(Chromosome chr, int startBin, int endBin, HiCGridAxis gridAxis, double scaleFactor, WindowFunction windowFunction);

    Collection<WindowFunction> getAvailableWindowFunctions();
}
