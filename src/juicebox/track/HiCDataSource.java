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

    void setName(String text);

    Color getColor();

    void setColor(Color selectedColor);

    Color getAltColor();

    void setAltColor(Color selectedColor);

    DataRange getDataRange();

    void setDataRange(DataRange dataRange);

    boolean isLog();

    HiCDataPoint[] getData(Chromosome chr, int startBin, int endBin, HiCGridAxis gridAxis, double scaleFactor, WindowFunction windowFunction);

    Collection<WindowFunction> getAvailableWindowFunctions();
}
