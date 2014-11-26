package juicebox.data;

import juicebox.windowui.NormalizationType;

//import java.util.Map;

/**
 * @author jrobinso
 *         Date: 12/26/12
 *         Time: 9:30 PM
 */
public interface ExpectedValueFunction {

    double getExpectedValue(int chrIdx, int distance);

    int getLength();

    NormalizationType getNormalizationType();

    String getUnit();

    int getBinSize();

    double[] getExpectedValues();
}
