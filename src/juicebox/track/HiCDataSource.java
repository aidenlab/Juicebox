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
