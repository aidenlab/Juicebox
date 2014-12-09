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
