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

/**
 * @author jrobinso
 *         Date: 9/14/12
 *         Time: 8:54 AM
 */
public interface HiCGridAxis {

    int getGenomicStart(double binNumber);

    int getGenomicEnd(double binNumber);

    int getGenomicMid(double binNumber);

    int getIGVZoom();

    int getBinCount();

    int getBinNumberForGenomicPosition(int genomePosition);

    int getBinNumberForFragment(int fragmentX);
}
