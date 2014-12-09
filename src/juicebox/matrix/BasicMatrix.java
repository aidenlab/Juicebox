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

package juicebox.matrix;

/**
 * @author jrobinso
 *         Date: 7/13/12
 *         Time: 1:05 PM
 */
public interface BasicMatrix {

    float getEntry(int row, int col);

    int getRowDimension();

    int getColumnDimension();

    float getLowerValue();

    float getUpperValue();

    void setEntry(int i, int j, float corr);
}
