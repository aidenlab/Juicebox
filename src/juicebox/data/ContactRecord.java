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

package juicebox.data;

/**
 * @author jrobinso
 * @date Aug 3, 2010
 */
public class ContactRecord implements Comparable<ContactRecord> {

    /**
     * Bin number in x coordinate
     */
    private final int binX;

    /**
     * Bin number in y coordinate
     */
    private final int binY;

    /**
     * Total number of counts, or cumulative score
     */
    private float counts;
    private String key;

    public ContactRecord(int binX, int binY, float counts) {
        this.binX = binX;
        this.binY = binY;
        this.counts = counts;
    }

    public void incrementCount(float score) {
        counts += score;
    }


    public int getBinX() {
        return binX;
    }

    public int getBinY() {
        return binY;
    }

    public float getCounts() {
        return counts;
    }

    @Override
    public int compareTo(ContactRecord contactRecord) {
        if (this.binX != contactRecord.binX) {
            return binX - contactRecord.binX;
        } else if (this.binY != contactRecord.binY) {
            return binY - contactRecord.binY;
        } else return 0;
    }

    public String toString() {
        return "" + binX + " " + binY + " " + counts;
    }

    public String getKey() {
        if (key == null) {
            key = binX + "_" + binY;
        }
        return key;
    }
}
