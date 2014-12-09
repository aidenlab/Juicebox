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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * List of two-dimensional features.  Hashtable for each chromosome for quick viewing.
 * Visibility depends on user selection.
 *
 * @author Neva Durand
 * @since 9/2013
 */
public class Feature2DList {

    /**
     * List of 2D features stored by chromosome
     */
    private final Map<String, List<Feature2D>> featureList;

    /**
     * Visibility as set by user
     */
    private boolean isVisible;

    /**
     * Initialized hashtable
     */
    public Feature2DList() {
        featureList = new HashMap<String, List<Feature2D>>();
        isVisible = true;
    }

    /**
     * Returns list of features on this chromosome pair
     *
     * @param chr1Idx First chromosome index
     * @param chr2Idx Second chromosome index
     * @return List of 2D features at that point
     */
    public List<Feature2D> get(int chr1Idx, int chr2Idx) {
        return featureList.get(getKey(chr1Idx, chr2Idx));
    }

    /**
     * Adds feature to appropriate chromosome pair list; key stored so that first chromosome always less than second
     *
     * @param chr1Idx First chromosome index
     * @param chr2Idx Second chromosome index
     * @param feature Feature to add
     */
    public void add(int chr1Idx, int chr2Idx, Feature2D feature) {

        String key = getKey(chr1Idx, chr2Idx);
        List<Feature2D> loops = featureList.get(key);
        if (loops == null) {
            loops = new ArrayList<Feature2D>();
            loops.add(feature);
            featureList.put(key, loops);
        } else {
            loops.add(feature);
        }

    }

    /**
     * Returns visibility of list
     *
     * @return If list is visible
     */
    public boolean isVisible() {
        return isVisible;
    }

    /**
     * Set visibility of list
     *
     * @param flag Visibility
     */
    public void setVisible(boolean flag) {
        isVisible = flag;
    }

    /**
     * Helper method to get the key, lowest ordinal chromosome first
     *
     * @param chr1Idx First chromosome index
     * @param chr2Idx Second chromosome index
     * @return key
     */
    private String getKey(int chr1Idx, int chr2Idx) {

        int c1;
        int c2;
        if (chr1Idx < chr2Idx) {
            c1 = chr1Idx;
            c2 = chr2Idx;
        } else {
            c1 = chr2Idx;
            c2 = chr1Idx;
        }

        return "" + c1 + "_" + c2;
    }

}
