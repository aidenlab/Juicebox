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

package juicebox.windowui;

/**
 * @author jrobinso Date: 8/31/13  9:47 PM
 */
public enum NormalizationType {
    NONE("None"),
    VC("Coverage"),
    VC_SQRT("Coverage (Sqrt)"),
    KR("Balanced"),
    GW_KR("Genome-wide balanced"),
    INTER_KR("Inter balanced"),
    GW_VC("Genome-wide coverage"),
    INTER_VC("Inter coverage"),
    LOADED("Loaded");
    private final String label;

    NormalizationType(String label) {
        this.label = label;
    }

    public String getLabel() {
        return label;
    }

}
