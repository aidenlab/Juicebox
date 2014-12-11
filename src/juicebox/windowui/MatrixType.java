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


public enum MatrixType {
    OBSERVED("Observed"),
    OE("O/E"),
    PEARSON("Pearson"),
    EXPECTED("Expected"),
    RATIO("Observed/Control"),
    CONTROL("Control");
    private final String value;

    MatrixType(String value) {
        this.value = value;
    }

    public String toString() {
        return value;
    }

}
