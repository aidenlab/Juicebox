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

import juicebox.HiC;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyEvent;

public class HiCKeyDispatcher implements KeyEventDispatcher {

    private final HiC hic;
    private final JComboBox<MatrixType> displayOptionComboBox;

    public HiCKeyDispatcher(HiC hic, JComboBox<MatrixType> displayOptionComboBox) {
        super();
        this.hic = hic;
        this.displayOptionComboBox = displayOptionComboBox;
    }

    @Override
    public boolean dispatchKeyEvent(KeyEvent e) {

        if (e.getID() == KeyEvent.KEY_PRESSED && e.getKeyCode() == KeyEvent.VK_F1) {

            if (hic.getControlZd() != null) {
                MatrixType displayOption = (MatrixType) displayOptionComboBox.getSelectedItem();
                if (displayOption == MatrixType.CONTROL) {
                    displayOptionComboBox.setSelectedItem(MatrixType.OBSERVED);

                } else if (displayOption == MatrixType.OBSERVED) {
                    displayOptionComboBox.setSelectedItem(MatrixType.CONTROL);
                }

            }
            return true;
        } else {

            return false;
        }
    }
}
