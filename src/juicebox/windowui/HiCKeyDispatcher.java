/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2016 Broad Institute, Aiden Lab
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.windowui;

import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.data.HiCFileLoader;
import org.broad.igv.ui.util.MessageUtils;

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
        } else if (e.getID() == KeyEvent.KEY_PRESSED && e.getKeyCode() == KeyEvent.VK_F10) {
            String newURL = MessageUtils.showInputDialog("Specify a new properties file",
                    HiCGlobals.defaultPropertiesURL);
            if (newURL != null) {
                HiCFileLoader.changeJuiceboxPropertiesFile(newURL);
            }
            return true;
        } else {
            return false;
        }
    }
}
