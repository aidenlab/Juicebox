/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2017 Broad Institute, Aiden Lab
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

import juicebox.gui.SuperAdapter;

import javax.swing.*;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;

/**
 * Created by ragibmostofa on 6/13/17.
 */
public class SelectChromosomesDialog extends JDialog {
    private static final int DIALOG_WIDTH = 800;
    private static final int DIALOG_HEIGHT = 600;
    private boolean showSelectChromosomesDialog = false;

    public SelectChromosomesDialog(final SuperAdapter superAdapter) {
        super(superAdapter.getMainWindow(), "Select Custom Chromosomes");

        setSize(getDialogWidth(), getDialogHeight());
        setVisible(getShowSelectChromosomesDialog());

        this.addWindowListener(new WindowListener() {
            @Override
            public void windowOpened(WindowEvent e) {

            }

            @Override
            public void windowClosing(WindowEvent e) {
                setShowSelectChromosomesDialog(!getShowSelectChromosomesDialog());
            }

            @Override
            public void windowClosed(WindowEvent e) {

            }

            @Override
            public void windowIconified(WindowEvent e) {

            }

            @Override
            public void windowDeiconified(WindowEvent e) {

            }

            @Override
            public void windowActivated(WindowEvent e) {

            }

            @Override
            public void windowDeactivated(WindowEvent e) {

            }
        });
    }

    public int getDialogWidth() {
        return DIALOG_WIDTH;
    }

    public int getDialogHeight() {
        return DIALOG_HEIGHT;
    }

    public boolean getShowSelectChromosomesDialog() {
        return this.showSelectChromosomesDialog;
    }

    public void setShowSelectChromosomesDialog(boolean showSelectChromosomesDialog) {
        this.showSelectChromosomesDialog = showSelectChromosomesDialog;
        setVisible(getShowSelectChromosomesDialog());
    }

}
