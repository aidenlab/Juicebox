/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2018 Broad Institute, Aiden Lab
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

package juicebox.assembly;

import juicebox.HiCGlobals;
import juicebox.gui.SuperAdapter;

import javax.swing.*;

/**
 * Created by olga on 11/22/17.
 */
public class UnsavedAssemblyWarning {

    private final SuperAdapter superAdapter;

    public UnsavedAssemblyWarning(SuperAdapter superAdapter) {
        this.superAdapter = superAdapter;
    }

    public boolean checkAndDelete() {
        superAdapter.getAssemblyStateTracker().resetState();
        superAdapter.getHeatmapPanel().disableAssemblyEditing();
        HiCGlobals.assemblyModeEnabled = false;
        HiCGlobals.hicMapScale = 1;
        superAdapter.resetAnnotationLayers();
//&& superAdapter.unsavedEditsExist()
        if (HiCGlobals.assemblyModeEnabled) {

            //return this.initPopup();
        }
        // There were no unsaved annotations, proceed
        return true;
    }

    private boolean initPopup() {
        // Check for unsaved annotations
        JDialog.setDefaultLookAndFeelDecorated(true);
        //Custom button text
        Object[] options = {"Save Annotations",
                "Discard Annotations", "Continue"};

        int response = JOptionPane.showOptionDialog(null,
                "There are unsaved hand annotations from this (or a previous) session.\n" +
                        "Would you like save them before you continue?",
                "Confirm",
                JOptionPane.YES_NO_CANCEL_OPTION,
                JOptionPane.QUESTION_MESSAGE,
                null,     //do not use a custom Icon
                options,  //the titles of buttons
                options[0]); //default button title

        if (response == JOptionPane.NO_OPTION) {
            //System.out.println("Deleting annotations");
            superAdapter.getActiveLayerHandler().clearAnnotations();
            return true;
        } else if (response == JOptionPane.CANCEL_OPTION || response == JOptionPane.CLOSED_OPTION) {
            //System.out.println("meh i don't wanna commit");
            return false;
        } else if (response == JOptionPane.YES_OPTION) {
            System.out.println("Saving annotations");
            // Save the annotations
            //superAdapter.exportAnnotations();
            superAdapter.getActiveLayerHandler().clearAnnotations();
            return true;
        }
        return false;
    }

}
