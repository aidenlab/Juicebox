/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2021 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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
 *  FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.windowui.layers;

import juicebox.DirectoryManager;
import juicebox.HiCGlobals;
import juicebox.gui.SuperAdapter;

import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import juicebox.track.feature.AnnotationLayerHandler;

/**
 * Created by Marie on 12/21/15.
 */
public class UnsavedAnnotationWarning {

    private final SuperAdapter superAdapter;

    public UnsavedAnnotationWarning(SuperAdapter adapter) {
        this.superAdapter = adapter;
    }

    public boolean checkAndDelete(boolean isCurrentSession) {
        if (superAdapter.unsavedEditsExist()) {
            return initPopup(isCurrentSession);
        }
        // There were no unsaved annotations, proceed
        return true;
    }


    private boolean initPopup(boolean isCurrentSession) {
        // Check for unsaved annotations
        JDialog.setDefaultLookAndFeelDecorated(true);
        //Custom button text
        Object[] options = {"Save Annotations", "Discard Annotations"};

        String sessionTime = isCurrentSession ? "this" : "a previous";
        int response = JOptionPane.showOptionDialog(null,
                "There are unsaved hand annotations from " + sessionTime + " session.\n" +
                        "Would you like save them before you continue?",
                "Confirm",
                JOptionPane.YES_NO_OPTION,
                JOptionPane.QUESTION_MESSAGE,
                null,     //do not use a custom Icon
                options,  //the titles of buttons
                options[0]); //default button title

        if (response == JOptionPane.NO_OPTION) {
            System.out.println("Deleting annotations");
            if (isCurrentSession) {
                for (AnnotationLayerHandler layer : superAdapter.getAllLayers()) layer.clearAnnotations();
            }
            try {
                removeAllOldAnnotationFiles();
                return true;
            } catch (IOException e) {
                SuperAdapter.showMessageDialog("Could not discard annotation backups: " + e.getMessage());
                return false;
            }
        }
        if (response == JOptionPane.CANCEL_OPTION || response == JOptionPane.CLOSED_OPTION) {
            System.out.println("Cancel");
            return false;
        }
        if (response == JOptionPane.YES_OPTION) {
            if (isCurrentSession) {
                for (AnnotationLayerHandler layer : superAdapter.getAllLayers()) layer.getAnnotationLayer().closeAutosave();
            }
            try {
                String prefix = moveOldAnnotationFiles();
                SuperAdapter.showMessageDialog("Files have been saved with prefix: " + prefix + "*\nin " + DirectoryManager.getHiCDirectory());
                return true;
            } catch (IOException e) {
                SuperAdapter.showMessageDialog("Could not save annotation backups: " + e.getMessage());
                return false;
            }
        }
        return false;
    }

    /**
     * simple deletion based on assumption of limited possible files
     * temp fix
     * todo something more sophisticated
     */
    private void removeAllOldAnnotationFiles() throws IOException {
        for (File temp : getBackupFiles()) {
            Files.delete(temp.toPath());
        }
    }

    private String moveOldAnnotationFiles() throws IOException {
        String timeStamp = System.nanoTime() + "_annotations_";
        for (File temp : getBackupFiles()) {
            String suffix = temp.getName().substring(HiCGlobals.BACKUP_FILE_STEM.length());
            Files.move(temp.toPath(), new File(DirectoryManager.getHiCDirectory(), timeStamp + suffix).toPath());
        }
        return timeStamp;
    }

    public static File[] getBackupFiles() {
        File[] files = DirectoryManager.getHiCDirectory().listFiles((directory, name) ->
                name.startsWith(HiCGlobals.BACKUP_FILE_STEM) &&
                        name.substring(HiCGlobals.BACKUP_FILE_STEM.length()).matches("[0-9]+\\.bedpe"));
        return files == null ? new File[0] : files;
    }

}
