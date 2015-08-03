/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2015 Broad Institute, Aiden Lab
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

import juicebox.MainWindow;
import juicebox.track.feature.CustomAnnotation;
import juicebox.track.feature.Feature2DList;

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * Created by Marie on 6/5/15.
 */
public class SaveAnnotationsDialog extends JFileChooser {

    private static final long serialVersionUID = -6338086600062738308L;
    private final CustomAnnotation annotations;
    private Feature2DList otherList = null;

    public SaveAnnotationsDialog(MainWindow mainWindow, CustomAnnotation customAnnotations) {
        super();
        this.annotations = customAnnotations;
        menuOptions(mainWindow);
    }

    public SaveAnnotationsDialog(MainWindow mainWindow, CustomAnnotation customAnnotations, Feature2DList otherList) {
        super();
        this.annotations = customAnnotations;
        this.otherList = otherList;
        menuOptions(mainWindow);
    }

    private void menuOptions(MainWindow mainWindow) {
        String timeStamp = new SimpleDateFormat("yyyy.MM.dd.HH.mm.ss").format(new Date());
        setSelectedFile(new File("Author-" + timeStamp + ".txt"));

        setCurrentDirectory(new File(System.getProperty("user.dir")));

        FileNameExtensionFilter filter = new FileNameExtensionFilter(
                "Text Files", "txt", "text");
        setFileFilter(filter);
        int actionDialog = showSaveDialog(mainWindow);
        if (actionDialog == JFileChooser.APPROVE_OPTION) {
            File file = getSelectedFile();
            String outputPath = file.getAbsolutePath();
            if (file.exists()) {
                actionDialog = JOptionPane.showConfirmDialog(mainWindow, "Replace existing file?");
                if (actionDialog == JOptionPane.NO_OPTION || actionDialog == JOptionPane.CANCEL_OPTION)
                    return;
            }
            if (otherList == null) {
                if (annotations.exportAnnotations(outputPath) < 0) {
                    JOptionPane.showMessageDialog(mainWindow, "No annotations to output", "Error",
                            JOptionPane.ERROR_MESSAGE);
                }
            } else {
                if (annotations.exportOverlap(otherList, outputPath) < 0) {
                }
            }

        }
    }
}