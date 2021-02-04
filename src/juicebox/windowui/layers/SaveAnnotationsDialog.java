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

import juicebox.HiCGlobals;
import juicebox.MainWindow;
import juicebox.track.feature.AnnotationLayer;
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

    private static final long serialVersionUID = 9000010;
    private final AnnotationLayer annotations;
    private final Feature2DList otherList = null;
    private String mapName = "";

    public SaveAnnotationsDialog(AnnotationLayer annotationsLayer, String mapName) {
        super();
        this.annotations = annotationsLayer;
        this.mapName = mapName;
        menuOptions();
    }

    private void menuOptions() {
        String timeStamp = new SimpleDateFormat("yyyy.MM.dd-HH.mm").format(new Date());
        setSelectedFile(new File(mapName + "-" + timeStamp + ".bedpe"));

        //setCurrentDirectory(new File(System.getProperty("user.dir")));

        FileNameExtensionFilter filter = new FileNameExtensionFilter(
                "BEDPE Files", "bedpe", "txt", "text");
        setFileFilter(filter);
        if (HiCGlobals.guiIsCurrentlyActive) {
            int actionDialog = showSaveDialog(MainWindow.getInstance());
            if (actionDialog == JFileChooser.APPROVE_OPTION) {
                File file = getSelectedFile();
                String outputPath = file.getAbsolutePath();
                if (file.exists()) {
                    actionDialog = JOptionPane.showConfirmDialog(MainWindow.getInstance(), "Replace existing file?");
                    if (actionDialog == JOptionPane.NO_OPTION || actionDialog == JOptionPane.CANCEL_OPTION)
                        return;
                }
                if (otherList == null) {
                    if (!annotations.exportAnnotations(outputPath)) {
                        JOptionPane.showMessageDialog(MainWindow.getInstance(), "No annotations to output", "Error",
                                JOptionPane.ERROR_MESSAGE);
                    }
                } else {
                    if (!annotations.exportOverlap(otherList, outputPath)) {
                        JOptionPane.showMessageDialog(MainWindow.getInstance(), "Unable to export annotations", "Error",
                                JOptionPane.ERROR_MESSAGE);
                    }
                }
            }
        }
    }
}