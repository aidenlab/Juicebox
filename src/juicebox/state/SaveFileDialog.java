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

package juicebox.state;

import juicebox.MainWindow;

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * Created by Zulkifl on 7/23/2015.
 */
public class SaveFileDialog extends JFileChooser {

    private static final long serialVersionUID = 2910799798390074194L;
    private File fileForExport;
    private int numStates = 0;


    public SaveFileDialog(File fileToSave, int savedStates) {
        super();
        this.fileForExport = fileToSave;
        numStates = savedStates;
        saveFile(fileToSave);

    }

    private void saveFile(File fileSave) {
        //String timeStamp = new SimpleDateFormat("yyyy.MM.dd.HH.mm.ss").format(new Date());
        //fileSave.renameTo(new File("Saved HiC Files- " + timeStamp + ".xml"));
        setSelectedFile(fileSave);
        setCurrentDirectory(new File(System.getProperty("user.dir")));
        FileNameExtensionFilter filter = new FileNameExtensionFilter(
                "XML Files", "xml", "XML");
        setFileFilter(filter);
        int actionDialog = showSaveDialog(MainWindow.getInstance());
        if (actionDialog == JFileChooser.APPROVE_OPTION) {
            File file = getSelectedFile();
            String savedFilePath = fileSave.getAbsolutePath();
            if (file.exists()) {
                if(numStates > 0) {
                    actionDialog = JOptionPane.showConfirmDialog(MainWindow.getInstance(), "Replace existing file?");
                    if (actionDialog == JOptionPane.NO_OPTION || actionDialog == JOptionPane.CANCEL_OPTION)
                        return;
                } else if (numStates == 0) {
                    JOptionPane.showMessageDialog(MainWindow.getInstance(), "No saved HiC maps to export", "Error",
                            JOptionPane.ERROR_MESSAGE);
                }
            }

        }

    }
}
