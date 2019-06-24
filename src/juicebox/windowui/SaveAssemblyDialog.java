/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2019 Broad Institute, Aiden Lab
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

import juicebox.HiCGlobals;
import juicebox.MainWindow;
import juicebox.assembly.AssemblyFileExporter;
import juicebox.assembly.AssemblyScaffoldHandler;

import javax.swing.*;
import java.io.File;

/**
 * Created by nathanielmusial on 6/30/17.
 */
public class SaveAssemblyDialog extends JFileChooser {

    private static final long serialVersionUID = 18237123123L;

    private final AssemblyScaffoldHandler assemblyScaffoldHandler;
    private final String mapName;

    public SaveAssemblyDialog(AssemblyScaffoldHandler assemblyScaffoldHandler, String mapName) {
        super();
        this.mapName = mapName;
        this.assemblyScaffoldHandler = assemblyScaffoldHandler;
        menuOptions();
    }

    // todo see SaveAnnotationsDialog
    // todo add file extension filter
    private void menuOptions() {
        setSelectedFile(new File(mapName + ".review"));
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
                AssemblyFileExporter assemblyFileExporter = new AssemblyFileExporter(assemblyScaffoldHandler, outputPath);
                assemblyFileExporter.exportAssemblyFile();
            }
        }
    }
}
