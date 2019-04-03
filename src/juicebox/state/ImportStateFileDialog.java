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

package juicebox.state;

import juicebox.MainWindow;
import org.xml.sax.ErrorHandler;
import org.xml.sax.InputSource;
import org.xml.sax.SAXParseException;

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.channels.FileChannel;


/**
 * Created by Zulkifl on 7/23/2015.
 */
public class ImportStateFileDialog extends JFileChooser {

    private static final long serialVersionUID = -1038991737399792883L;
    private final File originalStates = new File("OriginalSavedStates.xml");

    public ImportStateFileDialog(File currentStates, MainWindow mainWindow) {
        super();
        loadFile(currentStates, mainWindow);
    }

    private static void copyFile(File sourceFile, File destFile) throws IOException {
        if (!destFile.exists()) {
            destFile.createNewFile();
        }

        FileChannel source = null;
        FileChannel destination = null;

        try {
            source = new FileInputStream(sourceFile).getChannel();
            destination = new FileOutputStream(destFile).getChannel();
            destination.transferFrom(source, 0, source.size());
        } finally {
            if (source != null) {
                source.close();
            }
            if (destination != null) {
                destination.close();
            }
        }
    }

    private void loadFile(File currentFile, MainWindow mainWindow) {

        FileNameExtensionFilter filter = new FileNameExtensionFilter(
                "XML Files", "xml", "XML");
        setFileFilter(filter);

        int actionDialog = showOpenDialog(mainWindow);
        if (actionDialog == APPROVE_OPTION) {
            File importedFile = getSelectedFile();
            String path = importedFile.getAbsolutePath();
            mainWindow.updateNamesFromImport(path);

            try {
                DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
                factory.setValidating(false);
                factory.setNamespaceAware(true);
                DocumentBuilder builder = factory.newDocumentBuilder();
                builder.setErrorHandler(new SimpleErrorHandler());
                builder.parse(new InputSource(path));
                copyFile(currentFile, originalStates);
                copyFile(importedFile, currentFile);
                JOptionPane.showMessageDialog(mainWindow, "Importing File:\n" + importedFile.getName(), "Opening",
                        JOptionPane.INFORMATION_MESSAGE);
                JOptionPane.showMessageDialog(mainWindow, "Previous states have been saved under file:\n" + originalStates.getName(), "Creating Backup",
                        JOptionPane.INFORMATION_MESSAGE);
            } catch (Exception e) {
                JOptionPane.showMessageDialog(mainWindow, "Error while importing file:\n" + e.getMessage(), "Error",
                        JOptionPane.ERROR_MESSAGE);
                e.printStackTrace();
            }

            System.out.println("Opening File: " + importedFile.getName());

        }
    }

    private static class SimpleErrorHandler implements ErrorHandler {
        public void warning(SAXParseException e) {
            System.out.println(e.getMessage());
        }

        public void error(SAXParseException e) {
            System.out.println(e.getMessage());
        }

        public void fatalError(SAXParseException e) {
            System.out.println(e.getMessage());
        }
    }


}

