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

package juicebox.state.archive;

import juicebox.DirectoryManager;
import juicebox.HiCGlobals;
import juicebox.MainWindow;
import juicebox.windowui.RecentMenu;
import org.apache.commons.io.FileUtils;

import org.broad.igv.ui.util.FileDialogUtils;
import juicebox.windowui.RecentMenu;
import org.w3c.dom.Document;
import org.xml.sax.ErrorHandler;
import org.xml.sax.InputSource;
import org.xml.sax.SAXException;
import org.xml.sax.SAXParseException;
import sun.applet.Main;

import javax.swing.*;
import javax.swing.filechooser.FileFilter;
import javax.swing.filechooser.FileNameExtensionFilter;
import javax.xml.XMLConstants;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.Source;
import javax.xml.transform.stream.StreamSource;
import javax.xml.validation.Schema;
import javax.xml.validation.SchemaFactory;
import javax.xml.validation.Validator;
import java.io.*;
import java.net.URL;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;



/**
 * Created by Zulkifl on 7/23/2015.
 */
public class ImportFileDialog extends JFileChooser {

    private static final long serialVersionUID = -1038991737399792883L;
    private File currentJuiceboxStates = new File(HiCGlobals.xmlSavedStatesFileName);
    private MainWindow mainWindow;
    private File originalStates = new File("OriginalJuiceboxSavedStates.xml");

    public ImportFileDialog(File currentStates, MainWindow mainWindow ) {
        super();
        this.mainWindow = mainWindow;
        loadFile(currentStates);
    }

    private void loadFile(File currentFile){

        FileNameExtensionFilter filter = new FileNameExtensionFilter(
                "XML Files", "xml", "XML");
        setFileFilter(filter);

        int actionDialog = showOpenDialog(MainWindow.getInstance());
        if(actionDialog == APPROVE_OPTION){
            File importedFile = getSelectedFile();
            String path = importedFile.getAbsolutePath();
            //mainWindow.getPrevousStateMenu().updateNamesFromImport(path); OLD TODO Delete

            try {
                DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
                factory.setValidating(false);
                factory.setNamespaceAware(true);
                DocumentBuilder builder = factory.newDocumentBuilder();
                builder.setErrorHandler(new SimpleErrorHandler());
                Document document = builder.parse(new InputSource(path));
                copyFile(currentFile,originalStates);
                copyFile(importedFile,currentFile);
                JOptionPane.showMessageDialog(MainWindow.getInstance(), "Importing File:\n" + importedFile.getName(),"Opening",
                        JOptionPane.INFORMATION_MESSAGE);
            } catch (IOException e) {
                JOptionPane.showMessageDialog(MainWindow.getInstance(), "Error while importing file:\n" + e.getMessage(), "Error",
                        JOptionPane.ERROR_MESSAGE);
                e.printStackTrace();
            }
            catch (ParserConfigurationException pce) {
                JOptionPane.showMessageDialog(MainWindow.getInstance(), "Error while importing file:\n" + pce.getMessage(), "Error",
                        JOptionPane.ERROR_MESSAGE);
                pce.printStackTrace();
            } catch (SAXException sax) {
                JOptionPane.showMessageDialog(MainWindow.getInstance(), "Error while importing file:\n" + sax.getMessage(), "Error",
                        JOptionPane.ERROR_MESSAGE);
                sax.printStackTrace();
            }

            System.out.println("Opening File: " + importedFile.getName());

        }
    }

    private static void copyFile(File sourceFile, File destFile) throws IOException {
        if(!destFile.exists()) {
            destFile.createNewFile();
        }

        FileChannel source = null;
        FileChannel destination = null;

        try {
            source = new FileInputStream(sourceFile).getChannel();
            destination = new FileOutputStream(destFile).getChannel();
            destination.transferFrom(source, 0, source.size());
        }
        finally {
            if(source != null) {
                source.close();
            }
            if(destination != null) {
                destination.close();
            }
        }
    }

    public static class SimpleErrorHandler implements ErrorHandler {
        public void warning(SAXParseException e) throws SAXException {
            System.out.println(e.getMessage());
        }

        public void error(SAXParseException e) throws SAXException {
            System.out.println(e.getMessage());
        }

        public void fatalError(SAXParseException e) throws SAXException {
            System.out.println(e.getMessage());
        }
    }


}

