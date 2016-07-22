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

import de.erichseifert.vectorgraphics2d.PDFGraphics2D;
import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.MainWindow;
import org.apache.batik.dom.GenericDOMImplementation;
import org.apache.batik.svggen.SVGGraphics2D;
import org.w3c.dom.DOMImplementation;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;


public class SaveImageDialog extends JFileChooser {
    private static final long serialVersionUID = -611947177404923808L;
    private JTextField width;
    private JTextField height;

    public SaveImageDialog(String saveImagePath, final HiC hic, final MainWindow mainWindow, final JPanel hiCPanel) {
        super();
        if (saveImagePath != null) {
            setSelectedFile(new File(saveImagePath));
        } else {
            String timeStamp = new SimpleDateFormat("yyyy.MM.dd.HH.mm.ss").format(new Date());
            setSelectedFile(new File(timeStamp + ".HiCImage.svg"));


        }
        if (HiCGlobals.guiIsCurrentlyActive) {
            int actionDialog = showSaveDialog(mainWindow);
            if (actionDialog == JFileChooser.APPROVE_OPTION) {
                File selectedFile = getSelectedFile();
                final File outputFile;
                if (selectedFile.getPath().endsWith(".svg") || selectedFile.getPath().endsWith(".SVG")
                        || selectedFile.getPath().endsWith(".pdf") || selectedFile.getPath().endsWith(".PDF")) {
                    outputFile = selectedFile;
                } else {
                    outputFile = new File(selectedFile + ".pdf");
                }
                //saveImagePath = file.getPath();
                if (outputFile.exists()) {
                    actionDialog = JOptionPane.showConfirmDialog(MainWindow.getInstance(), "Replace existing file?");
                    if (actionDialog == JOptionPane.NO_OPTION || actionDialog == JOptionPane.CANCEL_OPTION)
                        return;
                }

                mainWindow.executeLongRunningTask(new Runnable() {
                    @Override
                    public void run() {
                        try {
                            int w = Integer.valueOf(width.getText());
                            int h = Integer.valueOf(height.getText());
                            if (HiCGlobals.printVerboseComments) System.out.println("Exporting another figure");
                            if (outputFile.getPath().endsWith(".svg") || outputFile.getPath().endsWith(".SVG")) {
                                exportAsSVG(outputFile, mainWindow, hic, hiCPanel, w, h);
                            } else {
                                exportAsPDF(outputFile, mainWindow, hic, hiCPanel, w, h);
                            }

                        } catch (IOException error) {
                            JOptionPane.showMessageDialog(mainWindow, "Error while saving file:\n" + error, "Error",
                                    JOptionPane.ERROR_MESSAGE);
                        } catch (NumberFormatException error) {
                            JOptionPane.showMessageDialog(mainWindow, "Width and Height must be integers", "Error",
                                    JOptionPane.ERROR_MESSAGE);
                        }
                    }
                }, "Exporting Figure", "Exporting...");
            }
        }
    }

    protected JDialog createDialog(Component parent) {
        JDialog myDialog = super.createDialog(parent);
        JLabel wLabel = new JLabel("Width");
        JLabel hLabel = new JLabel("Height");
        width = new JTextField("" + MainWindow.getInstance().getWidth());
        width.setColumns(6);
        height = new JTextField("" + MainWindow.getInstance().getHeight());
        height.setColumns(6);
        JPanel panel = new JPanel();
        panel.add(wLabel);
        panel.add(width);
        panel.add(hLabel);
        panel.add(height);
        myDialog.add(panel, BorderLayout.NORTH);
        return myDialog;
    }

    private void exportAsSVG(File file, MainWindow mainWindow, HiC hic, final JPanel hiCPanel,
                             final int w, final int h) throws IOException {

        try {

            // Get a DOMImplementation.
            DOMImplementation domImpl = GenericDOMImplementation.getDOMImplementation();
            Dimension size = mainWindow.getSize();

            // Create an instance of org.w3c.dom.Document.
            String svgNS = "http://www.w3.org/2000/svg";
            org.w3c.dom.Document document = domImpl.createDocument(svgNS, "svg", null);

            // Create an instance of the SVG Generator.
            SVGGraphics2D svgGenerator = new SVGGraphics2D(document);

            // Ask the test to render into the SVG Graphics2D implementation.

            // Print the panel on created SVG graphics.
            if (w == mainWindow.getWidth() && h == mainWindow.getHeight()) {
                hiCPanel.printAll(svgGenerator);
            } else {
                JDialog waitDialog = new JDialog();
                JPanel panel1 = new JPanel();
                panel1.add(new JLabel("  Creating and saving " + w + " by " + h + " image  "));
                //panel1.setPreferredSize(new Dimension(250,50));
                waitDialog.add(panel1);
                waitDialog.setTitle("Please wait...");
                waitDialog.pack();
                waitDialog.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);

                waitDialog.setLocation(100, 100);
                waitDialog.setVisible(true);
                mainWindow.setVisible(false);

                Dimension minSize = mainWindow.getMinimumSize();
                Dimension prefSize = mainWindow.getPreferredSize();

                hic.centerBP(0, 0);
                mainWindow.setMinimumSize(new Dimension(w, h));
                mainWindow.setPreferredSize(new Dimension(w, h));
                mainWindow.pack();

                mainWindow.setState(Frame.ICONIFIED);
                mainWindow.setState(Frame.NORMAL);
                mainWindow.setVisible(true);
                mainWindow.setVisible(false);

                final Runnable painter = new Runnable() {
                    public void run() {
                        hiCPanel.paintImmediately(0, 0, w, h);
                    }
                };

                Thread thread = new Thread(painter) {
                    public void run() {

                        try {
                            SwingUtilities.invokeAndWait(painter);
                        } catch (Exception e) {
                            e.printStackTrace();
                        }

                    }
                };

                thread.start();
                hiCPanel.printAll(svgGenerator);
                mainWindow.setPreferredSize(prefSize);
                mainWindow.setMinimumSize(minSize);
                mainWindow.setSize(size);
                waitDialog.setVisible(false);
                waitDialog.dispose();
                mainWindow.setVisible(true);
            }

            // Finally, stream out SVG to the standard output using
            // UTF-8 encoding.
            boolean useCSS = true; // we want to use CSS style attributes
            svgGenerator.stream(new FileWriter(file), useCSS);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void exportAsPDF(File file, MainWindow mainWindow, HiC hic, final JPanel hiCPanel,
                             final int w, final int h) throws IOException {

        try {

            //PDF pdf = new PDF(new BufferedOutputStream(new FileOutputStream(file)));


            try {
                PDFGraphics2D g = new PDFGraphics2D(0, 0, w, h);

                // Print the panel on created graphics.
                if (w == mainWindow.getWidth() && h == mainWindow.getHeight()) {
                    hiCPanel.printAll(g);
                } else {
                    JDialog waitDialog = new JDialog();
                    JPanel panel1 = new JPanel();
                    panel1.add(new JLabel("  Creating and saving " + w + " by " + h + " image  "));
                    //panel1.setPreferredSize(new Dimension(250,50));
                    waitDialog.add(panel1);
                    waitDialog.setTitle("Please wait...");
                    waitDialog.pack();
                    waitDialog.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);

                    waitDialog.setLocation(100, 100);
                    waitDialog.setVisible(true);
                    mainWindow.setVisible(false);

                    Dimension minSize = mainWindow.getMinimumSize();
                    Dimension prefSize = mainWindow.getPreferredSize();

                    hic.centerBP(0, 0);
                    mainWindow.setMinimumSize(new Dimension(w, h));
                    mainWindow.setPreferredSize(new Dimension(w, h));
                    mainWindow.pack();

                    mainWindow.setState(Frame.ICONIFIED);
                    mainWindow.setState(Frame.NORMAL);
                    mainWindow.setVisible(true);
                    mainWindow.setVisible(false);

                    final Runnable painter = new Runnable() {
                        public void run() {
                            hiCPanel.paintImmediately(0, 0, w, h);
                        }
                    };

                    Thread thread = new Thread(painter) {
                        public void run() {

                            try {
                                SwingUtilities.invokeAndWait(painter);
                            } catch (Exception e) {
                                e.printStackTrace();
                            }

                        }
                    };

                    thread.start();
                    hiCPanel.printAll(g);
                    mainWindow.setPreferredSize(prefSize);
                    mainWindow.setMinimumSize(minSize);
                    mainWindow.setSize(new Dimension(w, h));
                    waitDialog.setVisible(false);
                    waitDialog.dispose();
                    mainWindow.setVisible(true);
                }


                System.out.println(g.getFont());

                FileOutputStream fileOutputStream = new FileOutputStream(file);
                try {
                    fileOutputStream.write(g.getBytes());
                } finally {
                    fileOutputStream.close();
                }

            } catch (Exception io) {
                System.err.println("PDF Export failed " + io);
            }


        } catch (Exception e) {
            System.err.println("Export PDF failed " + e);
        }
    }
}