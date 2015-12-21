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

import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.MainWindow;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class SaveImageDialog extends JFileChooser {
    private static final long serialVersionUID = -611947177404923808L;
    private JTextField width;
    private JTextField height;

    public SaveImageDialog(String saveImagePath, HiC hic, JPanel hiCPanel) {
        super();
        if (saveImagePath != null) {
            setSelectedFile(new File(saveImagePath));
        } else {
            String timeStamp = new SimpleDateFormat("yyyy.MM.dd.HH.mm.ss").format(new Date());
            setSelectedFile(new File(timeStamp + ".HiCImage.png"));
        }
        if (HiCGlobals.guiIsCurrentlyActive) {
            MainWindow mainWindow = MainWindow.getInstance();
            int actionDialog = showSaveDialog(mainWindow);
            if (actionDialog == JFileChooser.APPROVE_OPTION) {
                File file = getSelectedFile();
                //saveImagePath = file.getPath();
                if (file.exists()) {
                    actionDialog = JOptionPane.showConfirmDialog(MainWindow.getInstance(), "Replace existing file?");
                    if (actionDialog == JOptionPane.NO_OPTION || actionDialog == JOptionPane.CANCEL_OPTION)
                        return;
                }
                try {
                    int w = Integer.valueOf(width.getText());
                    int h = Integer.valueOf(height.getText());
                    saveImage(file, mainWindow, hic, hiCPanel, w, h);
                } catch (IOException error) {
                    JOptionPane.showMessageDialog(mainWindow, "Error while saving file:\n" + error, "Error",
                            JOptionPane.ERROR_MESSAGE);
                } catch (NumberFormatException error) {
                    JOptionPane.showMessageDialog(mainWindow, "Width and Height must be integers", "Error",
                            JOptionPane.ERROR_MESSAGE);
                }
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

    private void saveImage(File file, MainWindow mainWindow, HiC hic, final JPanel hiCPanel, final int w, final int h) throws IOException {

        // default if they give no format or invalid format
        String fmt = "jpg";
        int ind = file.getName().indexOf(".");
        if (ind != -1) {
            String ext = file.getName().substring(ind + 1);
            String[] strs = ImageIO.getWriterFormatNames();
            for (String aStr : strs)
                if (ext.equals(aStr))
                    fmt = ext;
        }
        BufferedImage image = (BufferedImage) MainWindow.getInstance().createImage(w, h);
        Graphics g = image.createGraphics();

        Dimension size = MainWindow.getInstance().getSize();

        if (w == mainWindow.getWidth() && h == MainWindow.getInstance().getHeight()) {
            hiCPanel.paint(g);
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

            hiCPanel.paint(g);
            mainWindow.setPreferredSize(prefSize);
            mainWindow.setMinimumSize(minSize);
            mainWindow.setSize(size);
            waitDialog.setVisible(false);
            waitDialog.dispose();
            mainWindow.setVisible(true);
        }

        ImageIO.write(image.getSubimage(0, 0, w, h), fmt, file);
        g.dispose();
    }

}
