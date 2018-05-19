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

import juicebox.gui.SuperAdapter;
import juicebox.windowui.DisabledGlassPane;

import javax.swing.*;
import javax.swing.border.Border;
import javax.swing.filechooser.FileSystemView;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.List;


/**
 * Created by muhammadsaadshamim on 8/4/16.
 */
public class AssemblyPlaybackPanel extends JDialog {

    public static final DisabledGlassPane disabledGlassPane = new DisabledGlassPane(Cursor.WAIT_CURSOR);
    private static final long serialVersionUID = 8124112892179L;
    boolean paused = false;
    int playBackSpeed = 1000;
    private String playLocation;
    private String saveLocation;
    private List<String> results;
    // private final JPanel assemblyPanel;


    public AssemblyPlaybackPanel(final SuperAdapter superAdapter) {
        super(superAdapter.getMainWindow(), "Assembly Tracking Panel");
        rootPane.setGlassPane(disabledGlassPane);

        Border padding = BorderFactory.createEmptyBorder(20, 20, 5, 20);

        add(generateAssemblyPanel(superAdapter));
        // if (assemblyPanel != null) assemblyPanel.setBorder(padding);

        setSize(300, 150);

        setPlayLocationAssemblyTracking(superAdapter);
        AssemblyOperationExecutor.loadAssemblyTracking(superAdapter, playLocation);

    }

    private JPanel generateAssemblyPanel(final SuperAdapter superAdapter) {


        final JPanel pane = new JPanel(new BorderLayout());
        JButton playButton, pauseButton;
        JPanel buttonPanel = new JPanel();
        playButton = new JButton("Play");
        playButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                paused = false;
                play(superAdapter);
            }
        });

        pauseButton = new JButton("Pause");
        pauseButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                paused = true;
            }
        });
        buttonPanel.add(playButton);
        buttonPanel.add(pauseButton);
        pane.add(buttonPanel);
        return pane;
    }


    public void setSaveLocationForAssemblyTracking() {

        JFileChooser jfc = new JFileChooser(FileSystemView.getFileSystemView().getHomeDirectory());
        jfc.setDialogTitle("Choose a directory to save your file: ");
        jfc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);

        int returnValue = jfc.showSaveDialog(null);
        if (returnValue == JFileChooser.APPROVE_OPTION) {
            if (jfc.getSelectedFile().isDirectory()) {
                System.out.println("You selected the directory: " + jfc.getSelectedFile());
                AssemblyOperationExecutor.enableAssemblyTracking(jfc.getSelectedFile().getPath());
            }
        }
    }


    public void setPlayLocationAssemblyTracking(SuperAdapter superAdapter) {
        JFileChooser jfc = new JFileChooser(FileSystemView.getFileSystemView().getHomeDirectory());
        jfc.setDialogTitle("Choose a directory to open: ");
        jfc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);

        int returnValue = jfc.showSaveDialog(null);
        if (returnValue == JFileChooser.APPROVE_OPTION) {
            if (jfc.getSelectedFile().isDirectory()) {
                System.out.println("You selected the directory: " + jfc.getSelectedFile());
                playLocation = jfc.getSelectedFile().getPath();
            }
        }
    }


    public void play(final SuperAdapter superAdapter) {
        final AssemblyStateTracker assemblyStateTracker = superAdapter.getAssemblyStateTracker();

        Thread thread = new Thread(
                new Runnable() {
                    public void run() {
                        while (superAdapter.getAssemblyStateTracker().checkRedo() && !paused) {


                            assemblyStateTracker.redo();
                            try {
                                Thread.sleep(playBackSpeed);
                            } catch (InterruptedException e) {
                                e.printStackTrace();
                            }
                        }
                    }
                }
        );
        thread.start();
    }

}
