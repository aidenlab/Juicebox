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

package juicebox;

import juicebox.MainWindow;
import juicebox.state.SaveFileDialog;
import juicebox.track.LoadAction;
import juicebox.track.LoadEncodeAction;
import juicebox.track.feature.CustomAnnotation;
import juicebox.track.feature.Feature2DParser;
import juicebox.windowui.DumpDialog;
import juicebox.windowui.RecentMenu;
import juicebox.windowui.SaveAnnotationsDialog;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.util.Arrays;

/**
 * Created by Marie on 7/31/15.
 */
public class MainMenuBar extends JMenuBar{

    private MainWindow mainWindow;
    private HiC hic;
    private final int recentMapListMaxItems = 10;
    private final int recentLocationMaxItems = 20;
    private final String recentMapEntityNode = "hicMapRecent";
    private RecentMenu recentMapMenu;
    private JMenu annotationsMenu;

    MainMenuBar(MainWindow mainWindow, HiC hic) {
        super();
        this.mainWindow = mainWindow;
        this.hic = hic;
        createMenuBar();
    }

    private void createMenuBar() {
        //======== fileMenu ========
        JMenu fileMenu = new JMenu("File");
        fileMenu.setMnemonic('F');

        //---- Open Map ----
        JMenuItem openItem = new JMenuItem("Open...");
        openItem.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                mainWindow.loadFromListActionPerformed(false);
            }
        });
        //---- Open Control ----
        JMenuItem loadControlFromList = new JMenuItem();
        loadControlFromList.setText("Open Control...");
        loadControlFromList.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                mainWindow.loadFromListActionPerformed(true);
            }
        });
        //---- Open Recent ----
        recentMapMenu = new RecentMenu("Open Recent", recentMapListMaxItems, recentMapEntityNode) {
            private static final long serialVersionUID = 4202L;

            public void onSelectPosition(String mapPath) {
                String delimiter = "@@";
                String[] temp;
                temp = mapPath.split(delimiter);
                //initProperties();         // don't know why we're doing this here
                loadFromRecentActionPerformed((temp[1]), (temp[0]), false);
            }
        };
        recentMapMenu.setMnemonic('R');

        //---- Show Metrics ----
//        JMenuItem showStats = new JMenuItem("Show Dataset Metrics");
//        showStats.addActionListener(new ActionListener() {
//            @Override
//            public void actionPerformed(ActionEvent actionEvent) {
//                if (hic.getDataset() == null) {
//                    JOptionPane.showMessageDialog(MainWindow.this, "File must be loaded to show info", "Error", JOptionPane.ERROR_MESSAGE);
//                } else {
//                    new QCDialog(MainWindow.this, hic, MainWindow.this.getTitle() + " info");
//                }
//            }
//        });
//        fileMenu.add(showStats);


        //---- Export Image ----
//        JMenuItem saveToImage = new JMenuItem();
//        saveToImage.setText("Export Image...");
//        saveToImage.addActionListener(new ActionListener() {
//            public void actionPerformed(ActionEvent e) {
//                new SaveImageDialog(null, hic, hiCPanel);
//            }
//        });
//        fileMenu.add(saveToImage);

        // TODO: make this an export of the data on screen instead of a GUI for CLT
        if (!HiCGlobals.isRestricted) {
            JMenuItem dump = new JMenuItem("Export Data...");
            dump.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent actionEvent) {
                    if (hic.getDataset() == null) {
                        JOptionPane.showMessageDialog(mainWindow, "File must be loaded to show info", "Error", JOptionPane.ERROR_MESSAGE);
                    } else {
                        new DumpDialog(mainWindow, hic);
                    }
                }
            });
            fileMenu.add(dump);
        }

        //---- About ----
//        JMenuItem creditsMenu = new JMenuItem();
//        creditsMenu.setText("About");
//        creditsMenu.addActionListener(new ActionListener() {
//            public void actionPerformed(ActionEvent e) {
//                ImageIcon icon = new ImageIcon(getClass().getResource("/images/juicebox.png"));
//                JLabel iconLabel = new JLabel(icon);
//                JPanel iconPanel = new JPanel(new GridBagLayout());
//                iconPanel.add(iconLabel);
//
//                JPanel textPanel = new JPanel(new GridLayout(0, 1));
//                textPanel.add(new JLabel("<html><center>" +
//                        "<h2 style=\"margin-bottom:30px;\" class=\"header\">" +
//                        "Juicebox: Visualization software for Hi-C data" +
//                        "</h2>" +
//                        "</center>" +
//                        "<p>" +
//                        "Juicebox is Aiden Lab's software for visualizing data from proximity ligation experiments, such as Hi-C, 5C, and Chia-PET.<br>" +
//                        "Juicebox was created by Jim Robinson, Neva C. Durand, and Erez Aiden. Ongoing development work is carried out by Neva C. Durand,<br>" +
//                        "Muhammad Shamim, and Ido Machol.<br><br>" +
//                        "Copyright © 2014. Broad Institute and Aiden Lab" +
//                        "<br><br>" +
//                        "If you use Juicebox in your research, please cite:<br><br>" +
//                        "<strong>Suhas S.P. Rao*, Miriam H. Huntley*, Neva C. Durand, Elena K. Stamenova, Ivan D. Bochkov, James T. Robinson,<br>" +
//                        "Adrian L. Sanborn, Ido Machol, Arina D. Omer, Eric S. Lander, Erez Lieberman Aiden.<br>" +
//                        "\"A 3D Map of the Human Genome at Kilobase Resolution Reveals Principles of Chromatin Looping.\" <em>Cell</em> 159, 2014.</strong><br>" +
//                        "* contributed equally" +
//                        "</p></html>"));
//
//                JPanel mainPanel = new JPanel(new BorderLayout());
//                mainPanel.add(textPanel);
//                mainPanel.add(iconPanel, BorderLayout.WEST);
//
//                JOptionPane.showMessageDialog(null, mainPanel, "About", JOptionPane.PLAIN_MESSAGE);//INFORMATION_MESSAGE
//            }
//        });
//        fileMenu.add(creditsMenu);

        //---- exit ----
        JMenuItem exit = new JMenuItem();
        exit.setText("Exit");
        exit.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                mainWindow.exitActionPerformed();
            }
        });

        // add menu items
        fileMenu.add(openItem);
        fileMenu.add(loadControlFromList);
        fileMenu.add(recentMapMenu);
        fileMenu.addSeparator();
        fileMenu.add(exit);

        //======== Annotations Menu ========
        annotationsMenu = new JMenu("Annotations");

        //---- Load Basic ----
        JMenuItem newLoadMI = new JMenuItem();
        newLoadMI.setAction(new LoadAction("Load Basic Annotations...", mainWindow, hic));
        annotationsMenu.add(newLoadMI);

        //---- Load ENCODE ----
        JMenuItem loadEncodeMI = new JMenuItem();
        loadEncodeMI.setAction(new LoadEncodeAction("Load ENCODE Tracks...", mainWindow, hic));
        annotationsMenu.add(loadEncodeMI);

//---- Hand Annotations ----
        final JMenu customAnnotationMenu = new JMenu("Hand Annotations");
        // Import

        // Export
        exportAnnotationsMI = new JMenuItem("Export...");
        exportAnnotationsMI.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                new SaveAnnotationsDialog(customAnnotations);
            }
        });

//        // -- export overlap
//        final JMenuItem exportOverlapMI = new JMenuItem("Export Overlap...");
//        exportOverlapMI.addActionListener(new ActionListener() {
//            @Override
//            public void actionPerformed(ActionEvent e) {
//                java.util.List<Feature2DList> loops = hic.getAllVisibleLoopLists();
//                if (loops.size() != 1)
//                    JOptionPane.showMessageDialog(MainWindow.this, "Please merge ONE loaded set of annotations at a time.", "Error", JOptionPane.ERROR_MESSAGE);
//                else
//                    new SaveAnnotationsDialog(customAnnotations, loops.get(0));
//            }
//        });

// -- merge visible
        final JMenuItem mergeVisibleMI = new JMenuItem("Merge Visible");
        mergeVisibleMI.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                customAnnotations = customAnnotationHandler.addVisibleLoops(customAnnotations);
            }
        });

        // Undo
        undoMenuItem = new JMenuItem("Undo Annotation");
        undoMenuItem.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                customAnnotationHandler.undo(customAnnotations);
                repaint();
            }
        });
        undoMenuItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_Z, 0));

        // Load Last
        loadLastMI = new JMenuItem("Load Last");
        loadLastMI.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                customAnnotations = new CustomAnnotation(Feature2DParser.parseLoopFile(temp.getAbsolutePath(),
                        hic.getChromosomes(), 0, 0, 0, true, null), "1");
                temp.delete();
                loadLastMI.setEnabled(false);
            }
        });

// Clear All
        final JMenuItem clearCurrentMI = new JMenuItem("Clear All");
        clearCurrentMI.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                int n = JOptionPane.showConfirmDialog(
                        MainWindow.getInstance(),
                        "Are you sure you want to clear all custom annotations?",
                        "Confirm",
                        JOptionPane.YES_NO_OPTION);

                if (n == JOptionPane.YES_OPTION) {
                    //TODO: do something with the saving... just update temp?
                    customAnnotations.clearAnnotations();
                    exportAnnotationsMI.setEnabled(false);
                    loadLastMI.setEnabled(false);
                    repaint();
                }
            }
        });

        //Add annotate menu items
        customAnnotationMenu.add(exportAnnotationsMI);
        //-customAnnotationMenu.add(exportOverlapMI);
        //-customAnnotationMenu.add(mergeVisibleMI);
        customAnnotationMenu.add(undoMenuItem);
        if (unsavedEdits) {
            customAnnotationMenu.add(loadLastMI);
            loadLastMI.setEnabled(true);
        }
        customAnnotationMenu.add(clearCurrentMI);

        exportAnnotationsMI.setEnabled(false);
        undoMenuItem.setEnabled(false);

        annotationsMenu.add(customAnnotationMenu);

//        final JMenuItem annotate = new JMenuItem("Annotate Mode");
//        customAnnotationMenu.add(annotate);
//
//        // Add peak annotations
//        // TODO: Semantic inconsistency between what user sees (loop) and back end (peak) -- same thing.
//        final JCheckBoxMenuItem annotatePeak = new JCheckBoxMenuItem("Loops");
//
//        annotatePeak.setSelected(false);
//        annotatePeak.addActionListener(new ActionListener() {
//            @Override
//            public void actionPerformed(ActionEvent e) {
//                customAnnotationHandler.doPeak();
//            }
//        });
//        annotatePeak.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_L, 0));
//        annotate.add(annotatePeak);
//
//        // Add domain annotations
//        final JCheckBoxMenuItem annotateDomain = new JCheckBoxMenuItem("Domains");
//
//        annotateDomain.setSelected(false);
//        annotateDomain.addActionListener(new ActionListener() {
//            @Override
//            public void actionPerformed(ActionEvent e) {
//                customAnnotationHandler.doDomain();
//            }
//        });
//        annotateDomain.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_D, 0));
//        annotate.add(annotateDomain);
//
//        // Add generic annotations
//        final JCheckBoxMenuItem annotateGeneric = new JCheckBoxMenuItem("Generic feature");
//
//        annotateGeneric.setSelected(false);
//        annotateGeneric.addActionListener(new ActionListener() {
//            @Override
//            public void actionPerformed(ActionEvent e) {
//                customAnnotationHandler.doGeneric();
//            }
//        });
//        annotateDomain.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_F, 0));
//        annotate.add(annotateDomain);

        final JCheckBoxMenuItem showLoopsItem = new JCheckBoxMenuItem("Show 2D Annotations");

        showLoopsItem.setSelected(true);
        showLoopsItem.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                hic.setShowLoops(showLoopsItem.isSelected());
                repaint();
            }
        });
        showLoopsItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_F2, 0));

        annotationsMenu.add(showLoopsItem);

        final JCheckBoxMenuItem showCustomLoopsItem = new JCheckBoxMenuItem("Show Custom Annotations");

        showCustomLoopsItem.setSelected(true);
        showCustomLoopsItem.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                customAnnotations.setShowCustom(showCustomLoopsItem.isSelected());
                repaint();
            }
        });
        showCustomLoopsItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_F3, 0));

        annotationsMenu.add(showCustomLoopsItem);
        // meh

        annotationsMenu.setEnabled(false);

        JMenuItem loadFromURLItem = new JMenuItem("Load Annotation from URL...");
        loadFromURLItem.addActionListener(new AbstractAction() {

            private static final long serialVersionUID = 4203L;

            @Override
            public void actionPerformed(ActionEvent e) {
                if (hic.getDataset() == null) {
                    JOptionPane.showMessageDialog(mainWindow, "HiC file must be loaded to load tracks", "Error", JOptionPane.ERROR_MESSAGE);
                    return;
                }

                String url = JOptionPane.showInputDialog("Enter URL: ");
                if (url != null) {
                    hic.loadTrack(url);

                }

            }
        });

        JMenu bookmarksMenu = new JMenu("Bookmarks");
        //---- Save location ----
        saveLocationList = new JMenuItem();
        saveLocationList.setText("Save current location");
        saveLocationList.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                //code to add a recent location to the menu
                String stateString = hic.saveState();
                String stateDescriptionString = hic.getDefaultLocationDescription();
                String stateDescription = JOptionPane.showInputDialog(mainWindow,
                        "Enter description for saved location:", stateDescriptionString);
                if (null != stateDescription) {
                    getRecentStateMenu().addEntry(stateDescription + "@@" + stateString, true);
                }
            }
        });
        saveLocationList.setEnabled(false);
        bookmarksMenu.add(saveLocationList);
        //---Save State test-----
        saveStateForReload = new JMenuItem();
        saveStateForReload.setText("Save current state");
        saveStateForReload.addActionListener(new ActionListener() {

            public void actionPerformed(ActionEvent e) {
                //code to add a recent location to the menu
                String stateDescriptionString = hic.getDefaultLocationDescription();
                String stateDescription = JOptionPane.showInputDialog(mainWindow,
                        "Enter description for saved state:", stateDescriptionString);
                if (null != stateDescription) {
                    getPrevousStateMenu().addEntry(stateDescription, true);
                }
                hic.storeStateID();
                try {
                    hic.writeState();
                    hic.writeStateForXML();
                } catch (Exception e1) {
                    e1.printStackTrace();
                }
            }
        });

        saveStateForReload.setEnabled(true);
        bookmarksMenu.add(saveStateForReload);

        recentLocationMenu = new RecentMenu("Restore saved location", recentLocationMaxItems, recentLocationEntityNode) {

            private static final long serialVersionUID = 4204L;

            public void onSelectPosition(String mapPath) {
                String delimiter = "@@";
                String[] temp;
                temp = mapPath.split(delimiter);
                hic.restoreState(temp[1]);//temp[1]
                setNormalizationDisplayState();
            }
        };
        recentLocationMenu.setMnemonic('S');
        recentLocationMenu.setEnabled(false);
        bookmarksMenu.add(recentLocationMenu);

        previousStates = new RecentMenu("Restore previous states", recentLocationMaxItems, recentStateEntityNode) {

            private static final long serialVersionUID = 4205L;

            public void onSelectPosition(String mapPath) {
                hic.getMapPath(mapPath);
                hic.clearTracksForReloadState();
                hic.reloadPreviousState(hic.currentStates); //TODO use XML file instead
                hic.readXML(mapPath);
                updateThumbnail();
                previousStates.setSelected(true);
            }
        };
        previousStates.setEnabled(true);
        bookmarksMenu.add(previousStates);

        //---Export Menu-----
        JMenu shareMenu = new JMenu("Share States");

        //---Export Maps----
        exportMapAsFile = new JMenuItem();
        exportMapAsFile.setText("Export Saved States");
        exportMapAsFile.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                new SaveFileDialog(fileForExport, previousStates.getItemCount());
            }
        });


        //---Import Maps----
        /*importMapAsFile = new JMenuItem();
        importMapAsFile.setText("Import State From File");
        importMapAsFile.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                new ImportFileDialog(fileForExport, MainWindow.getInstance());
            }
        });*/


        //Add menu items
        shareMenu.add(exportMapAsFile);
        //shareMenu.add(importMapAsFile);

        this.add(fileMenu);
        this.add(annotationsMenu);
        this.add(bookmarksMenu);
        this.add(shareMenu);
    }

    public void enableAnnotations(){
        annotationsMenu.setEnabled(true);
    }

    private void loadFromRecentActionPerformed(String url, String title, boolean control) {
        if (url != null) {
            recentMapMenu.addEntry(title.trim() + "@@" + url, true);
            mainWindow.safeLoad(Arrays.asList(url), control, title);
        }
    }
}
