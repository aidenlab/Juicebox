/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2017 Broad Institute, Aiden Lab
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

package juicebox.gui;

import juicebox.DirectoryManager;
import juicebox.HiCGlobals;
import juicebox.ProcessHelper;
import juicebox.mapcolorui.Feature2DHandler;
import juicebox.state.SaveFileDialog;
import juicebox.tools.dev.Private;
import juicebox.windowui.HiCRulerPanel;
import juicebox.windowui.LoadAssemblyAnnotationsDialog;
import juicebox.windowui.RecentMenu;
import juicebox.windowui.SaveAssemblyDialog;
import org.apache.log4j.Logger;
import org.broad.igv.ui.util.MessageUtils;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.IOException;

/**
 * Created by muhammadsaadshamim on 8/4/15.
 */
public class MainMenuBar {
    private static final int recentMapListMaxItems = 10;
    private static final int recentLocationMaxItems = 20;
    private static final String recentMapEntityNode = "hicMapRecent";
    private static final String recentLocationEntityNode = "hicLocationRecent";
    private static final String recentStateEntityNode = "hicStateRecent";
    private static final Logger log = Logger.getLogger(MainMenuBar.class);

    private static JMenuItem loadLastMI;
    private static RecentMenu recentMapMenu, recentControlMapMenu;
    private static RecentMenu recentLocationMenu;
    private static JMenuItem saveLocationList;
    private static JMenuItem saveStateForReload;
    private static RecentMenu previousStates;
    private static JMenuItem exportSavedStateMenuItem;
    private static JMenuItem importMapAsFile;
    private static JMenuItem slideShow;
    private static JMenuItem showStats, showControlStats;
    private static File temp;
    private static boolean unsavedEdits;
    private static JMenu annotationsMenu;
    private static JMenu assemblyMenu;
    private static JMenuItem exportAssembly;
    private static JMenuItem resetAssembly;
    private static JCheckBoxMenuItem enableAssembly;
    private static JMenuItem setScale;
    private static JMenuItem importModifiedAssembly;

    private final JCheckBoxMenuItem layersItem = new JCheckBoxMenuItem("Show Annotation Panel");
    // created separately because it will be enabled after an initial map is loaded
    private final JMenuItem loadControlFromList = new JMenuItem();
    private File currentStates = new File("testStates");


    public boolean unsavedEditsExist() {
        String tempPath = "/unsaved-hiC-annotations1";
        temp = new File(DirectoryManager.getHiCDirectory(), tempPath + ".txt");
        unsavedEdits = temp.exists();
        return unsavedEdits;
    }

    public void addRecentMapMenuEntry(String title, boolean status) {
        recentMapMenu.addEntry(title, status);
        recentControlMapMenu.addEntry(title, status);
    }

    public void addRecentStateMenuEntry(String title, boolean status) {
        recentLocationMenu.addEntry(title, status);
    }



    public JMenuBar createMenuBar(final SuperAdapter superAdapter) {

        JMenuBar menuBar = new JMenuBar();

        //======== fileMenu ========
        JMenu fileMenu = new JMenu("File");
        fileMenu.setMnemonic('F');

        JMenuItem newWindow = new JMenuItem("New Window");
        newWindow.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                ProcessHelper p = new ProcessHelper();
                try {
                    p.startNewJavaProcess();
                } catch (IOException error) {
                    superAdapter.launchGenericMessageDialog(error.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
                }
            }
        });

        fileMenu.add(newWindow);

        //---- openMenuItem ----

        // create control first because it is enabled by regular open
        loadControlFromList.setText("Open as Control...");
        loadControlFromList.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                superAdapter.loadFromListActionPerformed(true);
            }
        });
        loadControlFromList.setEnabled(false);

        JMenuItem openItem = new JMenuItem("Open...");
        openItem.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                superAdapter.loadFromListActionPerformed(false);
            }
        });
        fileMenu.add(openItem);
        fileMenu.add(loadControlFromList);

        recentMapMenu = new RecentMenu("Open Recent", recentMapListMaxItems, recentMapEntityNode, HiCGlobals.menuType.MAP) {

            private static final long serialVersionUID = 4202L;

            public void onSelectPosition(String mapPath) {
                String delimiter = "@@";
                String[] temp;
                temp = mapPath.split(delimiter);
//                initProperties();         // don't know why we're doing this here
                superAdapter.loadFromRecentActionPerformed((temp[1]), (temp[0]), false);
            }
        };
        recentMapMenu.setMnemonic('R');

        fileMenu.add(recentMapMenu);

        recentControlMapMenu = new RecentMenu("Open Recent as Control", recentMapListMaxItems, recentMapEntityNode, HiCGlobals.menuType.MAP) {

            private static final long serialVersionUID = 42012L;

            public void onSelectPosition(String mapPath) {
                String delimiter = "@@";
                String[] temp;
                temp = mapPath.split(delimiter);
                //initProperties();         // don't know why we're doing this here
                superAdapter.loadFromRecentActionPerformed((temp[1]), (temp[0]), true);
            }
        };
        //recentControlMapMenu.setMnemonic('r');
        recentControlMapMenu.setEnabled(false);
        fileMenu.add(recentControlMapMenu);
        fileMenu.addSeparator();

        showStats = new JMenuItem("Show Dataset Metrics");
        showStats.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                superAdapter.showDataSetMetrics(false);
            }
        });
        showStats.setEnabled(false);

        showControlStats = new JMenuItem("Show Control Dataset Metrics");
        showControlStats.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                superAdapter.showDataSetMetrics(true);
            }
        });
        showControlStats.setEnabled(false);


        fileMenu.add(showStats);
        fileMenu.add(showControlStats);
        fileMenu.addSeparator();


        // TODO: make this an export of the data on screen instead of a GUI for CLT
        if (!HiCGlobals.isRestricted) {
            JMenuItem dump = new JMenuItem("Export Data...");
            dump.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent actionEvent) {
                    superAdapter.exportDataLauncher();
                }
            });
            fileMenu.add(dump);
        }

        JMenuItem creditsMenu = new JMenuItem();
        creditsMenu.setText("About");
        creditsMenu.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                ImageIcon icon = new ImageIcon(getClass().getResource("/images/juicebox.png"));
                JLabel iconLabel = new JLabel(icon);
                JPanel iconPanel = new JPanel(new GridBagLayout());
                iconPanel.add(iconLabel);

                JPanel textPanel = new JPanel(new GridLayout(0, 1));
                textPanel.add(new JLabel("<html><center>" +
                        "<h3 style=\"margin-bottom:30px;\" class=\"header\">" +
                        "Juicebox: Visualization software for Hi-C data" +
                        "</h3>" +
                        "</center>" +
                        "<p>" +
                        "Juicebox is the Aiden Lab's software for visualizing data<br>"+
                        "from proximity ligation experiments, such as Hi-C.<br>" +
                        "Juicebox was created by Jim Robinson, Neva C. Durand,<br>"+
                        "and Erez Aiden. Ongoing development work is carried<br>" +
                        "out by " +
                        "Neva C. Durand, Muhammad S. Shamim, Ido <br>Machol, Zulkifl Gire, " +
                        "and Marie Hoeger.<br><br>" +
                        "Current version: " + HiCGlobals.versionNum + "<br>" +
                        "Copyright © 2014. Broad Institute and Aiden Lab" +
                        "<br><br>" +
                        "" +
                        "If you use Juicebox in your research, please cite:<br><br>" +
                        "" +
                        "<strong>Neva C. Durand*, James T. Robinson*, Muhammad S.<br>Shamim, " +
                        "Ido Machol, Jill P. Mesirov, Eric S. Lander, and<br>Erez Lieberman Aiden.<br>" +
                        " \"Juicebox provides a visualization system for Hi-C<br>contact maps " +
                        "with unlimited zoom.\" <em>Cell Systems</em><br>July 2016.</strong>" +
                        "<br><br>" +
                        "<strong>Suhas S.P. Rao*, Miriam H. Huntley*, Neva C. Durand, <br>" +
                        "Elena K. Stamenova, Ivan D. Bochkov, James T. Robinson,<br>" +
                        "Adrian L. Sanborn, Ido Machol, Arina D. Omer, Eric S.<br>Lander, " +
                        "Erez Lieberman Aiden. \"A 3D Map of the<br>Human Genome at Kilobase "+
                        "Resolution Reveals<br>Principles of Chromatin Looping.\" <em>Cell</em> 159, 2014.</strong><br>" +
                        "* contributed equally" +
                        "</p></html>"));

                JPanel mainPanel = new JPanel(new BorderLayout());
                mainPanel.add(textPanel);
                mainPanel.add(iconPanel, BorderLayout.WEST);

                JOptionPane.showMessageDialog(superAdapter.getMainWindow(), mainPanel, "About", JOptionPane.PLAIN_MESSAGE);//INFORMATION_MESSAGE
            }
        });
        fileMenu.add(creditsMenu);

        //---- exit ----
        JMenuItem exit = new JMenuItem();
        exit.setText("Exit");
        exit.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                superAdapter.exitActionPerformed();
            }
        });
        fileMenu.add(exit);

        // "Annotations" menu items
        annotationsMenu = new JMenu("Annotations");
        annotationsMenu.setEnabled(false);

        layersItem.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (layersItem.isSelected()) {
                    superAdapter.setLayersPanelVisible(true);
                } else {
                    superAdapter.setLayersPanelVisible(false);
                }

            }
        });
        annotationsMenu.add(layersItem);

        loadLastMI = new JMenuItem("Load Last Session's Hand Annotations ");
        loadLastMI.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                superAdapter.generateNewCustomAnnotation(temp);
                temp.delete();
                loadLastMI.setEnabled(false);
                superAdapter.getActiveLayerHandler().setExportAbility(true);
            }
        });
        if (unsavedEditsExist()) {
            loadLastMI.setEnabled(true);
            annotationsMenu.add(loadLastMI);
        }

        JMenu bookmarksMenu = new JMenu("Bookmarks");
        //---- Save location ----
        saveLocationList = new JMenuItem("Save current location");
        saveLocationList.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                //code to add a recent location to the menu
                String stateString = superAdapter.getLocationDescription();
                String stateDescription = superAdapter.getDescription("location");
                if (stateDescription != null && stateDescription.length() > 0) {
                    superAdapter.addRecentStateMenuEntry(stateDescription + "@@" + stateString, true);
                    recentLocationMenu.setEnabled(true);
                }
            }
        });
        bookmarksMenu.add(saveLocationList);
        saveLocationList.setEnabled(false);
        //---Save State test-----
        saveStateForReload = new JMenuItem();
        saveStateForReload.setText("Save current state");
        saveStateForReload.addActionListener(new ActionListener() {

            public void actionPerformed(ActionEvent e) {
                //code to add a recent location to the menu
                try {
                    String stateDescription = superAdapter.getDescription("state");
                    if (stateDescription != null && stateDescription.length() > 0) {
                        stateDescription = previousStates.checkForDuplicateNames(stateDescription);
                        if (stateDescription == null || stateDescription.length() < 0) {
                            return;
                        }
                        previousStates.addEntry(stateDescription, true);
                        superAdapter.addNewStateToXML(stateDescription);
                        previousStates.setEnabled(true);
                    }
                } catch (Exception e1) {
                    e1.printStackTrace();
                }
            }
        });

        saveStateForReload.setEnabled(false);
        bookmarksMenu.add(saveStateForReload);

        recentLocationMenu = new RecentMenu("Restore saved location", recentLocationMaxItems, recentLocationEntityNode, HiCGlobals.menuType.LOCATION) {

            private static final long serialVersionUID = 4204L;

            public void onSelectPosition(String mapPath) {
                String delimiter = "@@";
                String[] temp;
                temp = mapPath.split(delimiter);
                superAdapter.restoreLocation(temp[1]);
                superAdapter.setNormalizationDisplayState();

            }
        };
        recentLocationMenu.setMnemonic('S');
        recentLocationMenu.setEnabled(false);
        bookmarksMenu.add(recentLocationMenu);

        //---Export States----
        exportSavedStateMenuItem = new JMenuItem();
        exportSavedStateMenuItem.setText("Export Saved States");
        exportSavedStateMenuItem.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                new SaveFileDialog(HiCGlobals.xmlSavedStatesFile);
            }
        });

        // restore recent saved states
        previousStates = new RecentMenu("Restore previous states", recentLocationMaxItems, recentStateEntityNode, HiCGlobals.menuType.STATE) {

            private static final long serialVersionUID = 4205L;

            public void onSelectPosition(String mapPath) {
                superAdapter.launchLoadStateFromXML(mapPath);
            }

            @Override
            public void setEnabled(boolean b) {
                super.setEnabled(b);
                exportSavedStateMenuItem.setEnabled(b);
            }
        };

        bookmarksMenu.add(previousStates);

        //---Import States----
        importMapAsFile = new JMenuItem();
        importMapAsFile.setText("Import State From File");
        importMapAsFile.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                superAdapter.launchImportState(HiCGlobals.xmlSavedStatesFile);
                importMapAsFile.setSelected(true);
            }
        });


        //---Slideshow----
        slideShow = new JMenuItem();
        slideShow.setText("View Slideshow");
        slideShow.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                superAdapter.launchSlideShow();
                HiCGlobals.slideshowEnabled = true;
            }
        });
        //bookmarksMenu.add(slideShow);

        bookmarksMenu.addSeparator();
        bookmarksMenu.add(exportSavedStateMenuItem);
        bookmarksMenu.add(importMapAsFile);

        //---View Menu-----
        JMenu viewMenu = new JMenu("View");

        JMenuItem addCustomChromosome = new JMenuItem("Make Custom Chromosome...");
        addCustomChromosome.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                superAdapter.createCustomChromosomes();
            }
        });
        viewMenu.add(addCustomChromosome);

        //---Axis Layout mode-----
        final JCheckBoxMenuItem axisEndpoint = new JCheckBoxMenuItem("Axis Endpoints Only");
        axisEndpoint.setSelected(HiCRulerPanel.getShowOnlyEndPts());
        axisEndpoint.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                HiCRulerPanel.setShowOnlyEndPts(axisEndpoint.isSelected());
                superAdapter.repaint();
            }
        });
        viewMenu.add(axisEndpoint);

        //---ShowChromosomeFig mode-----
        //drawLine, drawArc or draw polygon// draw round rect
        // fill Rect according to the chormsome location.
        final JCheckBoxMenuItem showChromosomeFig = new JCheckBoxMenuItem("Chromosome Context");
        showChromosomeFig.setSelected(HiCRulerPanel.getShowChromosomeFigure());
        showChromosomeFig.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                superAdapter.setShowChromosomeFig(showChromosomeFig.isSelected());
                superAdapter.repaint();
            }
        });
        viewMenu.add(showChromosomeFig);

        //---Grids mode-----
        // turn grids on/off
        final JCheckBoxMenuItem showGrids = new JCheckBoxMenuItem("Gridlines");
        showGrids.setSelected(superAdapter.getShowGridLines());
        showGrids.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                superAdapter.setShowGridLines(showGrids.isSelected());
                superAdapter.repaint();
            }
        });
        viewMenu.add(showGrids);

        viewMenu.addSeparator();

        //---Export Image Menu-----
        JMenuItem saveToPDF = new JMenuItem("Export PDF Figure...");
        saveToPDF.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                superAdapter.launchExportPDF();
            }
        });
        viewMenu.add(saveToPDF);

        JMenuItem saveToSVG = new JMenuItem("Export SVG Figure...");
        saveToSVG.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                superAdapter.launchExportSVG();
            }
        });
        viewMenu.add(saveToSVG);

        final JMenu devMenu = new JMenu("Dev");

        final JCheckBoxMenuItem displayTiles = new JCheckBoxMenuItem("Display Tiles");
        displayTiles.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                HiCGlobals.displayTiles = !HiCGlobals.displayTiles;
                superAdapter.getHeatmapPanel().repaint();
            }
        });
        displayTiles.setSelected(HiCGlobals.displayTiles);
        devMenu.add(displayTiles);

        JMenuItem editPearsonsColorItem = new JMenuItem("Edit Pearson's Color Scale");
        editPearsonsColorItem.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                superAdapter.launchPearsonColorScaleEditor();
            }
        });
        devMenu.add(editPearsonsColorItem);

        JMenuItem mapSubset = new JMenuItem("Select map subset...");
        mapSubset.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                Private.launchMapSubsetGUI(superAdapter);
            }
        });
        devMenu.add(mapSubset);

        final JTextField numSparse = new JTextField("" + Feature2DHandler.numberOfLoopsToFind);
        numSparse.setEnabled(true);
        numSparse.isEditable();
        numSparse.setToolTipText("Set how many 2D annotations to plot at a time.");

        final JButton updateSparseOptions = new JButton("Update");
        updateSparseOptions.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (numSparse.getText().length() > 0) {
                    Feature2DHandler.numberOfLoopsToFind = Integer.parseInt(numSparse.getText());
                }
            }
        });
        updateSparseOptions.setToolTipText("Set how many 2D annotations to plot at a time.");

        final JPanel sparseOptions = new JPanel();
        sparseOptions.setLayout(new GridLayout(0, 2));
        sparseOptions.add(numSparse);
        sparseOptions.add(updateSparseOptions);
        sparseOptions.setToolTipText("Set how many 2D annotations to plot at a time.");

        devMenu.addSeparator();
        devMenu.add(sparseOptions);

        JMenuItem chrSubset = new JMenuItem("Select genome subset...");
        chrSubset.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {

            }
        });

        assemblyMenu = new JMenu("Assembly");
        assemblyMenu.setEnabled(false);

        enableAssembly = new JCheckBoxMenuItem("Enable edits");
        enableAssembly.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (enableAssembly.isSelected()) {
                    superAdapter.getHeatmapPanel().enableAssemblyEditing();
                } else {
                    superAdapter.getHeatmapPanel().disableAssemblyEditing();
                }
            }
        });

        resetAssembly = new JMenuItem("Reset assembly");

        resetAssembly.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                int option = JOptionPane.showConfirmDialog(null, "Are you sure you want to reset?", "warning", JOptionPane.YES_NO_OPTION);
                if (option == 0) { //The ISSUE is here
                    superAdapter.getAssemblyStateTracker().resetState();
                    superAdapter.refresh();
                    System.out.print("yes");
                }
            }
        });

        exportAssembly = new JMenuItem("Export assembly");
        exportAssembly.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                String mapName = SuperAdapter.getDatasetTitle();
                new SaveAssemblyDialog(superAdapter.getAssemblyStateTracker().getAssemblyHandler(), mapName.substring(0, mapName.lastIndexOf("."))); //find how to get HiC filename

            }
        });

        final JMenuItem importMapAssembly = new JMenuItem("Import Map assembly");
        importMapAssembly.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (superAdapter.getLayersPanel() == null) {
                    superAdapter.intializeLayersPanel();
                }
                LoadAssemblyAnnotationsDialog loadAssemblyDialog;
                loadAssemblyDialog = new LoadAssemblyAnnotationsDialog(superAdapter.getLayersPanel(), superAdapter, superAdapter.getLayersPanel().getLayerBoxGUI2DAnnotations());
                loadAssemblyDialog.addLocalButtonActionPerformed(superAdapter);
            }
        });

        importModifiedAssembly = new JMenuItem("Import Modified assembly");
        importModifiedAssembly.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (superAdapter.getLayersPanel() == null) {
                    superAdapter.intializeLayersPanel();
                }
                LoadAssemblyAnnotationsDialog loadAssemblyDialog;
                loadAssemblyDialog = new LoadAssemblyAnnotationsDialog(superAdapter.getLayersPanel(), superAdapter, superAdapter.getLayersPanel().getLayerBoxGUI2DAnnotations());
                loadAssemblyDialog.addLocalButtonActionPerformed(superAdapter);
            }
        });

        setScale = new JMenuItem("Set scale");
        setScale.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                double scale;
                String newURL = MessageUtils.showInputDialog("Specify a scale", Double.toString(HiCGlobals.hicMapScale));
                try {
                    scale = Double.parseDouble(newURL);
                    if (scale == 0.0) { //scale cannot be zero
                        scale = 1.0;
                    }
                    HiCGlobals.hicMapScale = scale;
                } catch (NumberFormatException t) {
                    JOptionPane.showMessageDialog(null, "Value must be an integer!");
                }
            }
        });

        boolean enabled;
        if (superAdapter.getAssemblyStateTracker() != null)
            enabled = superAdapter.getAssemblyStateTracker().getAssemblyHandler() != null;
        else
            enabled = false;

        exportAssembly.setEnabled(enabled);
        resetAssembly.setEnabled(enabled);
        enableAssembly.setEnabled(enabled);
        setScale.setEnabled(enabled);
        importModifiedAssembly.setEnabled(enabled);

        assemblyMenu.add(enableAssembly);
        assemblyMenu.add(resetAssembly);
        assemblyMenu.add(importMapAssembly);
        assemblyMenu.add(importModifiedAssembly);
        assemblyMenu.add(exportAssembly);
        assemblyMenu.add(setScale);

        menuBar.add(fileMenu);
        menuBar.add(annotationsMenu);
        menuBar.add(bookmarksMenu);
        menuBar.add(viewMenu);
        menuBar.add(assemblyMenu);
        menuBar.add(devMenu);
        return menuBar;
    }

    public RecentMenu getRecentLocationMenu() {
        return recentLocationMenu;
    }

    public void setEnableForAllElements(boolean status) {
        annotationsMenu.setEnabled(status);
        assemblyMenu.setEnabled(status);
        saveLocationList.setEnabled(status);
        saveStateForReload.setEnabled(status);
        saveLocationList.setEnabled(status);
    }

    public void enableAssemblyResetAndExport() {
        resetAssembly.setEnabled(true);
        exportAssembly.setEnabled(true);
        enableAssembly.setEnabled(true);
        setScale.setEnabled(true);
        importModifiedAssembly.setEnabled(true);
    }

    public void enableAssemblyEditsOnImport(SuperAdapter superAdapter) {
        enableAssembly.setState(true);
        superAdapter.getHeatmapPanel().enableAssemblyEditing();
    }

    public void updatePrevStateNameFromImport(String path) {
        previousStates.updateNamesFromImport(path);
    }

    public void updateMainMapHasBeenLoaded(boolean status) {
        loadControlFromList.setEnabled(status);
        recentControlMapMenu.setEnabled(status);
        // if a control map can be loaded, that means main is loaded and its stats can be viewed
        showStats.setEnabled(status);
    }

    public void updateControlMapHasBeenLoaded(boolean status) {
        showControlStats.setEnabled(status);
    }

    public void setAnnotationPanelMenuItemSelected(boolean status) {
        layersItem.setSelected(status);
    }
}