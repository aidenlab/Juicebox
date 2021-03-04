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

package juicebox.gui;

import juicebox.DirectoryManager;
import juicebox.HiCGlobals;
import juicebox.ProcessHelper;
import juicebox.assembly.AssemblyFileImporter;
import juicebox.assembly.IGVFeatureCopy;
import juicebox.mapcolorui.ColorScaleHandler;
import juicebox.mapcolorui.Feature2DHandler;
import juicebox.state.SaveFileDialog;
import juicebox.tools.dev.Private;
import juicebox.windowui.*;
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
public class MainMenuBar extends JMenuBar {

  private static final long serialVersionUID = 9000025;
  private static final int recentMapListMaxItems = 10;
  private static final int recentLocationMaxItems = 20;
  private static final String recentMapEntityNode = "hicMapRecent";
  private static final String recentLocationEntityNode = "hicLocationRecent";
  private static final String recentStateEntityNode = "hicStateRecent";

  //private static JMenuItem loadOldAnnotationsMI;
  private static RecentMenu recentMapMenu, recentControlMapMenu;
  private static RecentMenu recentLocationMenu;
  private static JMenuItem saveLocationList;
  private static JMenuItem saveStateForReload;
  private static RecentMenu previousStates;
  private static JMenuItem exportSavedStateMenuItem;
  private static JMenuItem importMapAsFile;
  private static JMenuItem slideShow;
  private static JMenuItem showStats, showControlStats;
  private static JMenuItem renameGenome;
  //private static JMenu annotationsMenu;
  private static JMenu viewMenu;
  private static JMenu bookmarksMenu;
  private static JMenu assemblyMenu;
  private static JMenu devMenu;
  private static JMenuItem exportAssembly;
  private static JMenuItem resetAssembly;
  private static JMenuItem exitAssembly;
  private static JCheckBoxMenuItem enableAssembly;
  private static JMenuItem setScale;
  private static JMenuItem importModifiedAssembly;

  private final JCheckBoxMenuItem layersItem = new JCheckBoxMenuItem("Show Annotation Panel");
  // created separately because it will be enabled after an initial map is loaded
  private final JMenuItem loadControlFromList = new JMenuItem();

  public MainMenuBar(SuperAdapter superAdapter) {
    createMenuBar(superAdapter);
  }

  public static void exitAssemblyMode() {
    resetAssembly.setEnabled(false);
    exportAssembly.setEnabled(false);
    //  setScale.setEnabled(false);

    importModifiedAssembly.setEnabled(false);
    exitAssembly.setEnabled(false);
  }

  public boolean unsavedEditsExist() {
    File unsavedSampleFile = new File(DirectoryManager.getHiCDirectory(), HiCGlobals.BACKUP_FILE_STEM + "0.bedpe");
    return unsavedSampleFile.exists();
  }

  public void addRecentMapMenuEntry(String title, boolean status) {
    recentMapMenu.addEntry(title, status);
    recentControlMapMenu.addEntry(title, status);
  }

  private void addRecentStateMenuEntry(String title, boolean status) {
    recentLocationMenu.addEntry(title, status);
  }

  private void createMenuBar(final SuperAdapter superAdapter) {
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

      private static final long serialVersionUID = 9000021;

      public void onSelectPosition(String mapPath) {
        String[] temp = encodeSafeDelimeterSplit(mapPath);
        superAdapter.loadFromRecentActionPerformed((temp[1]), (temp[0]), false);
      }
    };
    recentMapMenu.setMnemonic('R');

    fileMenu.add(recentMapMenu);

    recentControlMapMenu = new RecentMenu("Open Recent as Control", recentMapListMaxItems, recentMapEntityNode, HiCGlobals.menuType.MAP) {

      private static final long serialVersionUID = 9000022;

      public void onSelectPosition(String mapPath) {
        String[] temp = encodeSafeDelimeterSplit(mapPath);
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
    JMenuItem dump = new JMenuItem("Export Data...");
    dump.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent actionEvent) {
        superAdapter.exportDataLauncher();
      }
    });
    fileMenu.add(dump);

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
                "Juicebox with Assembly Tools</h3>" +
                "</center>" +
                "<p>" +
                "Juicebox is the Aiden Lab's software for visualizing data from proximity ligation experiments, " +
                "such as Hi-C. Juicebox was created by Jim Robinson, Neva C. Durand, and Erez Aiden.<br><br>" +
                "Ongoing development work is carried out by Muhammad S. Shamim, Neva Durand, Olga Dudchenko, " +
                "Suhas Rao, and other members of the Aiden Lab.<br><br>" +
                "Current version: " + HiCGlobals.versionNum + "<br>" +
                "Copyright © 2014-2021. Broad Institute and Aiden Lab" +
                "<br><br>" +
                "" +
                "If you use Juicebox or Assembly Tools in your research, please cite:<br><br>" +
                "" +
                "<strong>Neva C. Durand*, James T. Robinson*, et al. " +
                "\"Juicebox provides a visualization system for Hi-C contact maps " +
                "with unlimited zoom.\" <em>Cell Systems</em> 2016.</strong>" +
                "<br><br>" +
                "<strong>Olga Dudchenko, et al. " +
                "\"The Juicebox Assembly Tools module facilitates de novo assembly of " +
                "mammalian genomes with chromosome-length scaffolds for under $1000.\" " +
                "<em>Biorxiv</em> 2018.</strong>" +
                "<br><br>" +
                "<strong>Suhas S.P. Rao*, Miriam H. Huntley*, et al. \"A 3D Map of the Human Genome at Kilobase " +
                "Resolution Reveals Principles of Chromatin Looping.\" <em>Cell</em> 2014.</strong><br>" +
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

    bookmarksMenu = new JMenu("Bookmarks");
    //---- Save location ----
    saveLocationList = new JMenuItem("Save Current Location");
    saveLocationList.addActionListener(new ActionListener() {
      public void actionPerformed(ActionEvent e) {
        //code to add a recent location to the menu
        String stateString = superAdapter.getLocationDescription();
        String stateDescription = superAdapter.getDescription("location");
        if (stateDescription != null && stateDescription.length() > 0) {
            addRecentStateMenuEntry(stateDescription + RecentMenu.delimiter + stateString, true);
          recentLocationMenu.setEnabled(true);
        }
      }
    });
    bookmarksMenu.add(saveLocationList);
    saveLocationList.setEnabled(false);
    //---Save State test-----
    saveStateForReload = new JMenuItem();
    saveStateForReload.setText("Save Current State");
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
    //bookmarksMenu.add(saveStateForReload);

    recentLocationMenu = new RecentMenu("Restore Saved Location", recentLocationMaxItems, recentLocationEntityNode, HiCGlobals.menuType.LOCATION) {

      private static final long serialVersionUID = 9000023;

      public void onSelectPosition(String mapPath) {
        String[] temp = encodeSafeDelimeterSplit(mapPath);
        superAdapter.restoreLocation(temp[1]);
        superAdapter.setNormalizationDisplayState();

      }
    };
    recentLocationMenu.setMnemonic('S');
    recentLocationMenu.setEnabled(false);
    bookmarksMenu.add(recentLocationMenu);
    bookmarksMenu.setEnabled(false);

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
    previousStates = new RecentMenu("Restore Previous States", recentLocationMaxItems, recentStateEntityNode, HiCGlobals.menuType.STATE) {

      private static final long serialVersionUID = 9000024;

      public void onSelectPosition(String mapPath) {
        superAdapter.launchLoadStateFromXML(mapPath);
      }

      @Override
      public void setEnabled(boolean b) {
        super.setEnabled(b);
        exportSavedStateMenuItem.setEnabled(b);
      }
    };

    //bookmarksMenu.add(previousStates);

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

    // todo replace with a save state URL
    //bookmarksMenu.addSeparator();
    //bookmarksMenu.add(exportSavedStateMenuItem);
    //bookmarksMenu.add(importMapAsFile);

    //---View Menu-----
    viewMenu = new JMenu("View");

    layersItem.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        superAdapter.setLayersPanelVisible(layersItem.isSelected());

      }
    });
    viewMenu.add(layersItem);
    viewMenu.setEnabled(false);

    final JMenuItem colorItem = new JMenuItem("Change Heatmap Color");
    colorItem.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        JColorChooser colorChooser = new JColorChooser(ColorScaleHandler.HIC_MAP_COLOR);
        JDialog dialog = JColorChooser.createDialog(MainMenuBar.this, "Select Heatmap Color",
                true, colorChooser, null, null);
        dialog.setVisible(true);
        Color color = colorChooser.getColor();
        if (color != null) {
          ColorScaleHandler.HIC_MAP_COLOR = color;
          superAdapter.getMainViewPanel().resetAllColors();
          superAdapter.refresh();
        }
      }
    });
    viewMenu.add(colorItem);

    final JCheckBoxMenuItem darkulaMode = new JCheckBoxMenuItem("Darkula Mode");
    darkulaMode.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        HiCGlobals.isDarkulaModeEnabled = !HiCGlobals.isDarkulaModeEnabled;
        superAdapter.getMainViewPanel().resetAllColors();
        //superAdapter.safeClearAllMZDCache();
        superAdapter.refresh();
      }
    });
    darkulaMode.setSelected(HiCGlobals.isDarkulaModeEnabled);
    viewMenu.add(darkulaMode);

    final JCheckBoxMenuItem advancedViewsMode = new JCheckBoxMenuItem("Advanced Views Mode");
    advancedViewsMode.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        MatrixType.toggleAdvancedViews();
        //superAdapter.getMainViewPanel().resetAllColors();
        //superAdapter.refresh();
      }
    });
    advancedViewsMode.setSelected(MatrixType.getAdvancedViewEnabled());
    viewMenu.add(advancedViewsMode);

    JMenuItem addCustomChromosome = new JMenuItem("Make Custom Chromosome (from .bed)...");
    addCustomChromosome.addActionListener(new ActionListener() {
      public void actionPerformed(ActionEvent e) {
        superAdapter.createCustomChromosomesFromBED();
      }
    });

    JMenuItem addGWChromosome = new JMenuItem("Make Genomewide Chromosome");
    addGWChromosome.addActionListener(new ActionListener() {
      public void actionPerformed(ActionEvent e) {
        superAdapter.createGenomewideChromosomeFromChromDotSizes();
      }
    });

    if (HiCGlobals.isDevCustomChromosomesAllowedPublic) {
      //viewMenu.add(addGWChromosome);
      viewMenu.add(addCustomChromosome);
    }

    viewMenu.addSeparator();

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

    devMenu = new JMenu("Dev");
    devMenu.setEnabled(false);

    final JMenuItem addRainbowTrack = new JMenuItem("Add a rainbow track...");
    addRainbowTrack.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        superAdapter.getHiC().generateRainbowBed();
      }
    });
    if (HiCGlobals.isDevAssemblyToolsAllowedPublic) {
      devMenu.add(addRainbowTrack);
    }

    final JCheckBoxMenuItem skipSortInPhase = new JCheckBoxMenuItem("Skip variant sorting in phase mode");
    skipSortInPhase.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        HiCGlobals.noSortInPhasing = !HiCGlobals.noSortInPhasing;
        superAdapter.getHeatmapPanel().repaint();
      }
    });
    skipSortInPhase.setSelected(HiCGlobals.noSortInPhasing);
    if (HiCGlobals.isDevAssemblyToolsAllowedPublic) {
      devMenu.add(skipSortInPhase);
    }

    final JMenuItem addCustomNormsObs = new JMenuItem("Add Custom Norms to Observed...");
    addCustomNormsObs.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        superAdapter.safeLaunchImportNormalizations(false);
      }
    });

    final JMenuItem addCustomNormsCtrl = new JMenuItem("Add Custom Norms to Control...");
    addCustomNormsCtrl.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        superAdapter.safeLaunchImportNormalizations(true);
      }
    });

    final JMenuItem addResolutionToDatasets = new JMenuItem("Add Custom Resolution...");
    addResolutionToDatasets.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        superAdapter.safeLaunchCreateNewResolution();
      }
    });

    if (HiCGlobals.isDevAssemblyToolsAllowedPublic) {
      devMenu.add(addCustomNormsObs);
      devMenu.add(addCustomNormsCtrl);
      devMenu.add(addResolutionToDatasets);
    }

    final JCheckBoxMenuItem displayTiles = new JCheckBoxMenuItem("Display Tiles");
    displayTiles.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        HiCGlobals.displayTiles = !HiCGlobals.displayTiles;
        superAdapter.getHeatmapPanel().repaint();
      }
    });

    final JCheckBoxMenuItem hackLinearColorScale = new JCheckBoxMenuItem("Hack linear color scale");
    hackLinearColorScale.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        HiCGlobals.HACK_COLORSCALE_LINEAR = !HiCGlobals.HACK_COLORSCALE_LINEAR;
        superAdapter.getHeatmapPanel().repaint();
      }
    });

    final JCheckBoxMenuItem hackColorScale = new JCheckBoxMenuItem("Hack color scale");
    hackColorScale.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        HiCGlobals.HACK_COLORSCALE = !HiCGlobals.HACK_COLORSCALE;
        superAdapter.getHeatmapPanel().repaint();
      }
    });

    final JCheckBoxMenuItem hackColorScaleEqual = new JCheckBoxMenuItem("Hack color scale equally");
    hackColorScaleEqual.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        HiCGlobals.HACK_COLORSCALE_EQUAL = !HiCGlobals.HACK_COLORSCALE_EQUAL;
        superAdapter.getHeatmapPanel().repaint();
      }
    });

    displayTiles.setSelected(HiCGlobals.displayTiles);
    if (HiCGlobals.isDevAssemblyToolsAllowedPublic) {
      devMenu.add(displayTiles);
      devMenu.add(hackColorScaleEqual);
      devMenu.add(hackColorScale);
      devMenu.add(hackLinearColorScale);
    }

    final JCheckBoxMenuItem colorFeatures = new JCheckBoxMenuItem("Recolor 1D Annotations in Assembly Mode");
    colorFeatures.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        IGVFeatureCopy.invertColorFeaturesChk();
        repaint();
      }
    });
    colorFeatures.setSelected(IGVFeatureCopy.colorFeaturesChk);
    if (HiCGlobals.isDevAssemblyToolsAllowedPublic) {
      devMenu.add(colorFeatures);
    }

    // todo MSS and Santiago - is this to be deleted?
    final JCheckBoxMenuItem useAssemblyMatrix = new JCheckBoxMenuItem("Use Assembly Chromosome Matrix");
    useAssemblyMatrix.setEnabled(!SuperAdapter.assemblyModeCurrentlyActive);
    useAssemblyMatrix.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        MainViewPanel.invertAssemblyMatCheck();
        superAdapter.createAssemblyChromosome();
        AssemblyFileImporter assemblyFileImporter;
        assemblyFileImporter = new AssemblyFileImporter(superAdapter);
        assemblyFileImporter.importAssembly();
//        superAdapter.assemblyModeCurrentlyActive = true;
        System.out.println(assemblyFileImporter.getAssemblyScaffoldHandler().toString());
      }
    });

    useAssemblyMatrix.setSelected(HiCGlobals.isAssemblyMatCheck);
    if (HiCGlobals.isDevAssemblyToolsAllowedPublic) {
      devMenu.add(useAssemblyMatrix);
    }


        renameGenome = new JMenuItem("Rename genome...");
        renameGenome.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                String curr_genome = superAdapter.getHiC().getDataset().getGenomeId();
                String response = JOptionPane.showInputDialog("Current genome is " + curr_genome +
                        "\nEnter another genome name or press cancel to exit");
                if (response != null) {
                    superAdapter.getHiC().getDataset().setGenomeId(response);
                }
            }
        });
        renameGenome.setEnabled(false);
        fileMenu.add(renameGenome);
    fileMenu.addSeparator();

    JMenuItem editPearsonsColorItem = new JMenuItem("Edit Pearson's Color Scale");
    editPearsonsColorItem.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        superAdapter.launchPearsonColorScaleEditor();
      }
    });
    devMenu.add(editPearsonsColorItem);

    JMenuItem editPseudoCounts = new JMenuItem("Change Pseudocount");
    editPseudoCounts.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        superAdapter.launchSetPseudoCountEditor();
      }
    });
    devMenu.add(editPseudoCounts);

    JMenuItem mapSubset = new JMenuItem("Select Map Subset...");
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


    /**    Assembly Menu     **/
    assemblyMenu = new JMenu("Assembly");
    assemblyMenu.setEnabled(false);

    enableAssembly = new JCheckBoxMenuItem("Enable Edits");
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

    resetAssembly = new JMenuItem("Reset Assembly");

    resetAssembly.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        int option = JOptionPane.showConfirmDialog(null, "Are you sure you want to reset?", "warning", JOptionPane.YES_NO_OPTION);
        if (option == 0) { //The ISSUE is here
          superAdapter.getAssemblyStateTracker().resetState();
          superAdapter.refresh();
        }
      }
    });

    exitAssembly = new JMenuItem("Exit Assembly");
    exitAssembly.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        int option = JOptionPane.showConfirmDialog(null, "Are you sure you want to exit?", "warning", JOptionPane.YES_NO_OPTION);
        if (option == 0) {
          superAdapter.exitAssemblyMode();
        }
      }
    });

    exportAssembly = new JMenuItem("Export Assembly");
    exportAssembly.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        String mapName = SuperAdapter.getDatasetTitle();
        new SaveAssemblyDialog(superAdapter.getAssemblyStateTracker().getAssemblyHandler(), mapName.substring(0, mapName.lastIndexOf("."))); //find how to get HiC filename

      }
    });

    final JMenuItem importMapAssembly = new JMenuItem("Import Map Assembly");
    importMapAssembly.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        if (superAdapter.getLayersPanel() == null) {
          superAdapter.intializeLayersPanel();
        }
        new LoadAssemblyAnnotationsDialog(superAdapter);
      }
    });

    importModifiedAssembly = new JMenuItem("Import Modified Assembly");
    importModifiedAssembly.addActionListener(new ActionListener() {

      //TODO: add warning if changes are present


            @Override
            public void actionPerformed(ActionEvent e) {
                if (superAdapter.getLayersPanel() == null) {
                    superAdapter.intializeLayersPanel();
                }
                new LoadModifiedAssemblyAnnotationsDialog(superAdapter);
            }
        });

        setScale = new JMenuItem("Set Scale");
        setScale.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                double scale;
                String newScale = MessageUtils.showInputDialog("Specify a scale", Double.toString(HiCGlobals.hicMapScale));
                try {
                    scale = Double.parseDouble(newScale);
                    if (scale == 0.0) {  // scale cannot be zero
                        scale = 1.0;
                    }
                    HiCGlobals.hicMapScale = scale;

                    // Rescale resolution slider labels
                    superAdapter.getMainViewPanel().getResolutionSlider().reset();

                  // Rescale axis tick labels
                  superAdapter.getMainViewPanel().getRulerPanelX().repaint();
                  superAdapter.getMainViewPanel().getRulerPanelY().repaint();

                  // Rescale and redraw assembly annotations
                  if (superAdapter.getAssemblyStateTracker() != null) {
                    superAdapter.getAssemblyStateTracker().resetState();
                  }


                } catch (NumberFormatException t) {
                  JOptionPane.showMessageDialog(null, "Value must be an integer!");
                }

      }
    });

    boolean enabled = superAdapter.getAssemblyStateTracker() != null && superAdapter.getAssemblyStateTracker().getAssemblyHandler() != null;

    exportAssembly.setEnabled(enabled);
    resetAssembly.setEnabled(enabled);
    enableAssembly.setEnabled(enabled);
    setScale.setEnabled(superAdapter.getHiC() != null && !superAdapter.getHiC().isWholeGenome());
    importModifiedAssembly.setEnabled(enabled);
    exitAssembly.setEnabled(enabled);


    assemblyMenu.add(importMapAssembly);
    assemblyMenu.add(importModifiedAssembly);
    assemblyMenu.add(exportAssembly);
    assemblyMenu.add(resetAssembly);
    assemblyMenu.add(resetAssembly);
    setScale.setEnabled(true);
    assemblyMenu.add(setScale);
    assemblyMenu.add(exitAssembly);
    // assemblyMenu.add(enableAssembly);
    add(fileMenu);
    // add(annotationsMenu);
    add(viewMenu);
    add(bookmarksMenu);
    if (HiCGlobals.isDevAssemblyToolsAllowedPublic) {
      add(assemblyMenu);
    }
    add(devMenu);
  }

    public RecentMenu getRecentLocationMenu() {
    return recentLocationMenu;
  }

  public void setEnableForAllElements(boolean status) {
    //annotationsMenu.setEnabled(status);
    viewMenu.setEnabled(status);
    bookmarksMenu.setEnabled(status);
    assemblyMenu.setEnabled(status);
    saveLocationList.setEnabled(status);
    saveStateForReload.setEnabled(status);
    saveLocationList.setEnabled(status);
    devMenu.setEnabled(status);
  }

  public void setEnableAssemblyMenuOptions(boolean status) {
    resetAssembly.setEnabled(status);
    exportAssembly.setEnabled(status);
    enableAssembly.setEnabled(status);
    setScale.setEnabled(status);
    importModifiedAssembly.setEnabled(status);
    exitAssembly.setEnabled(status);
    devMenu.setEnabled(status);
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