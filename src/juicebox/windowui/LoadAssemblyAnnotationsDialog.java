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

import juicebox.DirectoryManager;
import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.MainWindow;
import juicebox.assembly.AssemblyFileImporter;
import juicebox.assembly.AssemblyStateTracker;
import juicebox.gui.SuperAdapter;
import juicebox.mapcolorui.FeatureRenderer;
import juicebox.track.feature.AnnotationLayer;
import juicebox.track.feature.AnnotationLayerHandler;
import juicebox.windowui.layers.LayersPanel;
import juicebox.windowui.layers.Load2DAnnotationsDialog;
import org.broad.igv.ui.util.FileDialogUtils;
import org.broad.igv.util.ResourceLocator;

import javax.swing.*;
import javax.swing.event.TreeSelectionEvent;
import javax.swing.event.TreeSelectionListener;
import javax.swing.tree.*;
import java.awt.*;
import java.awt.event.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;


/**
 * Created by nathanielmusial on 6/29/17.
 */

public class LoadAssemblyAnnotationsDialog extends JDialog implements TreeSelectionListener {

    private static final long serialVersionUID = 323844632613064L;
    private static DefaultMutableTreeNode customAddedFeatures = null;
    private final String[] searchHighlightColors = {"#ff0000", "#00ff00", "#0000ff", "#ff00ff", "#00ffff", "#ff9900", "#ff66ff", "#ffff00"};
    private final JTree tree;
    private final JButton openAssemblyButton;
    private final Map<String, MutableTreeNode> loadedAnnotationsMap = new HashMap<>();
    private File openAnnotationPath = DirectoryManager.getUserDirectory();
    private final ArrayList<String> mostRecentPaths = new ArrayList<>();

    public LoadAssemblyAnnotationsDialog(final SuperAdapter superAdapter) {
        super(superAdapter.getMainWindow(), "Select Assembly annotation file(s) to open");

        final LayersPanel layersPanel = superAdapter.getLayersPanel();

        //Create the nodes.
        final DefaultMutableTreeNode top =
                new DefaultMutableTreeNode(new ItemInfo("root", ""), true);

        createNodes(top, superAdapter.getHiC());

        //Create a tree that allows one selection at a time.
        tree = new JTree(top);
        tree.getSelectionModel().setSelectionMode(TreeSelectionModel.DISCONTIGUOUS_TREE_SELECTION);

        //Listen for when the selection changes.
        tree.addTreeSelectionListener(this);
        tree.setRootVisible(false);
        tree.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent mouseEvent) {
                TreePath selPath = tree.getPathForLocation(mouseEvent.getX(), mouseEvent.getY());
                if (selPath != null) {
                    if (mouseEvent.getClickCount() == 2) {
                        DefaultMutableTreeNode node = (DefaultMutableTreeNode) selPath.getLastPathComponent();
                        if (node != null && node.isLeaf()) {
                            TreePath[] paths = new TreePath[1];
                            paths[0] = selPath;
                            try {
                                safeLoadAssemblyFiles(paths, layersPanel, superAdapter);
                            } catch (Exception e) {
                                SuperAdapter.showMessageDialog("Unable to load file\n" + e.getLocalizedMessage());
                            }
                            LoadAssemblyAnnotationsDialog.this.setVisible(false);
                        }
                    }
                }
            }
        });
        //Create the scroll pane and add the tree to it.
        final JScrollPane treeView = new JScrollPane(tree);
        treeView.setPreferredSize(new Dimension(400, 400));
        JPanel centerPanel = new JPanel(new BorderLayout());
        centerPanel.add(treeView, BorderLayout.CENTER);
        add(centerPanel, BorderLayout.CENTER);

        JPanel buttonPanel = new JPanel();

        openAssemblyButton = new JButton("Open Assembly");
        openAssemblyButton.setEnabled(Boolean.FALSE);
        openAssemblyButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                mostRecentPaths.clear();
                safeLoadAssemblyFiles(tree.getSelectionPaths(), layersPanel, superAdapter);
                LoadAssemblyAnnotationsDialog.this.setVisible(false);
            }
        });


        setDefaultCloseOperation(JDialog.DO_NOTHING_ON_CLOSE);
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                closeWindow();
            }
        });

        JButton cancelButton = new JButton("Cancel");
        cancelButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                closeWindow();
            }
        });
        cancelButton.setPreferredSize(new Dimension((int) cancelButton.getPreferredSize().getWidth(),
                (int) openAssemblyButton.getPreferredSize().getHeight()));

        buttonPanel.add(openAssemblyButton);
        buttonPanel.add(cancelButton);

        add(buttonPanel, BorderLayout.SOUTH);
        Dimension minimumSize = new Dimension(700, 400);
        setMinimumSize(minimumSize);
        setLocation(100, 100);
        pack();

        addLocalButtonActionPerformed(superAdapter);
    }

    public static TreePath getPath(TreeNode treeNode) {
        java.util.List<Object> nodes = new ArrayList<>();
        if (treeNode != null) {
            nodes.add(treeNode);
            treeNode = treeNode.getParent();
            while (treeNode != null) {
                nodes.add(0, treeNode);
                treeNode = treeNode.getParent();
            }
        }

        return nodes.isEmpty() ? null : new TreePath(nodes.toArray());
    }

    private void closeWindow() {
        customAddedFeatures.removeFromParent();
        for (String path : mostRecentPaths) {
            customAddedFeatures.remove(loadedAnnotationsMap.get(path));
            loadedAnnotationsMap.remove(path);
        }
        mostRecentPaths.clear();
        loadedAnnotationsMap.remove(customAddedFeatures);
        LoadAssemblyAnnotationsDialog.this.setVisible(false);
    }

    private void addLocalButtonActionPerformed(final SuperAdapter superAdapter) {
        // Get the main window
        final MainWindow window = superAdapter.getMainWindow();

        DefaultTreeModel model = (DefaultTreeModel) tree.getModel();
        DefaultMutableTreeNode root = (DefaultMutableTreeNode) model.getRoot();

        Boolean localFilesAdded = Boolean.FALSE;

        File[] twoDfiles = FileDialogUtils.chooseMultiple("Choose 2D Annotation file", openAnnotationPath, null);

        if (twoDfiles != null && twoDfiles.length > 0) {
            for (File file : twoDfiles) {

                if (file == null || !file.exists()) continue;

                localFilesAdded = Boolean.TRUE;

                if (customAddedFeatures == null) {
                    customAddedFeatures = new DefaultMutableTreeNode(
                            new ItemInfo("Added Assembly Files", ""), true);
                    root.add(customAddedFeatures);
                }

                String path = file.getAbsolutePath();
                openAnnotationPath = new File(path);

                if (loadedAnnotationsMap.containsKey(path)) {
                    if (HiCGlobals.guiIsCurrentlyActive) {
                        int dialogResult = JOptionPane.showConfirmDialog(window,
                                file.getName() + " is already loaded. Would you like to overwrite it?", "Warning",
                                JOptionPane.YES_NO_OPTION);
                        if (dialogResult == JOptionPane.YES_OPTION) {
                            customAddedFeatures.remove(loadedAnnotationsMap.get(path));
                            loadedAnnotationsMap.remove(path);
                        } else {
                            continue;
                        }
                    }
                }

                DefaultMutableTreeNode treeNode = new DefaultMutableTreeNode(
                        new ItemInfo(file.getName(), path), false);

                loadedAnnotationsMap.put(path, treeNode);
                customAddedFeatures.add(treeNode);
                mostRecentPaths.add(path);
            }
            model.reload(root);
            expandTree();
        }
        LoadAssemblyAnnotationsDialog.this.setVisible(localFilesAdded);
    }

    private void expandTree() {
        TreeNode root = (TreeNode) tree.getModel().getRoot();
        TreePath rootPath = new TreePath(root);
        TreeNode node = (TreeNode) rootPath.getLastPathComponent();
        for (Enumeration<?> e = node.children(); e.hasMoreElements(); ) {
            TreePath childPath = rootPath.pathByAddingChild(e.nextElement());
            if (!tree.isExpanded(childPath)) {
                tree.expandPath(childPath);
            }
        }
        if (!tree.isExpanded(rootPath)) {
            tree.expandPath(rootPath);
        }
    }


    private void safeLoadAssemblyFiles(final TreePath[] paths, final LayersPanel layersPanel, final SuperAdapter superAdapter) {
        Runnable runnable = new Runnable() {
            @Override
            public void run() {
                unsafeLoadAssemblyFiles(paths, layersPanel, superAdapter);
            }
        };
        superAdapter.executeLongRunningTask(runnable, "load 2d annotation files");
    }

    private void unsafeLoadAssemblyFiles(TreePath[] paths, LayersPanel layersPanel, SuperAdapter superAdapter) {
        // two-file format
        String cpropsPath = null;
        String asmPath = null;
        // single-file format
        String assemblyPath = null;

        for (TreePath path : paths) {
            DefaultMutableTreeNode node = (DefaultMutableTreeNode) path.getLastPathComponent();
            if (node != null && node.isLeaf()) {
                ItemInfo info = (ItemInfo) node.getUserObject();
                if (info.itemURL.endsWith("assembly")) {
                    assemblyPath = info.itemURL;
                    SuperAdapter.setDatasetTitle(assemblyPath);
                } else if (info.itemURL.endsWith("cprops")) {
                    cpropsPath = info.itemURL;
                } else if (info.itemURL.endsWith("asm")) {
                    asmPath = info.itemURL;
                    SuperAdapter.setDatasetTitle(asmPath);
                } else {
                    JOptionPane.showMessageDialog(layersPanel, "Unable to load invalid file!",
                            "Error Message", JOptionPane.ERROR_MESSAGE);
                    return;
                }
            }
        }

        if ((asmPath != null && cpropsPath != null) || assemblyPath != null) {

            try {
                if (superAdapter.getAssemblyStateTracker() != null) {
                    superAdapter.getAssemblyStateTracker().resetState();
                    superAdapter.refresh();
                }

                AssemblyFileImporter assemblyFileImporter;
                if (assemblyPath != null) {
                    assemblyFileImporter = new AssemblyFileImporter(assemblyPath, false);
                } else {
                    assemblyFileImporter = new AssemblyFileImporter(cpropsPath, asmPath, false);
                }

                //temp layer to allow deleting of other layers
                layersPanel.createNewLayerAndAddItToPanels(superAdapter, null);
                if (superAdapter.getAssemblyLayerHandlers() != null) {
                    for (AnnotationLayerHandler annotationLayerHandler : superAdapter.getAssemblyLayerHandlers())
                        superAdapter.removeLayer(annotationLayerHandler);
                }
                assemblyFileImporter.importAssembly();
                // Rescale resolution slider labels
                superAdapter.getMainViewPanel().getResolutionSlider().reset();

                // Rescale axis tick labels
                superAdapter.getMainViewPanel().getRulerPanelX().repaint();
                superAdapter.getMainViewPanel().getRulerPanelY().repaint();

                // read in scaffold data
                AnnotationLayer scaffoldLayer = new AnnotationLayer(
                        assemblyFileImporter.getAssemblyScaffoldHandler().getScaffoldFeature2DHandler(), AnnotationLayer.LayerType.SCAFFOLD);
                AnnotationLayerHandler scaffoldLayerHandler = layersPanel.createNewLayerAndAddItToPanels(superAdapter, null);
                scaffoldLayerHandler.setProperties(scaffoldLayer, "Scaf", Color.green);

                // read in superscaffold data
                AnnotationLayer superscaffoldLayer = new AnnotationLayer(
                        assemblyFileImporter.getAssemblyScaffoldHandler().getSuperscaffoldFeature2DHandler(), AnnotationLayer.LayerType.SUPERSCAFFOLD);
                AnnotationLayerHandler superscaffoldLayerHandler = layersPanel.createNewLayerAndAddItToPanels(superAdapter, null);
                superscaffoldLayerHandler.setProperties(superscaffoldLayer, "Chr", Color.blue);

                AnnotationLayerHandler editLayerHandler = layersPanel.createNewLayerAndAddItToPanels(superAdapter, null);
                editLayerHandler.setColorOfAllAnnotations(Color.yellow);
                editLayerHandler.setLayerNameAndField("Edit");
                editLayerHandler.setLineStyle(FeatureRenderer.LineStyle.DASHED);
                editLayerHandler.getAnnotationLayer().setLayerType(AnnotationLayer.LayerType.EDIT);

                AssemblyStateTracker assemblyStateTracker = new AssemblyStateTracker(assemblyFileImporter.getAssemblyScaffoldHandler(), superAdapter);
                superAdapter.setAssemblyStateTracker(assemblyStateTracker);

                //superAdapter.getLayersPanel().updateAssemblyAnnotationsPanel(superAdapter);
                superAdapter.getMainMenuBar().enableAssemblyMenuOptions();
                superAdapter.getMainMenuBar().enableAssemblyEditsOnImport(superAdapter);
                for (AnnotationLayerHandler annotationLayerHandler : superAdapter.getAllLayers()) {
                    if (annotationLayerHandler.getAnnotationLayerType() != AnnotationLayer.LayerType.EDIT
                            && annotationLayerHandler.getAnnotationLayer().getFeatureList().getNumTotalFeatures() == 0)
                        superAdapter.removeLayer(annotationLayerHandler);
                    superAdapter.setActiveLayerHandler(scaffoldLayerHandler);
                    superAdapter.getLayersPanel().updateBothLayersPanels(superAdapter);
                }

            } catch (Exception ee) {
//                System.err.println("Could not load selected annotation: " + info.itemName + " - " + info.itemURL);
//                SuperAdapter.showMessageDialog("Could not load loop selection: " + ee.getMessage());
                if (assemblyPath != null) customAddedFeatures.remove(loadedAnnotationsMap.get(assemblyPath));
                if (cpropsPath != null) customAddedFeatures.remove(loadedAnnotationsMap.get(cpropsPath));
                if (asmPath != null)
                    customAddedFeatures.remove(loadedAnnotationsMap.get(asmPath)); //Todo needs to be a warning when trying to add annotations from a different genomeloadedAnnotationsMap.remove(path);
            }
        } else {
            System.err.println("Invalid files...");
        }
    }

    private void expandToWantedNode(DefaultMutableTreeNode dNode) {
        if (dNode != null) {
            tree.setExpandsSelectedPaths(true);
            TreePath path = new TreePath(dNode.getPath());
            tree.scrollPathToVisible(path);
            tree.setSelectionPath(path);
        }
    }

    //Overriding in order to change text color
    private void colorSearchStrings(final String[] parts) {
        tree.setCellRenderer(new DefaultTreeCellRenderer() {

            private static final long serialVersionUID = 4231L;


            @Override
            public Component getTreeCellRendererComponent(JTree tree, Object value, boolean sel, boolean expanded,
                                                          boolean leaf, int row, boolean hasFocus) {
                String text = value.toString();
                for (int i = 0; i < Math.min(parts.length, searchHighlightColors.length); i++) {
                    text = text.replaceAll(parts[i], "<font color=\"" + searchHighlightColors[i] + "\">" + parts[i] + "</font>");
                }
                String html = "<html>" + text + "</html>";

                return super.getTreeCellRendererComponent(
                        tree, html, sel, expanded, leaf, row, hasFocus);
            }
        });
    }

    private boolean createNodes(DefaultMutableTreeNode top, HiC hic) {

        // Add dataset-specific 2d annotations
        DefaultMutableTreeNode subParent = new DefaultMutableTreeNode(new ItemInfo("Dataset-specific 2D Features"), true);
        ResourceLocator[] locators = {hic.getDataset().getPeaks(), hic.getDataset().getBlocks(), hic.getDataset().getSuperLoops()};
        String[] locatorName = {"Peaks", "Contact Domains", "ChrX Super Loops"};

        boolean datasetSpecificFeatureAdded = false;
        for (int i = 0; i < 3; i++) {
            if (locators[i] != null) {
                DefaultMutableTreeNode treeNode = new DefaultMutableTreeNode(new ItemInfo(locatorName[i], locators[i].getURLPath()), false);
                subParent.add(treeNode);
                datasetSpecificFeatureAdded = true;
            }
        }
        if (datasetSpecificFeatureAdded) top.add(subParent); // allow specific dataset features to be top-level

        // load remaining features from file
        DefaultMutableTreeNode parent = new DefaultMutableTreeNode(new ItemInfo("Chromatin Features"), true);
        top.add(parent);

        InputStream is = Load2DAnnotationsDialog.class.getResourceAsStream("annotations2d.txt");
        BufferedReader reader = new BufferedReader(new InputStreamReader(is), HiCGlobals.bufferSize);
        String nextLine;

        try {
            while ((nextLine = reader.readLine()) != null) {
                final String[] values = nextLine.split(";");
                if (values.length != 1 && values.length != 2) {
                    JOptionPane.showMessageDialog(this, "Improper features file", "Error", JOptionPane.ERROR_MESSAGE);
                    return false;
                }
                if (values.length == 1) {
                    subParent = new DefaultMutableTreeNode(new ItemInfo(values[0]), true);
                    parent.add(subParent);

                    //node = new DefaultMutableTreeNode(new ItemInfo(key, values[0], values[1]));
                } else {
                    DefaultMutableTreeNode node = new DefaultMutableTreeNode(new ItemInfo(values[0], values[1]), false);
                    subParent.add(node);
                }
            }
        } catch (Exception ignored) {

        }
        if (customAddedFeatures != null) {
            top.add(customAddedFeatures);
        }
        return true;
    }

    /**
     * Required by TreeSelectionListener interface.
     */
    public void valueChanged(TreeSelectionEvent e) {
        DefaultMutableTreeNode node = (DefaultMutableTreeNode)
                tree.getLastSelectedPathComponent();

        if (node == null) return;

        if (node.isLeaf()) {
            openAssemblyButton.setEnabled(true);
        } else {
            openAssemblyButton.setEnabled(false);
        }
    }

    private void collapseAll(JTree tree) {
        int row = tree.getRowCount() - 1;
        while (row >= 0) {
            tree.collapseRow(row);
            row--;
        }
    }

    private class ItemInfo {
        final String itemName;
        final String itemURL;

        ItemInfo(String itemName, String itemURL) {
            this.itemName = itemName.trim();
            this.itemURL = itemURL.trim();
        }

        ItemInfo(String itemName) {
            this.itemName = itemName;
            itemURL = null;
        }

        public String toString() {
            return itemName;
        }
    }
}
