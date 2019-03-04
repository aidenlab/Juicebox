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

package juicebox.windowui.layers;

import com.jidesoft.swing.JideBoxLayout;
import juicebox.DirectoryManager;
import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.MainWindow;
import juicebox.data.ChromosomeHandler;
import juicebox.data.HiCFileTools;
import juicebox.gui.SuperAdapter;
import juicebox.track.feature.AnnotationLayerHandler;
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
import java.util.List;
import java.util.*;

public class Load2DAnnotationsDialog extends JDialog implements TreeSelectionListener {

    private static final long serialVersionUID = 323844632613064L;
    private static DefaultMutableTreeNode customAddedFeatures = null;
    private final String[] searchHighlightColors = {"#ff0000", "#00ff00", "#0000ff", "#ff00ff", "#00ffff", "#ff9900", "#ff66ff", "#ffff00"};
    private final JTree tree;
    private final JButton openButton;
    private final JTextField fTextField;
    private final Map<String, MutableTreeNode> loadedAnnotationsMap = new HashMap<>();
    private File openAnnotationPath = DirectoryManager.getUserDirectory();

    public Load2DAnnotationsDialog(final LayersPanel layersPanel, final SuperAdapter superAdapter) {
        super(layersPanel, "Select 2D annotation file(s) to open");
        setModal(true);

        final ChromosomeHandler chromosomeHandler = superAdapter.getHiC().getChromosomeHandler();
        final MainWindow window = superAdapter.getMainWindow();

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
                                safeLoadAnnotationFiles(paths, layersPanel, superAdapter, chromosomeHandler);
                            } catch (Exception e) {
                                SuperAdapter.showMessageDialog("Unable to load file\n" + e.getLocalizedMessage());
                            }
                            Load2DAnnotationsDialog.this.setVisible(false);
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

        openButton = new JButton("Open");
        openButton.setEnabled(false);
        openButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                safeLoadAnnotationFiles(tree.getSelectionPaths(), layersPanel, superAdapter, chromosomeHandler);
                Load2DAnnotationsDialog.this.setVisible(false);
            }
        });

        JButton urlButton = new JButton("URL...");
        urlButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {

                String url = JOptionPane.showInputDialog("Enter URL: ");

                if (url != null && url.length() > 0) {
                    if (HiCFileTools.isDropboxURL(url)) {
                        url = HiCFileTools.cleanUpDropboxURL(url);
                    }
                    url = url.trim();
                    if (customAddedFeatures == null) {
                        customAddedFeatures = new DefaultMutableTreeNode(
                                new ItemInfo("Added 2D Features", ""), true);
                        top.add(customAddedFeatures);
                    }

                    if (loadedAnnotationsMap.containsKey(url)) {
                        if (HiCGlobals.guiIsCurrentlyActive) {
                            int dialogResult = JOptionPane.showConfirmDialog(window,
                                    "File is already loaded. Would you like to overwrite it?", "Warning",
                                    JOptionPane.YES_NO_OPTION);
                            if (dialogResult == JOptionPane.YES_OPTION) {
                                customAddedFeatures.remove(loadedAnnotationsMap.get(url));
                                loadedAnnotationsMap.remove(url);
                            } else {
                                return;
                            }
                        }
                    }

                    DefaultMutableTreeNode treeNode = new DefaultMutableTreeNode(
                            new ItemInfo(url, url), false);

                    loadedAnnotationsMap.put(url, treeNode);
                    customAddedFeatures.add(treeNode);
                    expandTree();
                    tree.updateUI();
                }
            }
        });
        urlButton.setPreferredSize(new Dimension((int) urlButton.getPreferredSize().getWidth(),
                (int) openButton.getPreferredSize().getHeight()));
        //setVisible(false);

        JButton cancelButton = new JButton("Cancel");
        cancelButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                Load2DAnnotationsDialog.this.setVisible(false);
            }
        });
        cancelButton.setPreferredSize(new Dimension((int) cancelButton.getPreferredSize().getWidth(),
                (int) openButton.getPreferredSize().getHeight()));

        buttonPanel.add(openButton);
        buttonPanel.add(urlButton);
        buttonPanel.add(cancelButton);

        add(buttonPanel, BorderLayout.SOUTH);
        Dimension minimumSize = new Dimension(700, 400);
        setMinimumSize(minimumSize);
        setLocation(100, 100);
        pack();

        final JLabel fLabel = new JLabel();
        fTextField = new JTextField();
        fLabel.setText("Filter:");
        fTextField.setToolTipText("Case Sensitive Search");
        fTextField.setPreferredSize(new Dimension((int) cancelButton.getPreferredSize().getWidth(), (int) openButton.getPreferredSize().getHeight()));
        buttonPanel.add(fLabel, JideBoxLayout.FIX);
        buttonPanel.add(fTextField, JideBoxLayout.VARY);

        //*********************SEARCH FILTER*******************************

        fTextField.addKeyListener(new KeyAdapter() {
            public void keyReleased(KeyEvent e) {
                collapseAll(tree);
                @SuppressWarnings("unchecked")
                Enumeration<DefaultMutableTreeNode> en = (Enumeration<DefaultMutableTreeNode>) top.preorderEnumeration();
                if (!fTextField.getText().isEmpty()) {
                    String[] searchStrings = fTextField.getText().split(",");
                    colorSearchStrings(searchStrings); //Coloring text that matches input
                    while (en.hasMoreElements()) {
                        DefaultMutableTreeNode leaf = en.nextElement();
                        String str = leaf.toString();
                        for (String term : searchStrings) {
                            if (str.contains(term)) {
                                expandToWantedNode(leaf);
                                break;
                            }
                        }
                    }
                }
            }
        });
    }

    public static TreePath getPath(TreeNode treeNode) {
        List<Object> nodes = new ArrayList<>();
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

    public void addLocalButtonActionPerformed(final Component parentComponent) {
        // Get the main window

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
                            new ItemInfo("Added 2D Features", ""), true);
                    root.add(customAddedFeatures);
                }

                String path = file.getAbsolutePath();
                openAnnotationPath = new File(path);

                if (loadedAnnotationsMap.containsKey(path)) {
                    if (HiCGlobals.guiIsCurrentlyActive) {
                        int dialogResult = JOptionPane.showConfirmDialog(parentComponent,
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
            }
            model.reload(root);
            expandTree();
        }
        Load2DAnnotationsDialog.this.setVisible(localFilesAdded);
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

    private void safeLoadAnnotationFiles(final TreePath[] paths, final LayersPanel layersPanel, final SuperAdapter superAdapter,
                                         final ChromosomeHandler chromosomeHandler) {
        Runnable runnable = new Runnable() {
            @Override
            public void run() {
                unsafeLoadAnnotationFiles(paths, layersPanel, superAdapter, chromosomeHandler);
            }
        };
        superAdapter.executeLongRunningTask(runnable, "load 2d annotation files");
    }

    private void unsafeLoadAnnotationFiles(TreePath[] paths, LayersPanel layersPanel, SuperAdapter superAdapter,
                                           ChromosomeHandler chromosomeHandler) {
        for (TreePath path : paths) {
            DefaultMutableTreeNode node = (DefaultMutableTreeNode) path.getLastPathComponent();
            if (node != null && node.isLeaf()) {
                ItemInfo info = (ItemInfo) node.getUserObject();
                try {
                    AnnotationLayerHandler handler = layersPanel.createNewLayerAndAddItToPanels(superAdapter, null);
                    handler.setLayerNameAndField(info.itemName);
                    handler.loadLoopList(info.itemURL, chromosomeHandler);
                } catch (Exception ee) {
                    System.err.println("Could not load selected annotation: " + info.itemName + " - " + info.itemURL);
                    SuperAdapter.showMessageDialog("Could not load loop selection: the loop list in" + info.itemName + "does not correspond to the genome");
                    customAddedFeatures.remove(loadedAnnotationsMap.get(info.itemURL)); //Todo needs to be a warning when trying to add annotations from a different genome
                    loadedAnnotationsMap.remove(path);
                }
            }
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
            openButton.setEnabled(true);
        } else {
            openButton.setEnabled(false);
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
