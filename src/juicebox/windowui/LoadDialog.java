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

package juicebox.windowui;

import com.jidesoft.swing.JideBoxLayout;
import juicebox.DirectoryManager;
import juicebox.MainWindow;
import juicebox.data.HiCFileLoader;
import juicebox.gui.SuperAdapter;

import javax.swing.*;
import javax.swing.event.TreeSelectionEvent;
import javax.swing.event.TreeSelectionListener;
import javax.swing.tree.*;
import java.awt.*;
import java.awt.event.*;
import java.io.File;
import java.util.*;
import java.util.List;

public class LoadDialog extends JDialog implements TreeSelectionListener, ActionListener {

    private static final long serialVersionUID = 3238446384712613064L;
    public static File LAST_LOADED_HIC_FILE_PATH = DirectoryManager.getUserDirectory();
    private static boolean actionLock = false;
    private final boolean success;
    private final String[] searchHighlightColors = {"#ff0000", "#00ff00", "#0000ff", "#ff00ff", "#00ffff", "#ff9900", "#ff66ff", "#ffff00"};
    private final SuperAdapter superAdapter;
    private JTree tree;
    private JSplitButton openButton;
    private JMenuItem openButton30;
    private JButton cancelButton;
    private JButton localButton;
    private JButton urlButton;
    private JTextField fTextField;
    private boolean control;

    public LoadDialog(MainWindow mainWindow, Properties properties, SuperAdapter superAdapter) {
        super(mainWindow, "Select file(s) to open");

        this.superAdapter = superAdapter;

        //Create the nodes.
        final DefaultMutableTreeNode top =
                new DefaultMutableTreeNode(new ItemInfo("root", "root", ""));

        //System.out.println(properties);
        if (properties != null) {

            if (!createNodes(top, properties)) {
                dispose();
                success = false;
                return;
            }

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
                                loadFiles(paths, null);
                            }

                        }
                    }
                }
            });
            //Create the scroll pane and add the tree to it.
            JScrollPane treeView = new JScrollPane(tree);
            treeView.setPreferredSize(new Dimension(400, 400));
            JPanel centerPanel = new JPanel(new BorderLayout());
            centerPanel.add(treeView, BorderLayout.CENTER);
            add(centerPanel, BorderLayout.CENTER);
        } else {
            JLabel label = new JLabel("Can't find properties file; no online maps to load");
            label.setHorizontalAlignment(JLabel.CENTER);
            JPanel panel = new JPanel(new BorderLayout());
            panel.add(label, BorderLayout.CENTER);
            add(panel, BorderLayout.CENTER);
        }
        JPanel buttonPanel = new JPanel();

        openButton = createMAPQ0Button("Open (MAPQ > 0)");
        openButton.setEnabled(false);
        openButton30 = createMAPQ30Menu(openButton, "Open (MAPQ \u2265 30)");

        localButton = new JButton("Local...");
        localButton.addActionListener(this);
        localButton.setPreferredSize(new Dimension((int) localButton.getPreferredSize().getWidth(), (int) openButton.getPreferredSize().getHeight()));

        urlButton = new JButton("URL...");
        urlButton.addActionListener(this);
        urlButton.setPreferredSize(new Dimension((int) urlButton.getPreferredSize().getWidth(), (int) openButton.getPreferredSize().getHeight()));

        cancelButton = new JButton("Cancel");
        cancelButton.addActionListener(this);
        cancelButton.setPreferredSize(new Dimension((int) cancelButton.getPreferredSize().getWidth(), (int) openButton.getPreferredSize().getHeight()));

        buttonPanel.add(openButton);
        buttonPanel.add(localButton);
        buttonPanel.add(urlButton);
        buttonPanel.add(cancelButton);

        add(buttonPanel, BorderLayout.SOUTH);
        Dimension minimumSize = new Dimension(700, 400);
        setMinimumSize(minimumSize);
        setLocation(100, 100);
        pack();
        success = true;

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

    private JSplitButton createMAPQ0Button(String buttonText) {
        JSplitButton button = new JSplitButton(buttonText);
        button.addActionListener(this);
        return button;
    }

    private JMenuItem createMAPQ30Menu(JSplitButton button, String button30Text) {
        JMenuItem button30 = new JMenuItem(button30Text);
        button30.addActionListener(this);
        button30.setEnabled(false);
        JPopupMenu popupMenu = new JPopupMenu("MAPQ ≥ 30 Menu");
        popupMenu.add(button30);
        popupMenu.setEnabled(false);
        button.setComponentPopupMenu(popupMenu);
        return button30;
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

    public void setControl(boolean control) {
        this.control = control;
    }

    public boolean getSuccess() {
        return success;
    }

    private boolean createNodes(DefaultMutableTreeNode top, Properties properties) {
        // Enumeration<DefaultMutableTreeNode> enumeration = top.breadthFirstEnumeration();
        // TreeSet is sorted, so properties file is implemented in order
        TreeSet<String> keys = new TreeSet<>(properties.stringPropertyNames());
        HashMap<String, DefaultMutableTreeNode> hashMap = new HashMap<>();
        hashMap.put(((ItemInfo) top.getUserObject()).uid, top);
        //HashMap<String, DefaultMutableTreeNode> tempHash = new HashMap<String, DefaultMutableTreeNode>();
        //tempHash.put(((ItemInfo) top.getUserObject()).uid, top);

        for (String key : keys) {
            String value = properties.getProperty(key);
            DefaultMutableTreeNode node;
            final String[] values = value.split(",");
            if (values.length != 3 && values.length != 2) {
                JOptionPane.showMessageDialog(this, "Improperly formatted properties file; incorrect # of fields", "Error", JOptionPane.ERROR_MESSAGE);
                return false;
            }
            if (values.length == 2) {
                node = new DefaultMutableTreeNode(new ItemInfo(key, values[0], values[1]));
            } else {
                node = new DefaultMutableTreeNode(new ItemInfo(key, values[0], values[1], values[2]));
            }
            hashMap.put(key, node);
        }
        for (String key : keys) {
            DefaultMutableTreeNode node = hashMap.get(key);
            DefaultMutableTreeNode parent = hashMap.get(((ItemInfo) node.getUserObject()).parentKey);

            if (parent == null) {
                JOptionPane.showMessageDialog(this, "Improperly formatted properties file; unable to find parent menu "
                        + ((ItemInfo) hashMap.get(key).getUserObject()).parentKey + " for " +
                        key, "Error", JOptionPane.ERROR_MESSAGE);
                return false;
            } else {
                parent.add(node);
            }
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

            if (((ItemInfo) node.getUserObject()).itemName.contains("aternal")) {    // maternal paternal
                openButton30.setEnabled(false);
            } else {
                openButton30.setEnabled(true);
            }
        } else {
            openButton.setEnabled(false);
            openButton30.setEnabled(false);
        }
    }

    public void actionPerformed(ActionEvent e) {
        if (!actionLock) {
            try {
                //use lock as double click protection.
                actionLock = true;
                if (e.getSource() == openButton) {
                    loadFiles(tree.getSelectionPaths(), null);
                } else if (e.getSource() == openButton30) {
                    loadFiles(tree.getSelectionPaths(), "30");
                } else if (e.getSource() == localButton) {
                    LAST_LOADED_HIC_FILE_PATH = HiCFileLoader.loadMenuItemActionPerformed(superAdapter, control, LAST_LOADED_HIC_FILE_PATH);
                    setVisible(false);
                } else if (e.getSource() == urlButton) {
                    HiCFileLoader.loadFromURLActionPerformed(superAdapter, control);
                    setVisible(false);
                } else if (e.getSource() == cancelButton) {
                    setVisible(false);
                    dispose();
                }
            } finally {
                actionLock = false;
            }
        }
    }

    private void loadFiles(TreePath[] paths, String ext) {
        ArrayList<ItemInfo> filesToLoad = new ArrayList<>();
        StringBuilder title = new StringBuilder();

        for (TreePath path : paths) {
            DefaultMutableTreeNode node = (DefaultMutableTreeNode) path.getLastPathComponent();
            if (node != null && node.isLeaf()) {
                filesToLoad.add((ItemInfo) node.getUserObject());
                title.append(path.toString().replace("[", "").replace("]", "").replace(",", ""));
                if (ext != null) title.append(" MAPQ \u2265 ").append(ext);
            }
        }


        setVisible(false);
        java.util.List<String> urls = new ArrayList<>();
        for (ItemInfo info : filesToLoad) {
            if (info.itemURL == null || !info.itemURL.endsWith(".hic")) {
                JOptionPane.showMessageDialog(this, info.itemName + " is not a hic file, or the path to the file is not specified.");
                continue;
            }
            String toadd = info.itemURL;
            if (ext != null) {
                toadd = toadd.replace(".hic", "_" + ext + ".hic");
            }
            System.out.println("New URL: " + toadd);
            urls.add(toadd);
        }

        //code to add a recent file to the menu
        superAdapter.safeLoad(urls, control, title.toString());
    }


    private void collapseAll(JTree tree) {
        int row = tree.getRowCount() - 1;
        while (row >= 0) {
            tree.collapseRow(row);
            row--;
        }
    }


    private class ItemInfo {
        final String uid;
        final String itemName;
        final String parentKey;
        String itemURL;

        ItemInfo(String uid, String parentKey, String itemName, String itemURL) {
            this.uid = uid;
            this.parentKey = parentKey;
            this.itemName = itemName.trim();
            this.itemURL = itemURL.trim();
        }

        ItemInfo(String uid, String parentKey, String itemName) {
            this.parentKey = parentKey;
            this.itemName = itemName;
            this.uid = uid;
        }

        public String toString() {
            return itemName;
        }

    }
}