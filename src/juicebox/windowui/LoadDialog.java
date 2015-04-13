/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2014 Broad Institute, Aiden Lab
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

import juicebox.HiCGlobals;
import juicebox.MainWindow;

import javax.swing.*;
import javax.swing.event.TreeSelectionEvent;
import javax.swing.event.TreeSelectionListener;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.TreePath;
import javax.swing.tree.TreeSelectionModel;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Properties;
import java.util.TreeSet;

public class LoadDialog extends JDialog implements TreeSelectionListener, ActionListener {

    static final long serialVersionUID = 42L;
    private final boolean success;
    private final MainWindow mainWindow;
    private JTree tree;
    private JButton cancelButton;
    private JSplitButton openButton;
    //private JSplitButton localButton;
    private JButton localButton;
    private JMenuItem openURL;
    private JMenuItem open30;

    private static boolean actionLock = false;

    private boolean control;

    public LoadDialog(MainWindow mainWindow, Properties properties) {
        super(mainWindow, "Select file(s) to open");

        this.mainWindow = mainWindow;

        //Create the nodes.
        DefaultMutableTreeNode top =
                new DefaultMutableTreeNode(new ItemInfo("root", "root", ""));
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

        JPanel buttonPanel = new JPanel();

        openButton = new JSplitButton("Open MAPQ > 0");
        openButton.addActionListener(this);
        openButton.setEnabled(false);

        JPopupMenu popupMenu = new JPopupMenu("Popup Menu");
        open30 = new JMenuItem("Open MAPQ \u2265 30");
        open30.addActionListener(this);
        popupMenu.add(open30);
        openButton.setComponentPopupMenu(popupMenu);

        localButton = new JButton("Local...");
        localButton.addActionListener(this);
        localButton.setPreferredSize(new Dimension((int) localButton.getPreferredSize().getWidth(), (int) openButton.getPreferredSize().getHeight()));

        /*JPopupMenu popupMenu1 = new JPopupMenu("Popup1");
        openURL = new JMenuItem("Load URL...");
        openURL.addActionListener(this);
        popupMenu1.add(openURL);
        localButton.setComponentPopupMenu(popupMenu1);
          */
        cancelButton = new JButton("Cancel");
        cancelButton.addActionListener(this);
        cancelButton.setPreferredSize(new Dimension((int) cancelButton.getPreferredSize().getWidth(), (int) openButton.getPreferredSize().getHeight()));

        buttonPanel.add(openButton);
        buttonPanel.add(localButton);

        buttonPanel.add(cancelButton);

        add(buttonPanel, BorderLayout.SOUTH);
        Dimension minimumSize = new Dimension(400, 400);
        setMinimumSize(minimumSize);
        setLocation(100, 100);
        pack();
        success = true;
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
        TreeSet<String> keys = new TreeSet<String>(properties.stringPropertyNames());
        HashMap<String, DefaultMutableTreeNode> hashMap = new HashMap<String, DefaultMutableTreeNode>();
        hashMap.put(((ItemInfo) top.getUserObject()).uid, top);

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
            if (((ItemInfo) node.getUserObject()).itemName.contains("aternal")) {
                open30.setEnabled(false);
            } else {
                open30.setEnabled(true);
            }
        } else {
            openButton.setEnabled(false);
            open30.setEnabled(false);
        }
    }

    public void actionPerformed(ActionEvent e) {
        if (!actionLock) {
            try {
                //use lock as double click protection.
                actionLock = true;
                if (e.getSource() == openButton) {
                    loadFiles(tree.getSelectionPaths(), null);
                } else if (e.getSource() == open30) {
                    loadFiles(tree.getSelectionPaths(), "30");
                }
                if (e.getSource() == localButton) {
                    mainWindow.loadMenuItemActionPerformed(control);
                    setVisible(false);
                } else if (e.getSource() == openURL) {
                    mainWindow.loadFromURLActionPerformed(control);
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
        ArrayList<ItemInfo> filesToLoad = new ArrayList<ItemInfo>();
        String title = "";

        for (TreePath path : paths) {
            DefaultMutableTreeNode node = (DefaultMutableTreeNode) path.getLastPathComponent();
            if (node != null && node.isLeaf()) {
                filesToLoad.add((ItemInfo) node.getUserObject());
                title += path.toString().replace("[", "").replace("]", "").replace(",", "");
                if (ext != null) title += " MAPQ \u2265 " + ext;
            }
        }


        setVisible(false);
        java.util.List<String> urls = new ArrayList<String>();
        for (ItemInfo info : filesToLoad) {
            if (info.itemURL == null || !info.itemURL.endsWith(".hic")) {
                JOptionPane.showMessageDialog(this, info.itemName + " is not a hic file, or the path to the file is not specified.");
                continue;
            }
            String toadd = info.itemURL;
            if (ext != null) {
                toadd = toadd.replace(".hic", "_" + ext + ".hic");
            }
            urls.add(toadd);
        }

        //code to add a recent file to the menu
        mainWindow.getRecentMapMenu().addEntry(title.trim() + "@@" + urls.get(0), true);
        mainWindow.safeLoad(urls, control);

        mainWindow.updateTitle(control, title);
    }


    private class ItemInfo {
        public final String uid;
        public final String itemName;
        public final String parentKey;
        public String itemURL;

        public ItemInfo(String uid, String parentKey, String itemName, String itemURL) {
            this.uid = uid;
            this.parentKey = parentKey;
            this.itemName = itemName.trim();
            this.itemURL = itemURL.trim();
        }

        public ItemInfo(String uid, String parentKey, String itemName) {
            this.parentKey = parentKey;
            this.itemName = itemName;
            this.uid = uid;
        }

        public String toString() {
            return itemName;
        }

    }
}