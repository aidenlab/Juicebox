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

package juicebox.track;

import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.gui.SuperAdapter;
import juicebox.windowui.NormalizationType;
import org.broad.igv.ui.color.ColorUtilities;
import org.broad.igv.ui.util.FileDialogUtils;
import org.broad.igv.ui.util.LinkCheckBox;
import org.broad.igv.util.ResourceLocator;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import javax.swing.*;
import javax.swing.tree.*;
import java.awt.*;
import java.awt.event.*;
import java.io.File;
import java.util.List;
import java.util.*;

import static org.broad.igv.util.ResourceLocator.AttributeType.*;

/**
 * Parses XML file of IGV resources, and displays them in tree format.
 *
 * @author eflakes
 */
public class ResourceTree {

    private static final String EIGENVECTOR = "Eigenvector";
    private static final String SUBCOMPARTMENTS = "Subcompartments";
    private static final String GLOBAL = "Global";
    private final List<CheckableResource> leafResources = new ArrayList<>();
    private final JTree dialogTree;
    private final Set<ResourceLocator> loadedLocators;
    private JDialog dialog;

    private DefaultMutableTreeNode oneDFeatureRoot;

    private LinkedHashSet<ResourceLocator> newLocators;
    private LinkedHashSet<ResourceLocator> deselectedLocators;
    private LinkedHashSet<DefaultMutableTreeNode> addedNodes;
    private File openAnnotationPath = null;

    public ResourceTree(final HiC hic, Document document) {
        dialog = null;
        loadedLocators = new HashSet<>();

        dialogTree = new JTree(new DefaultMutableTreeNode("Available feature sets"));
        dialogTree.setExpandsSelectedPaths(true);
        dialogTree.setCellRenderer(new NodeRenderer());
        dialogTree.setCellEditor(new ResourceEditor(dialogTree));
        dialogTree.setEditable(true);

        if (document != null) {
            DefaultMutableTreeNode node = createTreeFromDOM(document);
            ((DefaultMutableTreeNode) dialogTree.getModel().getRoot()).add(node);
        }

        createTreeFromDataset(hic, this);
        expandTree();
        hic.setResourceTree(this);
        dialogTree.setRootVisible(false);


        MouseListener ml = new MouseAdapter() {
            public void mousePressed(MouseEvent e) {
                int selRow = dialogTree.getRowForLocation(e.getX(), e.getY());
                final TreePath selPath = dialogTree.getPathForLocation(e.getX(), e.getY());
                if (selRow != -1 && selPath != null) {
                    if (SwingUtilities.isRightMouseButton(e)) {

                        // removing (DefaultMutableTreeNode) cast to selpath.getlast... (revert if error)
                        //noinspection SuspiciousMethodCalls
                        if (addedNodes != null &&
                                addedNodes.contains(selPath.getLastPathComponent())) {
                            JPopupMenu menu = new JPopupMenu("popup");

                            JMenuItem menuItem = new JMenuItem("Remove feature");
                            menuItem.addActionListener(new ActionListener() {
                                @Override
                                public void actionPerformed(ActionEvent e) {
                                    removeFeature((DefaultMutableTreeNode) selPath.getLastPathComponent());
                                }
                            });
                            menu.add(menuItem);
                            menu.show(dialogTree, e.getX(), e.getY());
                        }
                    }
                }
            }
        };
        dialogTree.addMouseListener(ml);

    }

    private static String getAttribute(Element element, String key) {

        String value = element.getAttribute(key);
        if (value != null) {
            if (value.trim().equals("")) {
                value = null;
            }
        }
        return value;
    }

    public LinkedHashSet<ResourceLocator> getLocators() {
        return newLocators;
    }

    public LinkedHashSet<ResourceLocator> getDeselectedLocators() {
        return deselectedLocators;
    }

    /**
     * Shows a tree of selectable resources.
     *
     * @param parent Parent window
     * @return the resources selected by user.
     */
    public void showResourceTreeDialog(JFrame parent) {
        newLocators = new LinkedHashSet<>();
        deselectedLocators = new LinkedHashSet<>();

        dialog = new JDialog(parent, "Available Features", true);

        JPanel treePanel = new JPanel(new BorderLayout());
        JScrollPane pane = new JScrollPane(dialogTree);
        treePanel.add(pane, BorderLayout.CENTER);

        pane.setPreferredSize(new Dimension(650, 500));
        pane.setOpaque(true);
        Color backGroundColor = HiCGlobals.isDarkulaModeEnabled ? Color.BLACK : Color.WHITE;
        pane.setBackground(backGroundColor);
        pane.setViewportView(dialogTree);

        dialog.setBackground(backGroundColor);
        dialog.getContentPane().setBackground(backGroundColor);

        Component[] children = treePanel.getComponents();
        if (children != null) {
            for (Component child : children) {
                child.setBackground(backGroundColor);
            }
        }

        JPanel buttonPanel = new JPanel();
        JButton okButton = new JButton("OK");
        JButton cancelButton = new JButton("Cancel");

        cancelButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {

                dialog.dispose();
                // get selected locators
                LinkedHashSet<ResourceLocator> selectedLocators = getSelectedResourceLocators();
                LinkedHashSet<ResourceLocator> newlyAddedLocators = new LinkedHashSet<>();
                // these have been added from dialog open to hitting the "cancel" button
                for (ResourceLocator locator : selectedLocators) {
                    if (!loadedLocators.contains(locator)) {
                        newlyAddedLocators.add(locator);
                    }
                }
                // roll back that change (remove the selected ones)
                for (ResourceLocator locator : newlyAddedLocators) {
                    remove(locator);
                }
                // add back in anything that was deselected
                for (ResourceLocator locator : loadedLocators) {
                    addBack(locator);
                }

            }
        });

        okButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                dialog.dispose();
                LinkedHashSet<ResourceLocator> selectedLocators = getSelectedResourceLocators();
                for (ResourceLocator locator : selectedLocators) {
                    if (!loadedLocators.contains(locator)) {
                        newLocators.add(locator);
                    }
                }
                for (ResourceLocator locator : loadedLocators) {
                    if (!selectedLocators.contains(locator)) {
                        deselectedLocators.add(locator);
                    }
                }
                // add these to loaded ones for next use
                loadedLocators.addAll(newLocators);
                for (ResourceLocator locator : deselectedLocators) {
                    loadedLocators.remove(locator);
                }
                dialogTree.clearSelection();
            }
        });

        buttonPanel.add(okButton);
        buttonPanel.add(cancelButton);

        dialog.add(treePanel);
        dialog.add(buttonPanel, BorderLayout.PAGE_END);

        dialog.setResizable(true);
        dialog.pack();

        dialog.setLocationRelativeTo(parent);
        dialog.setVisible(true);

    }

    public boolean addLocalButtonActionPerformed(final SuperAdapter superAdapter) {
        Boolean localFilesAdded = Boolean.FALSE;

        File[] oneDfiles = FileDialogUtils.chooseMultiple("Choose 1D Annotation file", openAnnotationPath, null);

        if (oneDfiles != null && oneDfiles.length > 0) {
            for (File file : oneDfiles) {

                if (file == null || !file.exists()) continue;

                localFilesAdded = Boolean.TRUE;

                String path = file.getAbsolutePath();
                openAnnotationPath = new File(path);

                ResourceLocator locator = new ResourceLocator(path);
                locator.setName(file.getName());
                locator.setType(file.getName());

                CheckableResource resource = new CheckableResource(file.getName(), true, locator);

                DefaultMutableTreeNode treeNode = new DefaultMutableTreeNode(file);

                treeNode.setUserObject(resource);

                if (containsDuplicate(treeNode)) {
                    if (HiCGlobals.guiIsCurrentlyActive) {
                        int dialogResult = JOptionPane.showConfirmDialog(superAdapter.getMainWindow(),
                                file.getName() + " is already loaded. Would you like to overwrite it?", "Warning",
                                JOptionPane.YES_NO_OPTION);
                        if (dialogResult == JOptionPane.YES_OPTION) {
                            removeChildNodeFromAddedNodes(treeNode);
                            removeChildNodeFromFeatureRoot(treeNode);
                            removeResourceFromLeaf(resource, leafResources);
                        } else {
                            setSelectionForMatchingChildNodes(treeNode, Boolean.TRUE);
                            continue;
                        }
                    }
                }

                leafResources.add(resource);

                oneDFeatureRoot.add(treeNode);
                ((CheckableResource) oneDFeatureRoot.getUserObject()).setSelected(true);

                if (addedNodes == null) {
                    addedNodes = new LinkedHashSet<>();
                }
                addedNodes.add(treeNode);

                expandTree();
                dialogTree.updateUI();
            }
        }
        return localFilesAdded;
    }

    private boolean containsDuplicate(DefaultMutableTreeNode treeNode) {
        return !getMatchingChildNodesFromAddedNodes(treeNode).isEmpty();
    }

    private void setSelectionForMatchingChildNodes(DefaultMutableTreeNode treeNode, Boolean selectionBoolean) {
        List<DefaultMutableTreeNode> desiredChildNodes = getMatchingChildNodesFromAddedNodes(treeNode);
        for (DefaultMutableTreeNode childNode : desiredChildNodes) {
            if (childNode.getUserObject() instanceof CheckableResource) {
                ((CheckableResource) childNode.getUserObject()).setSelected(selectionBoolean);
            }
        }
    }

    private void removeResourceFromLeaf(CheckableResource resource, List<CheckableResource> leafResources) {
        List<CheckableResource> resourcesToRemove = new ArrayList<>();
        for (CheckableResource res : leafResources) {
            if (res.getText().equals(resource.getText()))
                resourcesToRemove.add(res);
        }
        leafResources.removeAll(resourcesToRemove);
    }

    private void removeChildNodeFromFeatureRoot(DefaultMutableTreeNode treeNode) {
        List<DefaultMutableTreeNode> childNodesToRemove = new ArrayList<>();
        if (!(treeNode.getUserObject() instanceof CheckableResource)) {
            return;
        }
        String targetTreeNodePath = ((CheckableResource) treeNode.getUserObject()).getResourceLocator().getPath();
        Enumeration<?> childNodesPresent = oneDFeatureRoot.children();
        while (childNodesPresent.hasMoreElements()) {
            DefaultMutableTreeNode currentChildNode = (DefaultMutableTreeNode) childNodesPresent.nextElement();
            if (currentChildNode.getUserObject() instanceof CheckableResource) {
                String currentChildNodePath = ((CheckableResource) currentChildNode.getUserObject()).getResourceLocator().getPath();
                if (currentChildNodePath.equals(targetTreeNodePath)) {
                    childNodesToRemove.add(currentChildNode);
                }
            }
        }
        for (DefaultMutableTreeNode childNode : childNodesToRemove) {
            oneDFeatureRoot.remove(childNode);
        }
    }

    private void removeChildNodeFromAddedNodes(DefaultMutableTreeNode treeNode) {
        List<DefaultMutableTreeNode> childNodesToRemove = getMatchingChildNodesFromAddedNodes(treeNode);
        for (DefaultMutableTreeNode childNode : childNodesToRemove) {
            addedNodes.remove(childNode);
        }
    }

    private List<DefaultMutableTreeNode> getMatchingChildNodesFromAddedNodes(DefaultMutableTreeNode treeNode) {
        List<DefaultMutableTreeNode> matchingChildNodes = new ArrayList<>();
        if (addedNodes != null && treeNode.getUserObject() instanceof CheckableResource) {
            String targetPath = ((CheckableResource) treeNode.getUserObject()).getResourceLocator().getPath();
            for (DefaultMutableTreeNode childNode : addedNodes) {
                if (childNode.getUserObject() instanceof CheckableResource) {
                    String childPath = ((CheckableResource) childNode.getUserObject()).getResourceLocator().getPath();
                    if (childPath.equals(targetPath)) {
                        matchingChildNodes.add(childNode);
                    }
                }
            }
        }
        return matchingChildNodes;
    }

    private void removeFeature(DefaultMutableTreeNode node) {
        ((CheckableResource) node.getUserObject()).setSelected(false);
        ResourceEditor.checkOrUncheckParentNodesRecursively(node, false);
        addedNodes.remove(node);
        DefaultMutableTreeNode parent = (DefaultMutableTreeNode) node.getParent();
        parent.remove(node);


        ResourceLocator locator = ((CheckableResource) node.getUserObject()).getResourceLocator();

        // no longer needed? MSS
        //String path = locator.getPath();
        //hic.removeLoadedAnnotation(path); // actually removes the entry (at least 2d annotation) so that it can be reloaded

        deselectedLocators.add(locator);
        loadedLocators.remove(locator);
        newLocators.remove(locator);

        removeResourceFromLeaf((CheckableResource) node.getUserObject(), leafResources);
        //leafResources.remove(node);
        dialogTree.updateUI();


    }

    private void createTreeFromDataset(HiC hic, ResourceTree resourceTree) {
        oneDFeatureRoot = new DefaultMutableTreeNode("Dataset-specific 1D Features");
        ResourceLocator locator = new ResourceLocator("Dataset-specific 1D Features");
        locator.setName("Dataset-specific 1D Features");
        locator.setType("norm");
        CheckableResource rootResource = new CheckableResource("Dataset-specific 1D Features", false, locator);
        oneDFeatureRoot.setUserObject(rootResource);
        oneDFeatureRoot.setAllowsChildren(true);

        if (hic.getDataset().getVersion() >= 6) {
            if (hic.getDataset().getNormalizationTypes().size() > 0) {
                DefaultMutableTreeNode normNode = new DefaultMutableTreeNode("Coverage normalizations");
                oneDFeatureRoot.add(normNode);
                locator = new ResourceLocator("Coverage normalizations");
                locator.setName("Coverage normalizations");
                locator.setType("norm");
                CheckableResource resource = new CheckableResource("Coverage normalizations", false, locator);
                normNode.setUserObject(resource);
                normNode.setAllowsChildren(true);

                for (NormalizationType t : hic.getDataset().getNormalizationTypes()) {

                    String label = t.getDescription();
                    locator = new ResourceLocator(label);
                    locator.setType("norm");
                    locator.setName(label);
                    resource = new CheckableResource(label, false, locator);

                    DefaultMutableTreeNode treeNode = new DefaultMutableTreeNode(label);
                    normNode.add(treeNode);
                    treeNode.setUserObject(resource);
                    resource.setEnabled(resourceTree.dialogTree.isEnabled());
                    treeNode.setAllowsChildren(false);
                    leafResources.add(resource);
                }
            }
        }


        locator = new ResourceLocator(EIGENVECTOR);
        locator.setType(EIGENVECTOR.toLowerCase());
        locator.setName(EIGENVECTOR);
        CheckableResource resource = new CheckableResource(EIGENVECTOR, false, locator);

        DefaultMutableTreeNode treeNode = new DefaultMutableTreeNode(EIGENVECTOR);
        oneDFeatureRoot.add(treeNode);
        treeNode.setUserObject(resource);
        resource.setEnabled(resourceTree.dialogTree.isEnabled());
        treeNode.setAllowsChildren(false);
        leafResources.add(resource);

        locator = hic.getDataset().getSubcompartments();
        if (locator != null) {
            resource = new CheckableResource(SUBCOMPARTMENTS, false, locator);

            treeNode = new DefaultMutableTreeNode(SUBCOMPARTMENTS);
            oneDFeatureRoot.add(treeNode);
            treeNode.setUserObject(resource);
            resource.setEnabled(resourceTree.dialogTree.isEnabled());
            treeNode.setAllowsChildren(false);
            leafResources.add(resource);
        }

        ((DefaultMutableTreeNode) dialogTree.getModel().getRoot()).add(oneDFeatureRoot);
    }

    private DefaultMutableTreeNode createTreeFromDOM(Document document) {

        Element rootElement =
                (Element) document.getElementsByTagName(GLOBAL).item(0);

        if (rootElement == null) {
            return new DefaultMutableTreeNode("");
        }

        String nodeName = rootElement.getNodeName();
        if (!nodeName.equalsIgnoreCase(GLOBAL)) {
            throw new RuntimeException(rootElement +
                    " is not the root of the xml document!");
        }

        String rootLabel = getAttribute(rootElement, "name");
        DefaultMutableTreeNode rootNode = new DefaultMutableTreeNode(rootLabel);

        // Build and attach descendants of the root node to the tree
        buildLocatorTree(rootNode, rootElement);

        return rootNode;
    }

    /**
     * Build a tree of all resources, placed under {@code treeNode}, starting
     * from {@code xmlNode}.
     *
     * @param treeNode
     * @param xmlNode
     */
    private void buildLocatorTree(DefaultMutableTreeNode treeNode, Element xmlNode) {

        String name = getAttribute(xmlNode, NAME.getText());

        ResourceLocator locator = new ResourceLocator(
                getAttribute(xmlNode, URL.getText()),
                getAttribute(xmlNode, PATH.getText())
        );
        locator.setName(name);

        if (xmlNode.getTagName().equalsIgnoreCase("Resource")) {

            String resourceType = getAttribute(xmlNode, RESOURCE_TYPE.getText());
            locator.setType(resourceType);

            String infoLink = getAttribute(xmlNode, HYPERLINK.getText());
            if (infoLink == null) {
                infoLink = getAttribute(xmlNode, INFOLINK.getText());
            }
            locator.setFeatureInfoURL(infoLink);

            String sampleId = getAttribute(xmlNode, SAMPLE_ID.getText());
            if (sampleId == null) {
                // legacy option
                sampleId = getAttribute(xmlNode, ID.getText());
            }
            locator.setSampleId(sampleId);
            locator.setDescription(getAttribute(xmlNode, DESCRIPTION.getText()));
            locator.setTrackLine(getAttribute(xmlNode, TRACK_LINE.getText()));
            locator.setName(name);
            // Special element for alignment tracks
            locator.setCoverage(getAttribute(xmlNode, COVERAGE.getText()));

            String colorString = getAttribute(xmlNode, COLOR.getText());
            if (colorString != null) {
                try {
                    Color c = ColorUtilities.stringToColor(colorString);
                    locator.setColor(c);
                } catch (Exception e) {
                    System.err.println("Error setting color: " + e.getLocalizedMessage());
                }
            }
        }

        NodeList nodeList = xmlNode.getChildNodes();
        Node xmlChildNode;

        // If we have children treat it as a category not a leaf
        for (int i = 0; i < nodeList.getLength(); i++) {

            xmlChildNode = nodeList.item(i);
            String nodeName = xmlChildNode.getNodeName();
            if (nodeName.equalsIgnoreCase("#text")) {
                continue;
            }

            // Need to check class of child node, its not necessarily an
            // element (could be a comment for example).
            if (xmlChildNode instanceof Element) {
                String categoryLabel = getAttribute((Element) xmlChildNode, NAME.getText());
                DefaultMutableTreeNode treeChildNode = new DefaultMutableTreeNode(categoryLabel);
                treeNode.add(treeChildNode);
                buildLocatorTree(treeChildNode, (Element) xmlChildNode);
            }
        }

        // If it's a leaf set the checkbox to represent the resource
        if (treeNode.isLeaf()) {
            CheckableResource resource = new CheckableResource(name, false, locator);
            treeNode.setUserObject(resource);
            treeNode.setAllowsChildren(false);
            leafResources.add(resource);
        } else {
            locator = new ResourceLocator(name);
            CheckableResource resource = new CheckableResource(name, false, locator);
            treeNode.setUserObject(resource);
            treeNode.setAllowsChildren(true);
        }
    }

    public void remove(ResourceLocator locator) {
        //locator.
        loadedLocators.remove(locator);

        Enumeration<?> enumeration = ((DefaultMutableTreeNode) dialogTree.getModel().getRoot()).preorderEnumeration();
        // skip root
        enumeration.nextElement();
        while (enumeration.hasMoreElements()) {
            try {
                DefaultMutableTreeNode node = (DefaultMutableTreeNode) enumeration.nextElement();
                CheckableResource resource = (CheckableResource) node.getUserObject();
                if (locator.equals(resource.getResourceLocator())) {
                    resource.setSelected(false);
                    ResourceEditor.checkOrUncheckParentNodesRecursively(node, false);
                }
            } catch (Exception e) {
                System.err.println("There appears to be an invalid node in the resource tree");
            }
        }

    }

    private void addBack(ResourceLocator locator) {
        Enumeration<?> enumeration = ((DefaultMutableTreeNode) dialogTree.getModel().getRoot()).preorderEnumeration();
        // skip root
        enumeration.nextElement();
        while (enumeration.hasMoreElements()) {
            DefaultMutableTreeNode node = (DefaultMutableTreeNode) enumeration.nextElement();
            try {
                CheckableResource resource = (CheckableResource) node.getUserObject();
                if (locator.equals(resource.getResourceLocator())) {
                    resource.setSelected(true);
                    ResourceEditor.checkOrUncheckParentNodesRecursively(node, true);
                }
            } catch (Exception ignored) {
            }
        }
    }

    private LinkedHashSet<ResourceLocator> getSelectedResourceLocators() {

        LinkedHashSet<ResourceLocator> resourceLocators = new LinkedHashSet<>();

        for (CheckableResource resource : leafResources) {

            if (resource.isSelected()) {
                resourceLocators.add(resource.getResourceLocator());
            }
        }
        return resourceLocators;
    }

    public void checkTrackBoxesForReloadState(String track) {
        Enumeration<?> en = ((DefaultMutableTreeNode) dialogTree.getModel().getRoot()).preorderEnumeration();
        //skip root
        en.nextElement();
        while (en.hasMoreElements()) {
            try {
                DefaultMutableTreeNode node = (DefaultMutableTreeNode) en.nextElement();
                CheckableResource resource = (CheckableResource) node.getUserObject();
                if (node.isLeaf()) {
                    if (resource.dataResourceLocator.getPath() != null) {
                        if (resource.dataResourceLocator.getName().contains(track)) {
                            resource.setSelected(true);
                            resource.setEnabled(true);
                            loadedLocators.add(resource.getResourceLocator());
                            ResourceEditor.checkOrUncheckParentNodesRecursively(node, true);
                            //System.out.println("name: "+resource.dataResourceLocator.getName()); for debugging
                        }
                    }
                }
            } catch (Exception ignored) {
            }
            }
        }


    /**
     * Expands tree.
     */
    private void expandTree() {
        TreeNode root = (TreeNode) dialogTree.getModel().getRoot();
        TreePath rootPath = new TreePath(root);
        TreeNode node = (TreeNode) rootPath.getLastPathComponent();
        for (Enumeration<?> e = node.children(); e.hasMoreElements(); ) {
            TreePath childPath = rootPath.pathByAddingChild(e.nextElement());
            if (!dialogTree.isExpanded(childPath)) {
                dialogTree.expandPath(childPath);
            }
        }
        if (!dialogTree.isExpanded(rootPath)) {
            dialogTree.expandPath(rootPath);
        }
    }

    /**
     *
     */
    interface SelectableResource extends DataResource {

        boolean isSelected();

        void setSelected(boolean newValue);
    }

    /**
     *
     */
    interface DataResource {

        ResourceLocator getResourceLocator();

        String getText();

        void setText(String newValue);

        boolean isEnabled();

        void setEnabled(boolean value);
    }

    /**
     * Node's Renderer
     */
    static class NodeRenderer implements TreeCellRenderer {

        private final LinkCheckBox renderer = new LinkCheckBox();
        private final Color selectionForeground;
        private final Color selectionBackground;
        private final Color textForeground;
        private final Color textBackground;

        NodeRenderer() {

            Font fontValue;
            fontValue = UIManager.getFont("Tree.font");
            if (fontValue != null) {
                renderer.setFont(fontValue);
            }
            Boolean booleanValue =
                    (Boolean) UIManager.get("Tree.drawsFocusBorderAroundIcon");
            renderer.setFocusPainted(
                    (booleanValue != null) && booleanValue);

            selectionForeground = UIManager.getColor("Tree.selectionForeground");
            selectionBackground = UIManager.getColor("Tree.selectionBackground");
            textForeground = UIManager.getColor("Tree.textForeground");
            textBackground = UIManager.getColor("Tree.textBackground");
            renderer.setSelected(false);
        }

        public Component getTreeCellRendererComponent(JTree tree, Object value,
                                                      boolean isNodeSelected, boolean isNodeExpanded, boolean isLeaf,
                                                      int row, boolean hasFocus) {

            // Convert value into a usable string
            String stringValue = "";
            if (value != null) {
                String toStringValue = value.toString();
                if (toStringValue != null) {
                    stringValue = toStringValue;
                }
            }

            // Initialize checkbox state and selection
            renderer.setSelected(false);
            renderer.setText(stringValue);
            renderer.setEnabled(tree.isEnabled());

            // Tell renderer how to highlight nodes on selection
            if (isNodeSelected) {
                renderer.setForeground(selectionForeground);
                renderer.setBackground(selectionBackground);
            } else {
                renderer.setForeground(textForeground);
                renderer.setBackground(textBackground);
            }

            if (value != null) {
                if (value instanceof DefaultMutableTreeNode) {

                    DefaultMutableTreeNode node =
                            (DefaultMutableTreeNode) value;

                    Object userObject = node.getUserObject();
                    if (userObject instanceof CheckableResource) {

                        CheckableResource resource = (CheckableResource) userObject;
                        renderer.setText(resource.getText());
                        renderer.setSelected(resource.isSelected());
                        renderer.setEnabled(resource.isEnabled());

                        String hyperLink = resource.getResourceLocator().getFeatureInfoURL();
                        if (hyperLink == null) {
                            renderer.showHyperLink(false);
                        } else {
                            renderer.setHyperLink(hyperLink);
                            renderer.showHyperLink(true);
                        }
                    }
                }
            }

            return renderer;
        }

        LinkCheckBox getRendereringComponent() {
            return renderer;
        }
    }

    /**
     * Node's Resource Editor
     */
    static class ResourceEditor extends AbstractCellEditor
            implements TreeCellEditor {

        private static final long serialVersionUID = -5631405993171929756L;
        final NodeRenderer renderer = new NodeRenderer();
        final JTree tree;

        ResourceEditor(JTree tree) {
            this.tree = tree;
        }

        /**
         * Call to recursively check or uncheck the parent ancestors of the
         * passed node.
         */
        static void checkOrUncheckParentNodesRecursively(TreeNode node,
                                                         boolean checkParentNode) {

            if (node == null) {
                return;
            }

            TreeNode parentNode = node.getParent();
            if (parentNode == null) {
                return;
            }

            Object parentUserObject =
                    ((DefaultMutableTreeNode) parentNode).getUserObject();

            CheckableResource parentNodeResource = null;
            if (parentUserObject instanceof CheckableResource) {
                parentNodeResource = ((CheckableResource) parentUserObject);
            }

            if (parentNodeResource != null) {

                // If parent's current check state matchs what we want there
                // is nothing to do so just leave
                if (parentNodeResource.isSelected() == checkParentNode) {
                    return;
                } else if (checkParentNode) {
                    parentNodeResource.setSelected(true);
                } else { // Uncheck Only if their are no selected descendants

                    if (doesNotHaveSelectedChildren(parentNode)) {
                        parentNodeResource.setSelected(false);
                    }
                }
            }

            checkOrUncheckParentNodesRecursively(parentNode,
                    checkParentNode);
        }

        /*
        * Uncheck a node unless rule prevent this behavior.
        */

        static boolean hasSelectedDescendants(TreeNode treeNode) {

            Enumeration<?> children = treeNode.children();
            while (children.hasMoreElements()) {

                TreeNode childNode = (TreeNode) children.nextElement();

                Object childsUserObject =
                        ((DefaultMutableTreeNode) childNode).getUserObject();
                if (childsUserObject instanceof CheckableResource) {
                    CheckableResource childResource =
                            ((CheckableResource) childsUserObject);

                    // If has selected say so
                    if (childResource.isSelected()) {
                        return true;
                    }
                }

                // If has selected descendant say so
                if (hasSelectedDescendants(childNode)) {
                    return true;
                }
            }
            return false;
        }

        static boolean doesNotHaveSelectedChildren(TreeNode treeNode) {

            Enumeration<?> children = treeNode.children();
            while (children.hasMoreElements()) {

                TreeNode childNode = (TreeNode) children.nextElement();

                Object childsUserObject =
                        ((DefaultMutableTreeNode) childNode).getUserObject();
                if (childsUserObject instanceof CheckableResource) {
                    CheckableResource childResource =
                            ((CheckableResource) childsUserObject);
                    if (childResource.isSelected()) {
                        return false;
                    }
                }
            }
            return true;
        }

        public Object getCellEditorValue() {

            DataResource resource = null;
            TreePath treePath = tree.getEditingPath();
            if (treePath != null) {

                Object node = treePath.getLastPathComponent();

                if ((node instanceof DefaultMutableTreeNode)) {

                    LinkCheckBox checkbox = renderer.getRendereringComponent();

                    DefaultMutableTreeNode treeNode =
                            (DefaultMutableTreeNode) node;

                    Object userObject = treeNode.getUserObject();

                    resource = (CheckableResource) userObject;

                    // Don't change resource if disabled
                    if (!resource.isEnabled()) {
                        return resource;
                    }

                    boolean isChecked = checkbox.isSelected();

                    // Check/Uncheck the selected node. This code ONLY handles
                    // the clicked node. Not it's ancestors or decendants.
                    if (isChecked) {
                        ((CheckableResource) resource).setSelected(true);
                    } else {

                        // See if we are allowed to unchecking this specific
                        // node - if not, it won't be done. This does not
                        // prevent it's children from being unchecked.
                        uncheckCurrentNodeIfAllowed((CheckableResource) resource,
                                treeNode);
                    }


                    /*
                    * Now we have to check or uncheck the descendants and
                    * ancestors depending on what we did above.
                    */

                    boolean checkRelatives = isChecked;

                    // If we found a mix of select leave and selected but
                    // but disabled leave we must be trying to toggle off
                    // the children
                    if (hasSelectedAndLockedDescendants(treeNode)) {
                        checkRelatives = false;
                    }
                    // If we found only locked leave we must be trying to toggle
                    // on the unlocked children
                    else if (hasLockedDescendants(treeNode)) {
                        checkRelatives = true;
                    }
                    // Otherwise, just use the value of the checkbox


                    if (!treeNode.isLeaf()) { //check up and down the tree

                        // If not a leaf check/uncheck children as requested
                        checkOrUncheckChildNodesRecursively(treeNode, checkRelatives);

                        // If not a leaf check/uncheck ancestors
                        checkOrUncheckParentNodesRecursively(treeNode,
                                ((CheckableResource) resource).isSelected());
                    } else { // it must be a leaf - so check up the tree
                        checkOrUncheckParentNodesRecursively(treeNode, checkRelatives);
                    }
                }
                tree.treeDidChange();
            }
            return resource;
        }

        private void uncheckCurrentNodeIfAllowed(CheckableResource resource,
                                                 TreeNode treeNode) {

            // If we are unchecking a parent make sure there are
            // no checked children
            if (doesNotHaveSelectedChildren(treeNode)) {
                resource.setSelected(false);
            } else {

                // If node has selected children and has disabled descendants we
                // must not unselect
                if (hasLockedDescendants(treeNode)) {
                    resource.setSelected(true);
                } else {
                    // No disabled descendants so we can uncheck at will
                    resource.setSelected(false);
                }
            }
        }

        /**
         * Can only be called from getCellEditorValue() to recursively check
         * or uncheck the children of the passed parent node.
         */
        private void checkOrUncheckChildNodesRecursively(TreeNode currentNode,
                                                         boolean isCheckingNeeded) {

            Object parentUserObject =
                    ((DefaultMutableTreeNode) currentNode).getUserObject();

            CheckableResource currentTreeNodeResource = null;
            if (parentUserObject instanceof CheckableResource) {
                currentTreeNodeResource = ((CheckableResource) parentUserObject);
            }

            if (currentTreeNodeResource != null) {

                // Set all enabled children to the checked state of their parent
                Enumeration<?> children = currentNode.children();
                while (children.hasMoreElements()) {

                    TreeNode childNode = (TreeNode) children.nextElement();

                    Object childsUserObject =
                            ((DefaultMutableTreeNode) childNode).getUserObject();
                    if (childsUserObject instanceof CheckableResource) {

                        CheckableResource childResource =
                                ((CheckableResource) childsUserObject);

                        if (childResource.isEnabled()) {

                            // Child must be checked if it has selected
                            // selected and disabled descendants
                            if (hasLockedDescendants(childNode)) {
                                childResource.setSelected(true);
                            } else { // else check/uncheck  as requested
                                childResource.setSelected(isCheckingNeeded);
                            }
                        }
                    }
                    checkOrUncheckChildNodesRecursively(childNode,
                            isCheckingNeeded);
                }
            }
        }

        boolean hasLockedDescendants(TreeNode treeNode) {

            Enumeration<?> children = treeNode.children();
            while (children.hasMoreElements()) {

                TreeNode childNode = (TreeNode) children.nextElement();

                Object childsUserObject =
                        ((DefaultMutableTreeNode) childNode).getUserObject();
                if (childsUserObject instanceof CheckableResource) {
                    CheckableResource childResource =
                            ((CheckableResource) childsUserObject);

                    // If disabled say so
                    if (!childResource.isEnabled()) {
                        return true;
                    }
                }

                // If a descendant is disabled say so
                if (hasLockedDescendants(childNode)) {
                    return true;
                }
            }
            return false;
        }

        /**
         * Return true if it find nodes that ar both selected and disabled
         *
         * @param treeNode
         * @return true if we are working with preselected nodes
         */
      /*  public boolean hasLockedChildren(TreeNode treeNode) {

            boolean hasSelectedAndDisabled = false;
            Enumeration children = treeNode.children();
            while (children.hasMoreElements()) {

                TreeNode childNode = (TreeNode) children.nextElement();

                Object childsUserObject =
                        ((DefaultMutableTreeNode) childNode).getUserObject();
                if (childsUserObject instanceof CheckableResource) {

                    CheckableResource childResource =
                            ((CheckableResource) childsUserObject);

                    if (!childResource.isEnabled() && childResource.isSelected()) {
                        hasSelectedAndDisabled = true;
                    }

                    if (hasSelectedAndDisabled) {
                        break;
                    }
                }
            }
            return (hasSelectedAndDisabled);
        }          */

        /**
         * @param treeNode
         * @return true if we are working with preselected nodes
         */
     /*   public boolean hasSelectedAndLockedChildren(TreeNode treeNode) {

            boolean hasSelected = false;
            boolean hasSelectedAndDisabled = false;
            Enumeration children = treeNode.children();
            while (children.hasMoreElements()) {

                TreeNode childNode = (TreeNode) children.nextElement();

                Object childsUserObject =
                        ((DefaultMutableTreeNode) childNode).getUserObject();
                if (childsUserObject instanceof CheckableResource) {

                    CheckableResource childResource =
                            ((CheckableResource) childsUserObject);

                    if (childResource.isSelected() && childResource.isEnabled()) {
                        hasSelected = true;
                    }
                    if (!childResource.isEnabled() && childResource.isSelected()) {
                        hasSelectedAndDisabled = true;
                    }

                    if (hasSelected & hasSelectedAndDisabled) {
                        break;
                    }
                }
            }

            // If we have both we can return true
            return (hasSelected & hasSelectedAndDisabled);
        }                    /*

        /**
         * @param treeNode
         * @return true if we are working with preselected nodes
         */
        boolean hasSelectedAndLockedDescendants(TreeNode treeNode) {

            boolean hasSelected = false;
            boolean hasSelectedAndDisabled = false;
            Enumeration<?> children = treeNode.children();
            while (children.hasMoreElements()) {

                TreeNode childNode = (TreeNode) children.nextElement();

                Object childsUserObject =
                        ((DefaultMutableTreeNode) childNode).getUserObject();
                if (childsUserObject instanceof CheckableResource) {

                    CheckableResource childResource =
                            ((CheckableResource) childsUserObject);

                    if (childResource.isSelected() && childResource.isEnabled()) {
                        hasSelected = true;
                    }
                    if (!childResource.isEnabled() && childResource.isSelected()) {
                        hasSelectedAndDisabled = true;
                    }

                    if (hasSelected & hasSelectedAndDisabled) {
                        break;
                    }
                }

                // If has a mix of selected and checked but disableddescendant
                if (hasSelectedAndLockedDescendants(childNode)) {
                    return true;
                }

            }

            // If we have both we can return true
            return (hasSelected & hasSelectedAndDisabled);
        }

        @Override
        public boolean isCellEditable(EventObject event) {

            boolean returnValue = false;

            if (event instanceof MouseEvent) {

                MouseEvent mouseEvent = (MouseEvent) event;
                TreePath treePath = tree.getPathForLocation(
                        mouseEvent.getX(), mouseEvent.getY());

                if (treePath != null) {

                    Object node = treePath.getLastPathComponent();
                    if ((node instanceof DefaultMutableTreeNode)) {
                        DefaultMutableTreeNode treeNode =
                                (DefaultMutableTreeNode) node;
                        Object userObject = treeNode.getUserObject();

                        if (userObject instanceof CheckableResource) {
                            returnValue = true;
                        }
                    }
                }
            }
            return returnValue;
        }

        public Component getTreeCellEditorComponent(JTree tree, Object value,
                                                    boolean selected, boolean expanded, boolean leaf, int row) {

            Component rendererComponent = renderer.getTreeCellRendererComponent(
                    tree, value, true, expanded, leaf, row, true);


            ItemListener itemListener = new ItemListener() {

                public void itemStateChanged(ItemEvent itemEvent) {
                    if (stopCellEditing()) {
                        fireEditingStopped();
                    }
                }
            };
            if (rendererComponent instanceof LinkCheckBox) {
                ((LinkCheckBox) rendererComponent).addItemListener(itemListener);
            }

            return rendererComponent;
        }
    }

    public static class CheckableResource implements SelectableResource {

        final static Color partialSelectionColor =
                new Color(255, 128, 128);
        final boolean isParentOfPartiallySelectedChildren = false;
        final ResourceLocator dataResourceLocator;
        String text;
        boolean selected;
        boolean isEnabled = true;

        CheckableResource(String text, boolean selected,
                          ResourceLocator dataResourceLocator) {

            this.text = text;
            this.selected = selected;
            this.dataResourceLocator = dataResourceLocator;
        }

        public boolean isSelected() {
            return selected;
        }

        public void setSelected(boolean newValue) {
            selected = newValue;
        }

        public boolean isEnabled() {
            return isEnabled;
        }

        public void setEnabled(boolean value) {
            isEnabled = value;
        }

        public String getText() {
            return text;
        }

        public void setText(String newValue) {
            text = newValue;
        }

        public ResourceLocator getResourceLocator() {
            return dataResourceLocator;
        }

        public boolean isParentOfPartiallySelectedChildren() {
            return isParentOfPartiallySelectedChildren;
        }

        @Override
        public String toString() {
            return text + ":" + selected;
        }
    }
}
