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
import juicebox.MainWindow;
import juicebox.gui.SuperAdapter;
import org.broad.igv.util.ResourceLocator;
import org.broad.igv.util.Utilities;
import org.w3c.dom.*;
import org.xml.sax.SAXException;

import javax.swing.*;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import java.awt.event.ActionEvent;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;


/**
 * @author jrobinso
 */
public class LoadAction extends AbstractAction {

    private static final long serialVersionUID = -1122795124141741145L;
    private final JFrame parentFrame;
    private final MainWindow mainWindow;
    private final HiC hic;
    private Runnable repaint1DLayersPanel = null;


    public LoadAction(String s, MainWindow mainWindow, HiC hic, Runnable repaint1DLayersPanel) {
        super(s);
        this.parentFrame = mainWindow;
        this.mainWindow = mainWindow;
        this.hic = hic;
        this.repaint1DLayersPanel = repaint1DLayersPanel;
    }

    private static Document createMasterDocument(String xmlUrl, JFrame parentFrame) throws ParserConfigurationException {

        StringBuffer buffer = new StringBuffer();

        Document masterDocument = DocumentBuilderFactory.newInstance().newDocumentBuilder().newDocument();

        try {
            Document xmlDocument = readXMLDocument(xmlUrl, buffer);

            if (xmlDocument != null) {
                Element global = xmlDocument.getDocumentElement();
                masterDocument.appendChild(masterDocument.importNode(global, true));
            } else {
                masterDocument = null;
            }
        } catch (Exception e) {
            String message = "Cannot create an XML Document from " + xmlUrl;
            System.err.println(message + " " + e.getLocalizedMessage());
        }

        if (buffer.length() > 0) {
            String message = "<html>The following urls could not be processed due to load failures:<br>" + buffer.toString();
            JOptionPane.showMessageDialog(parentFrame, message);
        }

        return masterDocument;

    }

    private static Document readXMLDocument(String url, StringBuffer errors) {
        InputStream is;
        Document xmlDocument = null;
        is = LoadAction.class.getResourceAsStream(url);
        if (is == null) {
            System.err.println(url + " doesn't exist, so cannot read default annotations");
            return null;
        }
        try {
            xmlDocument = Utilities.createDOMDocumentFromXmlStream(is);

            xmlDocument = resolveIncludes(xmlDocument, errors);

        } catch (SAXException e) {
            System.err.println("Invalid XML resource: " + url + " " + e.getLocalizedMessage());
            errors.append(url).append("<br><i>").append(e.getMessage());
        } catch (java.net.SocketTimeoutException e) {
            System.err.println("Connection time out " + e.getLocalizedMessage());
            errors.append(url).append("<br><i>Connection time out");
        } catch (IOException e) {
            System.err.println("Error accessing " + url + " " + e.getLocalizedMessage());
            errors.append(url).append("<br><i>").append(e.getMessage());
        } catch (ParserConfigurationException e) {
            System.err.println("Parser configuration error for:" + url + " " + e.getLocalizedMessage());
            errors.append(url).append("<br><i>").append(e.getMessage());
        } finally {
            try {
                is.close();
            } catch (IOException e) {
                System.err.println("Error closing stream for: " + url + " " + e.getLocalizedMessage());
            }
        }
        return xmlDocument;
    }

    private static Document resolveIncludes(Document document, StringBuffer errors) {

        NodeList includeNodes = document.getElementsByTagName("Include");
        if (includeNodes.getLength() == 0) {
            return document;
        }

        int size = includeNodes.getLength();
        // Copy the nodes as we'll be modifying the tree.  This is neccessary!
        Node[] tmp = new Node[size];
        for (int i = 0; i < size; i++) {
            tmp[i] = includeNodes.item(i);
        }

        for (Node item : tmp) {
            NamedNodeMap nodeMap = item.getAttributes();
            if (nodeMap == null) {
                System.out.println("XML node " + item.getNodeName() + " has no attributes");
            } else {
                Attr path = (Attr) item.getAttributes().getNamedItem("path");
                if (path == null) {
                    System.out.println("XML node " + item.getNodeName() + " is missing a path attribute");
                } else {
                    Node parent = item.getParentNode();

                    //System.out.println("Loading node " + path.getValue());
                    Document doc = readXMLDocument(path.getValue(), errors);
                    if (doc != null) {
                        Element global = doc.getDocumentElement();
                        Node expandedNode = parent.getOwnerDocument().importNode(global, true);
                        parent.replaceChild(expandedNode, item);
                    }
                }
            }
        }


        return document;

    }

    private String getXmlUrl() {
        String genome = hic.getDataset().getGenomeId();
        if (genome == null) {
            genome = "hg19";
        }

        return "tracksMenu_" + genome + ".xml";
    }

    @Override
    public void actionPerformed(ActionEvent evt) {
        if (hic.getDataset() == null) {
            JOptionPane.showMessageDialog(parentFrame, "File must be loaded to load annotations",
                    "Error", JOptionPane.ERROR_MESSAGE);
            return;
        }
        String genome = hic.getDataset().getGenomeId();
        if (genome == null) {
            genome = "hg19";
        }

        String xmlURL = "tracksMenu_" + genome + ".xml";
        safeLoadNodes(xmlURL);
    }

    private void safeLoadNodes(final String xmlFile) {
        Runnable runnable = new Runnable() {
            @Override
            public void run() {
                List<ResourceLocator> locators = unsafeLoadNodes(xmlFile);
                if (locators != null && !locators.isEmpty()) {
                    // TODO MSS
                    hic.unsafeLoadHostedTracks(locators);
                }
                if (repaint1DLayersPanel != null) {
                    repaint1DLayersPanel.run();
                }
            }
        };
        mainWindow.executeLongRunningTask(runnable, "safe load nodes");
    }

    public void checkBoxesForReload(String track) {
        // TODO MSS
        ResourceTree resourceTree = hic.getResourceTree();
        try {
            if (resourceTree == null) {
                Document tempDoc = createMasterDocument(getXmlUrl(), parentFrame);
                resourceTree = new ResourceTree(hic, tempDoc);
                resourceTree.checkTrackBoxesForReloadState(track.trim());
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        resourceTree.checkTrackBoxesForReloadState(track);
    }

    private List<ResourceLocator> unsafeLoadNodes(String xmlFile) {

        // TODO MSS
        ResourceTree resourceTree = hic.getResourceTree();

        try {
            if (resourceTree == null) {
                Document masterDocument = createMasterDocument(xmlFile, parentFrame);
                resourceTree = new ResourceTree(hic, masterDocument);
            }
        } catch (Exception e) {
            System.err.println("Could not load from server" + e.getLocalizedMessage());
            SuperAdapter.showMessageDialog("Could not load from server: " + e.getMessage());
            return null;
        }

        resourceTree.showResourceTreeDialog(parentFrame);

        LinkedHashSet<ResourceLocator> selectedLocators = resourceTree.getLocators();
        LinkedHashSet<ResourceLocator> deselectedLocators = resourceTree.getDeselectedLocators();
        List<ResourceLocator> newLoadList = new ArrayList<>();

        boolean repaint = false;

        if (selectedLocators != null) {
            for (ResourceLocator locator : selectedLocators) {
                try {

                    if (locator.getType() != null && locator.getType().equals("norm")) {
                        hic.loadCoverageTrack(locator.getPath());
                    } else if (locator.getType() != null && locator.getType().equals("eigenvector")) {
                        hic.loadEigenvectorTrack();

                    } else newLoadList.add(locator);

                } catch (Exception e) {
                    System.err.println("Could not load selected locator" + e.getLocalizedMessage());
                    SuperAdapter.showMessageDialog("Could not load selection: " + e.getMessage());
                    deselectedLocators.add(locator);
                }
            }
        }

        if (deselectedLocators != null) {
            for (ResourceLocator locator : deselectedLocators) {
                System.out.println("Removing " + locator.getName());
                hic.removeTrack(locator);
                resourceTree.remove(locator);
            }
        }
        if (repaint) {
            mainWindow.repaint();
        }
        //hic.setShowLoops(true);
        return newLoadList;
    }

}
