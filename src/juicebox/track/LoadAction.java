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

package juicebox.track;

//~--- non-JDK imports --------------------------------------------------------

import juicebox.HiC;
import juicebox.MainWindow;
import juicebox.windowui.NormalizationType;
import org.apache.log4j.Logger;
import org.broad.igv.ui.util.MessageUtils;
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
    private static final Logger log = Logger.getLogger(LoadAction.class);

    private final MainWindow mainWindow;
    private final HiC hic;


    public LoadAction(String s, MainWindow mainWindow, HiC hic) {
        super(s);
        this.mainWindow = mainWindow;
        this.hic = hic;
    }

    private static Document createMasterDocument(String xmlUrl, MainWindow mainWindow) throws ParserConfigurationException {

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
            log.error(message, e);
        }

        if (buffer.length() > 0) {
            String message = "<html>The following urls could not be processed due to load failures:<br>" + buffer.toString();
            JOptionPane.showMessageDialog(mainWindow, message);
        }

        return masterDocument;

    }

    private static Document readXMLDocument(String url, StringBuffer errors) {
        InputStream is = null;
        Document xmlDocument = null;
        is = LoadAction.class.getResourceAsStream(url);
        if (is == null) {
            log.error(url + " doesn't exist, so cannot read default annotations");
            return null;
        }
        try {
            xmlDocument = Utilities.createDOMDocumentFromXmlStream(is);

            xmlDocument = resolveIncludes(xmlDocument, errors);

        } catch (SAXException e) {
            log.error("Invalid XML resource: " + url, e);
            errors.append(url).append("<br><i>").append(e.getMessage());
        } catch (java.net.SocketTimeoutException e) {
            log.error("Connection time out", e);
            errors.append(url).append("<br><i>Connection time out");
        } catch (IOException e) {
            log.error("Error accessing " + url, e);
            errors.append(url).append("<br><i>").append(e.getMessage());
        } catch (ParserConfigurationException e) {
            log.error("Parser configuration error for:" + url, e);
            errors.append(url).append("<br><i>").append(e.getMessage());
        } finally {
            try {
                is.close();
            } catch (IOException e) {
                log.error("Error closing stream for: " + url, e);
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
                log.info("XML node " + item.getNodeName() + " has no attributes");
            } else {
                Attr path = (Attr) item.getAttributes().getNamedItem("path");
                if (path == null) {
                    log.info("XML node " + item.getNodeName() + " is missing a path attribute");
                } else {
                    Node parent = item.getParentNode();

                    //log.info("Loading node " + path.getValue());
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
            JOptionPane.showMessageDialog(mainWindow, "File must be loaded to load annotations", "Error", JOptionPane.ERROR_MESSAGE);
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
                    hic.loadHostedTracks(locators);
                }
            }
        };
        mainWindow.executeLongRunningTask(runnable, "safe load nodes");
    }

    public void checkBoxesForReload(String track) {
        ResourceTree resourceTree = hic.getResourceTree();
        try {
            if (resourceTree == null) {
                Document tempDoc = createMasterDocument(getXmlUrl(), mainWindow);
                resourceTree = new ResourceTree(hic, tempDoc);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        resourceTree.checkTrackBoxesForReloadState(track);
    }

    private List<ResourceLocator> unsafeLoadNodes(String xmlFile) {

        ResourceTree resourceTree = hic.getResourceTree();

        try {
            if (resourceTree == null) {
                Document masterDocument = createMasterDocument(xmlFile, mainWindow);
                resourceTree = new ResourceTree(hic, masterDocument);
            }
        } catch (Exception e) {
            log.error("Could not load from server", e);
            MessageUtils.showMessage("Could not load from server: " + e.getMessage());
            return null;
        }

        resourceTree.showResourceTreeDialog(mainWindow);

        LinkedHashSet<ResourceLocator> selectedLocators = resourceTree.getLocators();
        LinkedHashSet<ResourceLocator> deselectedLocators = resourceTree.getDeselectedLocators();
        List<ResourceLocator> newLoadList = new ArrayList<ResourceLocator>();

        boolean repaint = false;

        if (selectedLocators != null) {
            for (ResourceLocator locator : selectedLocators) {
                try {

                    if (locator.getType() != null && locator.getType().equals("norm")) {
                        NormalizationType option = null;
                        for (NormalizationType no : NormalizationType.values()) {
                            if (locator.getPath().equals(no.getLabel())) {
                                option = no;
                                break;
                            }
                        }
                        hic.loadCoverageTrack(option);
                    } else if (locator.getType() != null && locator.getType().equals("loop")) {
                        try {
                            hic.loadLoopList(locator.getPath());
                            repaint = true;
                        } catch (Exception e) {
                            log.error("Could not load selected loop locator", e);
                            MessageUtils.showMessage("Could not load loop selection: " + e.getMessage());
                            deselectedLocators.add(locator);
                        }

                    } else if (locator.getType() != null && locator.getType().equals("eigenvector")) {
                        hic.loadEigenvectorTrack();

                    } else newLoadList.add(locator);

                } catch (Exception e) {
                    log.error("Could not load selected locator", e);
                    MessageUtils.showMessage("Could not load selection: " + e.getMessage());
                    deselectedLocators.add(locator);
                }
            }
        }

        if (deselectedLocators != null) {
            for (ResourceLocator locator : deselectedLocators) {
                System.out.println("Removing " + locator.getName());
                hic.removeTrack(locator);
                resourceTree.remove(locator);

                if (locator.getType() != null && locator.getType().equals("loop")) {
                    try {
                        hic.setLoopsInvisible(locator.getPath());
                        repaint = true;
                    } catch (Exception e) {
                        log.error("Error while making loops invisible ", e);
                        MessageUtils.showMessage("Error while removing loops: " + e.getMessage());
                    }
                }
            }
        }
        if (repaint) {
            mainWindow.repaint();
        }
        hic.setShowLoops(true);
        return newLoadList;
    }

}
