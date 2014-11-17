/*
 * Copyright (c) 2007-2012 The Broad Institute, Inc.
 * SOFTWARE COPYRIGHT NOTICE
 * This software and its documentation are the copyright of the Broad Institute, Inc. All rights are reserved.
 *
 * This software is supplied without any warranty or guaranteed support whatsoever. The Broad Institute is not responsible for its use, misuse, or functionality.
 *
 * This software is licensed under the terms of the GNU Lesser General Public License (LGPL),
 * Version 2.1 which is available at http://www.opensource.org/licenses/lgpl-2.1.php.
 */
/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package juicebox.track;

//~--- non-JDK imports --------------------------------------------------------

import org.apache.log4j.Logger;
import org.broad.igv.PreferenceManager;
import juicebox.HiC;
import juicebox.MainWindow;
import juicebox.NormalizationType;
import org.broad.igv.ui.util.MessageUtils;
import org.broad.igv.util.ParsingUtils;
import org.broad.igv.util.ResourceLocator;
import org.broad.igv.util.Utilities;
import org.w3c.dom.*;
import org.xml.sax.SAXException;

import javax.swing.*;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.*;
import java.util.List;

/**
 * @author jrobinso
 */
public class LoadAction extends AbstractAction {

    static Logger log = Logger.getLogger(LoadAction.class);

    private JFrame owner;
    private HiC hic;


    public LoadAction(String s, JFrame owner, HiC hic) {
        super(s);
        this.owner = owner;
        this.hic = hic;
    }

    @Override
    public void actionPerformed(ActionEvent evt) {
        if (hic.getDataset() == null) {
            JOptionPane.showMessageDialog(owner, "File must be loaded to load annotations", "Error", JOptionPane.ERROR_MESSAGE);
            return;
        }
        String genome = hic.getDataset().getGenomeId();
        if (genome == null) {
            genome = "hg19";
        }

        String xmlURL = "tracksMenu_" + genome + ".xml";

        List<ResourceLocator> locators = loadNodes(xmlURL);
        if (locators != null && !locators.isEmpty()) {
            hic.loadHostedTracks(locators);
        }

    }

    private List<ResourceLocator> loadNodes(String xmlFile) {

        ResourceTree resourceTree = hic.getResourceTree();

        try {
            if (resourceTree == null) {
                Document masterDocument = createMasterDocument(xmlFile);
                resourceTree = new ResourceTree(hic, masterDocument);
            }
        } catch (Exception e) {
            log.error("Could not load from server", e);
            MessageUtils.showMessage("Could not load from server: " + e.getMessage());
            return null;
        }

        resourceTree.showResourceTreeDialog(owner);

        LinkedHashSet<ResourceLocator> selectedLocators = resourceTree.getLocators();
        LinkedHashSet<ResourceLocator> deselectedLocators = resourceTree.getDeselectedLocators();
        List<ResourceLocator> newLoadList = new ArrayList<ResourceLocator>();

        boolean repaint = false;

        if(selectedLocators != null) {
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

        if(deselectedLocators != null) {
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

        if (repaint) owner.repaint();
        return newLoadList;
    }


    public static Document createMasterDocument(String xmlUrl) throws ParserConfigurationException {

        StringBuffer buffer = new StringBuffer();

        Document masterDocument = DocumentBuilderFactory.newInstance().newDocumentBuilder().newDocument();

        try {
            Document xmlDocument = readXMLDocument(xmlUrl, buffer);

            if (xmlDocument != null) {
                Element global = xmlDocument.getDocumentElement();
                masterDocument.appendChild(masterDocument.importNode(global, true));
            }
        } catch (Exception e) {
            String message = "Cannot create an XML Document from " + xmlUrl;
            log.error(message, e);

        }

        if (buffer.length() > 0) {
            String message = "<html>The following urls could not be processed due to load failures:<br>" + buffer.toString();
            JOptionPane.showMessageDialog(MainWindow.getInstance(), message);
        }

        return masterDocument;

    }

    private static Document readXMLDocument(String url, StringBuffer errors) {
        InputStream is = null;
        Document xmlDocument = null;
        try {
            is = LoadAction.class.getResourceAsStream(url);
            xmlDocument = Utilities.createDOMDocumentFromXmlStream(is);

            xmlDocument = resolveIncludes(xmlDocument, errors);

        } catch (SAXException e) {
            log.error("Invalid XML resource: " + url, e);
            errors.append(url + "<br><i>" + e.getMessage());
        } catch (java.net.SocketTimeoutException e) {
            log.error("Connection time out", e);
            errors.append(url + "<br><i>Connection time out");
        } catch (IOException e) {
            log.error("Error accessing " + url.toString(), e);
            errors.append(url + "<br><i>" + e.getMessage());
        } catch (ParserConfigurationException e) {
            log.error("Parser configuration error for:" + url, e);
            errors.append(url + "<br><i>" + e.getMessage());
        } finally {
            if (is != null) {
                try {
                    is.close();
                } catch (IOException e) {
                    log.error("Error closing stream for: " + url, e);
                }
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

}
