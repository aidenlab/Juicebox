/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2016 Broad Institute, Aiden Lab
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

package juicebox.state;

import juicebox.HiCGlobals;
import org.w3c.dom.DOMImplementation;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

/**
 * Created by Zulkifl Gire on 7/15/2015.
 */
public class XMLFileWriter {

    private static StreamResult streamResult;
    private static Document xmlDoc;

    public static void overwriteXMLFile() {
        try {
            streamResult = new StreamResult(HiCGlobals.xmlSavedStatesFile);
            Element root = initXML();

            for (String stateString : HiCGlobals.savedStatesList) {
                convert(stateString, root);
            }

            writeXML();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    private static Element initXML() throws ParserConfigurationException {

        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        DocumentBuilder builder = factory.newDocumentBuilder();
        DOMImplementation impl = builder.getDOMImplementation();

        xmlDoc = impl.createDocument(null, "SavedMaps", null);
        return xmlDoc.getDocumentElement();
    }

    private static void convert(String s, Element root) {

        String[] elements = s.split("\\$\\$");
        String[] title = elements[0].split("--");
        String[] mapName = elements[1].split("\\@\\@");
        Element e0 = xmlDoc.createElement("STATE");
        e0.setAttribute("SelectedPath", title[0]);

        for (int i = 0; i < State.stateVarNames.length; i++) {
            Element e = xmlDoc.createElement(State.stateVarNames[i]);
            Node n = xmlDoc.createTextNode("none");

            if (i == 0) {//mappath or id
                n = xmlDoc.createTextNode(title[0]);
                //System.out.println(title[0]);
            } else if (i == 1) {//mapname
                n = xmlDoc.createTextNode(mapName[0]);
                //System.out.println(mapName[0]);
            } else if (i == 2) {//mapurl
                n = xmlDoc.createTextNode(mapName[1]);
                //System.out.println(mapName[1]);
            } else if (i == 3) {//controlURL
                if (mapName.length > 2 && !mapName[2].contains("null") && mapName[2].length() > 1) {
                    n = xmlDoc.createTextNode(mapName[2]);
                    //System.out.println(mapName[2]);
                } else {
                    n = xmlDoc.createTextNode("null");
                }
            } else if (i < 19 || elements.length > 18) { // elements.length checked in case no annotations
                n = xmlDoc.createTextNode(elements[i - 2]);
                //System.out.println(elements[i-2]);
            }

            e.appendChild(n);
            e0.appendChild(e);
        }

        root.appendChild(e0);
    }

    private static void writeXML() throws TransformerException {
        DOMSource domSource = new DOMSource(xmlDoc);
        TransformerFactory tf = TransformerFactory.newInstance();
        Transformer transformer = tf.newTransformer();
        transformer.setOutputProperty(OutputKeys.METHOD, "xml");
        transformer.setOutputProperty(OutputKeys.ENCODING, "ISO-8859-1");
        transformer.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "4");
        transformer.setOutputProperty(OutputKeys.INDENT, "yes");
        transformer.transform(domSource, streamResult);
    }
}