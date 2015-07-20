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

package juicebox;

import java.io.*;

import org.w3c.dom.DOMImplementation;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;

import javax.xml.parsers.*;
import javax.xml.transform.*;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.*;
import javax.xml.transform.sax.*;


/**
 * Created by Zulkifl Gire on 7/15/2015.
 */
public class XMLForReloadState {


    BufferedReader bufferedReader;
    StreamResult streamResult;
    TransformerHandler transformerHandler;
    File currentStates = new File(HiCGlobals.stateFileName);
    File currentStatesToXML = new File(HiCGlobals.xmlFileName);
    Document xmlDoc;
    Element root;

    public static void main (String args[]) {
        new XMLForReloadState().begin();
    }

    public void begin() {
        try {
            bufferedReader = new BufferedReader(new FileReader(currentStates));
            streamResult = new StreamResult(currentStatesToXML);
            //openXml();
            initXML();
            String str;
            String temp = bufferedReader.readLine();
            while ((str = bufferedReader.readLine()) != null) {
                convert(str);
            }
            bufferedReader.close();
            writeXML();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /*public void openXml() throws ParserConfigurationException, TransformerConfigurationException, SAXException {

        SAXTransformerFactory tf = (SAXTransformerFactory) SAXTransformerFactory.newInstance();
        transformerHandler = tf.newTransformerHandler();

        // pretty XML output
        Transformer serializer = transformerHandler.getTransformer();
        serializer.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "4");
        serializer.setOutputProperty(OutputKeys.INDENT, "yes");

        transformerHandler.setResult(streamResult);
        transformerHandler.startDocument();
        transformerHandler.startElement(null, null, "Saved HiC States", null);
    }

    public void process(String s) throws SAXException {
        transformerHandler.startElement(null, null, "State", null);
        transformerHandler.characters(s.toCharArray(), 0, s.length());
        transformerHandler.endElement(null, null, "State");
    }

    public void closeXml() throws SAXException {
        transformerHandler.endElement(null, null, "Saved HiC States");
        transformerHandler.endDocument();
    }*/

    public void initXML() throws ParserConfigurationException{
        // JAXP + DOM
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        DocumentBuilder builder = factory.newDocumentBuilder();
        DOMImplementation impl = builder.getDOMImplementation();

        xmlDoc = impl.createDocument(null, "SavedMaps", null);
        root = xmlDoc.getDocumentElement();
    }

    public void convert(String s) {
        // Since the separator character "|" has special meaning
        // with regular expression, we need to escape it.
        String [] elements = s.split("\\$\\$");
        String [] title = elements[0].split("--");
        Element e0 = xmlDoc.createElement("STATE");

        Element eMP = xmlDoc.createElement("MapPath");
        Node nMP = xmlDoc.createTextNode(title[0]);
        eMP.appendChild(nMP);

        Element e1 = xmlDoc.createElement("Map");
        Node n1 = xmlDoc.createTextNode(elements[1]);
        e1.appendChild(n1);

        Element e2 = xmlDoc.createElement("XChromosome");
        Node  n2 = xmlDoc.createTextNode(elements[2]);
        e2.appendChild(n2);

        Element e3 = xmlDoc.createElement("YChromosome");
        Node  n3 = xmlDoc.createTextNode(elements[3]);
        e3.appendChild(n3);

        Element e4 = xmlDoc.createElement("UnitName");
        Node  n4 = xmlDoc.createTextNode(elements[4]);
        e4.appendChild(n4);

        Element e5 = xmlDoc.createElement("BinSize");
        Node  n5 = xmlDoc.createTextNode(elements[5]);
        e5.appendChild(n5);

        Element e6 = xmlDoc.createElement("xOrigin");
        Node  n6 = xmlDoc.createTextNode(elements[6]);
        e6.appendChild(n6);

        Element e7 = xmlDoc.createElement("yOrigin");
        Node  n7 = xmlDoc.createTextNode(elements[7]);
        e7.appendChild(n7);

        Element e8 = xmlDoc.createElement("ScaleFactor");
        Node  n8 = xmlDoc.createTextNode(elements[8]);
        e8.appendChild(n8);

        Element e9 = xmlDoc.createElement("DisplayOption");
        Node  n9 = xmlDoc.createTextNode(elements[9]);
        e9.appendChild(n9);

        Element e10 = xmlDoc.createElement("NormalizationType");
        Node  n10 = xmlDoc.createTextNode(elements[10]);
        e10.appendChild(n10);

        Element e11 = xmlDoc.createElement("MinColorVal");
        Node  n11 = xmlDoc.createTextNode(elements[11]);
        e11.appendChild(n11);

        Element e12 = xmlDoc.createElement("LowerColorVal");
        Node  n12 = xmlDoc.createTextNode(elements[12]);
        e12.appendChild(n12);

        Element e13 = xmlDoc.createElement("UpperColorVal");
        Node  n13 = xmlDoc.createTextNode(elements[13]);
        e13.appendChild(n13);

        Element e14 = xmlDoc.createElement("MaxColorVal");
        Node  n14 = xmlDoc.createTextNode(elements[14]);
        e14.appendChild(n14);

        String tracks = "";

        for(int i = 15; i<elements.length; i++){
            tracks+=elements[i]+"$$";
        }

        Element e15 = xmlDoc.createElement("LoadedTracks");
        Node  n15 = xmlDoc.createTextNode(tracks);
        e15.appendChild(n15);

        e0.appendChild(eMP);
        e0.appendChild(e1);
        e0.appendChild(e2);
        e0.appendChild(e3);
        e0.appendChild(e4);
        e0.appendChild(e5);
        e0.appendChild(e6);
        e0.appendChild(e7);
        e0.appendChild(e8);
        e0.appendChild(e9);
        e0.appendChild(e10);
        e0.appendChild(e11);
        e0.appendChild(e12);
        e0.appendChild(e13);
        e0.appendChild(e14);
        e0.appendChild(e15);
        root.appendChild(e0);
    }

    public void writeXML() throws TransformerConfigurationException,
            TransformerException {
        DOMSource domSource = new DOMSource(xmlDoc);
        TransformerFactory tf = TransformerFactory.newInstance();
        Transformer transformer = tf.newTransformer();
        //transformer.setOutputProperty(OutputKeys.OMIT_XML_DECLARATION, "yes");
        transformer.setOutputProperty(OutputKeys.METHOD, "xml");
        transformer.setOutputProperty(OutputKeys.ENCODING,"ISO-8859-1");
        // we want to pretty format the XML output
        // note : this is broken in jdk1.5 beta!
        transformer.setOutputProperty
                ("{http://xml.apache.org/xslt}indent-amount", "4");
        transformer.setOutputProperty(OutputKeys.INDENT, "yes");
        //
        transformer.transform(domSource, streamResult);

    }

}
