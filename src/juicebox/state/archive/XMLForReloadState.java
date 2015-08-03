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

package juicebox.state.archive;

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
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;


/**
 * Created by Zulkifl Gire on 7/15/2015.
 */
public class XMLForReloadState {


    private final File JuiceboxStatesXML = new File("JuiceboxStatesXML.txt");
    private final File currentStatesToXML = new File(HiCGlobals.xmlFileName);
    File currentStates = new File(HiCGlobals.stateFileName);
    private BufferedReader bufferedReader;
    private StreamResult streamResult;
    private Document xmlDoc;
    private Element root;

    public static void main (String args[]) {
        new XMLForReloadState().begin();
    }

    public void begin() {
        try {
            bufferedReader = new BufferedReader(new FileReader(JuiceboxStatesXML));
            streamResult = new StreamResult(currentStatesToXML);
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


    private void initXML() throws ParserConfigurationException {

        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        DocumentBuilder builder = factory.newDocumentBuilder();
        DOMImplementation impl = builder.getDOMImplementation();

        xmlDoc = impl.createDocument(null, "SavedMaps", null);
        root = xmlDoc.getDocumentElement();
    }

    private void convert(String s) {

        String[] elements = s.split("\\$\\$");
        String[] title = elements[0].split("--");
        String[] mapName = elements[1].split("\\@\\@");
        Element e0 = xmlDoc.createElement("STATE");
        e0.setAttribute("SelectedPath", title[0]);

        Element eMapPath = xmlDoc.createElement("MapPath");
        Node nMP = xmlDoc.createTextNode(title[0]);
        eMapPath.appendChild(nMP);

        Element eMap = xmlDoc.createElement("Map");
        Node n1a = xmlDoc.createTextNode(mapName[0]);
        eMap.appendChild(n1a);

        Element eMapURL = xmlDoc.createElement("MapURL");
        Node n1b = xmlDoc.createTextNode(mapName[1]);
        eMapURL.appendChild(n1b);

        Element XChromosome = xmlDoc.createElement("XChromosome");
        Node  n2 = xmlDoc.createTextNode(elements[2]);
        XChromosome.appendChild(n2);

        Element eYChromosome = xmlDoc.createElement("YChromosome");
        Node  n3 = xmlDoc.createTextNode(elements[3]);
        eYChromosome.appendChild(n3);

        Element eUnitName = xmlDoc.createElement("UnitName");
        Node  n4 = xmlDoc.createTextNode(elements[4]);
        eUnitName.appendChild(n4);

        Element eBinSize = xmlDoc.createElement("BinSize");
        Node  n5 = xmlDoc.createTextNode(elements[5]);
        eBinSize.appendChild(n5);

        Element eXOrigin = xmlDoc.createElement("xOrigin");
        Node  n6 = xmlDoc.createTextNode(elements[6]);
        eXOrigin.appendChild(n6);

        Element eYOrigin = xmlDoc.createElement("yOrigin");
        Node  n7 = xmlDoc.createTextNode(elements[7]);
        eYOrigin.appendChild(n7);

        Element eScaleFactor = xmlDoc.createElement("ScaleFactor");
        Node  n8 = xmlDoc.createTextNode(elements[8]);
        eScaleFactor.appendChild(n8);

        Element eDisplayOption = xmlDoc.createElement("DisplayOption");
        Node  n9 = xmlDoc.createTextNode(elements[9]);
        eDisplayOption.appendChild(n9);

        Element eNormalizationType = xmlDoc.createElement("NormalizationType");
        Node  n10 = xmlDoc.createTextNode(elements[10]);
        eNormalizationType.appendChild(n10);

        Element eMinColorVal = xmlDoc.createElement("MinColorVal");
        Node  n11 = xmlDoc.createTextNode(elements[11]);
        eMinColorVal.appendChild(n11);

        Element eLowerColorVal = xmlDoc.createElement("LowerColorVal");
        Node  n12 = xmlDoc.createTextNode(elements[12]);
        eLowerColorVal.appendChild(n12);

        Element eUpperColorVal = xmlDoc.createElement("UpperColorVal");
        Node  n13 = xmlDoc.createTextNode(elements[13]);
        eUpperColorVal.appendChild(n13);

        Element eMaxColorVal = xmlDoc.createElement("MaxColorVal");
        Node  n14 = xmlDoc.createTextNode(elements[14]);
        eMaxColorVal.appendChild(n14);

        e0.appendChild(eMapPath);
        e0.appendChild(eMap);
        e0.appendChild(eMapURL);
        e0.appendChild(XChromosome);
        e0.appendChild(eYChromosome);
        e0.appendChild(eUnitName);
        e0.appendChild(eBinSize);
        e0.appendChild(eXOrigin);
        e0.appendChild(eYOrigin);
        e0.appendChild(eScaleFactor);
        e0.appendChild(eDisplayOption);
        e0.appendChild(eNormalizationType);
        e0.appendChild(eMinColorVal);
        e0.appendChild(eLowerColorVal);
        e0.appendChild(eUpperColorVal);
        e0.appendChild(eMaxColorVal);

        //If tracks are loaded
        if(elements.length >15 ) {
            Element eLoadedTrackURLS = xmlDoc.createElement("LoadedTrackURLS");
            Node n15 = xmlDoc.createTextNode(elements[15]);
            eLoadedTrackURLS.appendChild(n15);

            Element eLoadedTrackNames = xmlDoc.createElement("LoadedTrackNames");
            Node n16 = xmlDoc.createTextNode(elements[16]);
            eLoadedTrackNames.appendChild(n16);

            e0.appendChild(eLoadedTrackURLS);
            e0.appendChild(eLoadedTrackNames);
        }
        else{
            Element eLoadedTrackURLS = xmlDoc.createElement("LoadedTrackURLS");
            Node n15 = xmlDoc.createTextNode("none");
            eLoadedTrackURLS.appendChild(n15);

            Element eLoadedTrackNames = xmlDoc.createElement("LoadedTrackNames");
            Node n16 = xmlDoc.createTextNode("none");
            eLoadedTrackNames.appendChild(n16);

            e0.appendChild(eLoadedTrackURLS);
            e0.appendChild(eLoadedTrackNames);
        }

        root.appendChild(e0);
    }

    private void writeXML() throws TransformerException {
        DOMSource domSource = new DOMSource(xmlDoc);
        TransformerFactory tf = TransformerFactory.newInstance();
        Transformer transformer = tf.newTransformer();
        transformer.setOutputProperty(OutputKeys.METHOD, "xml");
        transformer.setOutputProperty(OutputKeys.ENCODING,"ISO-8859-1");
        transformer.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "4");
        transformer.setOutputProperty(OutputKeys.INDENT, "yes");
        transformer.transform(domSource, streamResult);

    }

}
