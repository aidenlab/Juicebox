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

package juicebox.state;

import juicebox.HiCGlobals;
import org.w3c.dom.Document;
import org.w3c.dom.NodeList;
import org.xml.sax.ErrorHandler;
import org.xml.sax.InputSource;
import org.xml.sax.SAXException;
import org.xml.sax.SAXParseException;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

/**
 * Created by Zulkifl on 7/20/2015.
 */
class XMLFileParser {

    public static String[] parseXML(String mapSelection) {
        String[] infoForReload = new String[21];
        Document dom;
        DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
        try {
            DocumentBuilder db = dbf.newDocumentBuilder();
            dom = db.parse(HiCGlobals.xmlSavedStatesFileName);

            //validate
            dbf.setValidating(false);
            dbf.setNamespaceAware(true);
            DocumentBuilder builder = dbf.newDocumentBuilder();
            builder.setErrorHandler(new SimpleErrorHandler());
            builder.parse(new InputSource(HiCGlobals.xmlSavedStatesFileName));
            dom.getDocumentElement();


            NodeList nodeList = dom.getElementsByTagName("STATE");
            for (int i = 0; i < nodeList.getLength(); i++) {
                //System.out.println(nodeList.item(i).getAttributes().getNamedItem("SelectedPath").getNodeValue() +" mapPath: "+mapSelection);
                if (nodeList.item(i).getAttributes().getNamedItem("SelectedPath").getNodeValue().equals(mapSelection)) {

                    NodeList childNodes = nodeList.item(i).getChildNodes();
                    // +=2 because need to skip 2nd line which describes 1st line (but everything is type text)
                    for (int c = 1; c < childNodes.getLength(); c += 2) {

                        String varName = childNodes.item(c).getNodeName();
                        String varText = childNodes.item(c).getTextContent();

                        for (int k = 0; k < State.stateVarNames.length; k++) {
                            if (varName.equals(State.stateVarNames[k])) {
                                infoForReload[k] = varText;
                                break;
                            }
                        }
                    }
                    break;
                }
            }

        } catch (Exception e) {
            e.printStackTrace();
        }

        return infoForReload;
    }

    private static class SimpleErrorHandler implements ErrorHandler {
        public void warning(SAXParseException e) throws SAXException {
            System.err.println(e.getMessage());
        }

        public void error(SAXParseException e) throws SAXException {
            System.err.println(e.getMessage());
        }

        public void fatalError(SAXParseException e) throws SAXException {
            System.err.println(e.getMessage());
        }
    }
}
