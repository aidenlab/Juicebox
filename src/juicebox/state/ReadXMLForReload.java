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

/**
 * Created by Zulkifl on 7/20/2015.
 */

import juicebox.HiC;
import juicebox.MainWindow;
import org.apache.commons.math.stat.descriptive.rank.Min;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.ErrorHandler;
import org.xml.sax.InputSource;
import org.xml.sax.SAXException;
import org.xml.sax.SAXParseException;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import java.io.IOException;

public class ReadXMLForReload {

    private final HiC hic;

    private String mapPath = "MapPath";
    private String Map = "Map";
    private String MapURL = "MapURL";
    private String XChromosome = "XChromosome";
    private String YChromosome = "YChromosome";
    private String UnitName = "UnitName";
    private String BinSize = "BinSize";
    private String xOrigin = "xOrigin";
    private String yOrigin = "yOrigin";
    private String ScaleFactor = "ScaleFactor";
    private String DisplayOption = "DisplayOption";
    private String NormalizationType = "NormalizationType";
    private String MinColorVal = "MinColorVal";
    private String LowerColorVal = "LowerColorVal";
    private String UpperColorVal = "UpperColorVal";
    private String MaxColorVal = "MaxColorVal";
    private String LoadedTrackURLS = "LoadedTrackURLS";
    private String LoadedTrackNames = "LoadedTrackNames";
    private String[] infoForReload;

    public ReadXMLForReload(HiC hic){this.hic = hic;}


    public boolean readXML(String xml, String mapSelection) {
        infoForReload = new String[18];
        Document dom;
        DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
        try {
            DocumentBuilder db = dbf.newDocumentBuilder();
            dom = db.parse(xml);
            //validate
            dbf.setValidating(false);
            dbf.setNamespaceAware(true);
            DocumentBuilder builder = dbf.newDocumentBuilder();
            builder.setErrorHandler(new SimpleErrorHandler());
            Document document = builder.parse(new InputSource(xml));
            Element element = dom.getDocumentElement();
            NodeList nodeList = dom.getElementsByTagName("STATE");
            for (int i = 0; i < nodeList.getLength(); i++) {
                System.out.println(nodeList.item(i).getAttributes().getNamedItem("SelectedPath").getNodeValue() +" mapPath: "+mapSelection);
                if (nodeList.item(i).getAttributes().getNamedItem("SelectedPath").getNodeValue().equals(mapSelection)) {
                    NodeList childNodes = nodeList.item(i).getChildNodes();
                    for(int c=1; c<childNodes.getLength(); c+=2) {

                        if(childNodes.item(c).getNodeName().equals(mapPath)) {
                            if (mapPath != null) {
                                if (!mapPath.isEmpty())
                                    infoForReload[0] = childNodes.item(c).getTextContent();
                            }
                        }

                        if(childNodes.item(c).getNodeName().equals(Map)) {
                            if (Map != null) {
                                if (!Map.isEmpty())
                                    infoForReload[1] = childNodes.item(c).getTextContent();
                            }
                        }

                        if(childNodes.item(c).getNodeName().equals(MapURL)) {
                            if (MapURL != null) {
                                if (!MapURL.isEmpty())
                                    infoForReload[2] = childNodes.item(c).getTextContent();
                            }
                        }

                        if(childNodes.item(c).getNodeName().equals(XChromosome)) {
                            if (XChromosome != null) {
                                if (!XChromosome.isEmpty())
                                    infoForReload[3] = childNodes.item(c).getTextContent();
                            }
                        }

                        if(childNodes.item(c).getNodeName().equals(YChromosome)) {
                            if (YChromosome != null) {
                                if (!YChromosome.isEmpty())
                                    infoForReload[4] = childNodes.item(c).getTextContent();
                            }
                        }

                        if(childNodes.item(c).getNodeName().equals(UnitName)) {
                            if (UnitName != null) {
                                if (!UnitName.isEmpty())
                                    infoForReload[5] = childNodes.item(c).getTextContent();
                            }
                        }

                        if(childNodes.item(c).getNodeName().equals(BinSize)) {
                            if (BinSize != null) {
                                if (!BinSize.isEmpty())
                                    infoForReload[6] = childNodes.item(c).getTextContent();
                            }
                        }

                        if(childNodes.item(c).getNodeName().equals(xOrigin)) {
                            if (xOrigin != null) {
                                if (!xOrigin.isEmpty())
                                    infoForReload[7] = childNodes.item(c).getTextContent();
                            }
                        }

                        if(childNodes.item(c).getNodeName().equals(yOrigin)) {
                            if (yOrigin != null) {
                                if (!yOrigin.isEmpty())
                                    infoForReload[8] = childNodes.item(c).getTextContent();
                            }
                        }

                        if(childNodes.item(c).getNodeName().equals(ScaleFactor)) {
                            if (ScaleFactor != null) {
                                if (!ScaleFactor.isEmpty())
                                    infoForReload[9] = childNodes.item(c).getTextContent();
                            }
                        }

                        if(childNodes.item(c).getNodeName().equals(DisplayOption)) {
                            if (DisplayOption != null) {
                                if (!DisplayOption.isEmpty())
                                    infoForReload[10] = childNodes.item(c).getTextContent();
                            }
                        }

                        if(childNodes.item(c).getNodeName().equals(NormalizationType)) {
                            if (NormalizationType != null) {
                                if (!NormalizationType.isEmpty())
                                    infoForReload[11] = childNodes.item(c).getTextContent();
                            }
                        }

                        if(childNodes.item(c).getNodeName().equals(MinColorVal)) {
                            if (MinColorVal != null) {
                                if (!MinColorVal.isEmpty())
                                    infoForReload[12] = childNodes.item(c).getTextContent();
                            }
                        }

                        if(childNodes.item(c).getNodeName().equals(LowerColorVal)) {
                            if (LowerColorVal != null) {
                                if (!LowerColorVal.isEmpty())
                                    infoForReload[13] = childNodes.item(c).getTextContent();
                            }
                        }

                        if(childNodes.item(c).getNodeName().equals(UpperColorVal)) {
                            if (UpperColorVal != null) {
                                if (!UpperColorVal.isEmpty())
                                    infoForReload[14] = childNodes.item(c).getTextContent();
                            }
                        }

                        if(childNodes.item(c).getNodeName().equals(MaxColorVal)) {
                            if (MaxColorVal != null) {
                                if (!MaxColorVal.isEmpty())
                                    infoForReload[15] = childNodes.item(c).getTextContent();
                            }
                        }

                        if(childNodes.item(c).getNodeName().equals(LoadedTrackURLS)) {
                            if (LoadedTrackURLS != null) {
                                if (!LoadedTrackURLS.isEmpty())
                                    infoForReload[16] = childNodes.item(c).getTextContent();
                            }
                        }

                        if(childNodes.item(c).getNodeName().equals(LoadedTrackNames)) {
                            if (LoadedTrackNames != null) {
                                if (!LoadedTrackNames.isEmpty())
                                    infoForReload[17] = childNodes.item(c).getTextContent();
                            }
                        }
                    }
                }
            }

            hic.reloadPreviousStateFromXML(infoForReload);
            return true;

        } catch (ParserConfigurationException pce) {
            System.out.println(pce.getMessage());
        } catch (SAXException se) {
            System.out.println(se.getMessage());
        } catch (IOException ioe) {
            System.err.println(ioe.getMessage());
        }

        return false;
    }

    public static class SimpleErrorHandler implements ErrorHandler {
        public void warning(SAXParseException e) throws SAXException {
            System.out.println(e.getMessage());
        }

        public void error(SAXParseException e) throws SAXException {
            System.out.println(e.getMessage());
        }

        public void fatalError(SAXParseException e) throws SAXException {
            System.out.println(e.getMessage());
        }
    }

}

