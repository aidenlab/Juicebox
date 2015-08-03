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

import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.MainWindow;
import org.w3c.dom.Document;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import javax.swing.*;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.IOException;
import java.util.ArrayList;


/**
 * Created by Zulkifl on 7/31/2015.
 */


public class Slideshow {

    final static JFrame carouselFrame = new JFrame();
    final static JPanel nextPanel = new JPanel(new BorderLayout());
    final static JPanel prevPanel = new JPanel(new BorderLayout());
    final static JButton nextButton = new JButton("Next State");
    final static JButton prevButton = new JButton("Previous State");
    private static MainWindow mainWindow;
    private static String statesForSlideshow = HiCGlobals.xmlSavedStatesFileName;
    private static HiC hic;

    public static void viewShow() {
        try {
            ArrayList<String> savedStatePaths = new ArrayList<String>();
            Document dom;
            DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
            DocumentBuilder db = null;
            db = dbf.newDocumentBuilder();
            dom = db.parse(statesForSlideshow);
            NodeList nodeList = dom.getElementsByTagName("STATE");
            for (int i = 0; i < nodeList.getLength(); i++) {
                savedStatePaths.add(nodeList.item(i).getAttributes().getNamedItem("SelectedPath").getNodeValue());
            }
            System.out.println(savedStatePaths);

            carouselFrame.setLayout(new FlowLayout());
            carouselFrame.setResizable(true);
            carouselFrame.setVisible(true);
            carouselFrame.setSize(200, 200);
            carouselFrame.add(nextPanel);
            carouselFrame.add(prevPanel);

            nextPanel.add(nextButton, BorderLayout.EAST);
            nextPanel.setVisible(true);

            prevPanel.add(prevButton, BorderLayout.WEST);
            prevPanel.setVisible(false);

            nextButton.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    prevPanel.setVisible(true);
                }
            });

             /*for(String mapPath: savedStatePaths){

             }*/
        } catch (ParserConfigurationException e) {
            e.printStackTrace();
        } catch (SAXException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }


    }
}
