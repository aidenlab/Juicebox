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

package juicebox.windowui;

import juicebox.DirectoryManager;
import juicebox.HiCGlobals;
import juicebox.state.XMLFileWriter;
import org.broad.igv.Globals;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import javax.swing.*;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.prefs.Preferences;

/**
 * @author Ido Machol, Muhammad S Shamim, Neva Durand
 */
public abstract class RecentMenu extends JMenu {
    private static final long serialVersionUID = 4685393080959162312L;
    private static final int maxLengthEntryName = 100;
    private final int m_maxItems;
    private final String m_entry;
    private final Preferences prefs = Preferences.userNodeForPackage(Globals.class);
    private final File JuiceboxStatesXML = new File(DirectoryManager.getHiCDirectory(), "JuiceboxStatesXML.txt");
    private final HiCGlobals.menuType myType;
    private List<String> m_items = new ArrayList<String>();

    public RecentMenu(String name, int count, String prefEntry, HiCGlobals.menuType type) {
        super(name);

        this.m_maxItems = count;
        this.m_entry = prefEntry;
        this.myType = type;
        String[] recentEntries = new String[count];
        Arrays.fill(recentEntries, "");

        boolean addedItem = false;
        // load recent positions from properties
        for (int i = this.m_maxItems - 1; i >= 0; i--) {
            String val = prefs.get(this.m_entry + i, "");
            if (!val.equals("")) {
                addEntry(val, false);
                addedItem = true;
            }
        }
        if (!addedItem) {
            this.setEnabled(false);
        }
    }

    /**
     * Add "Clear" menu item to bottom of this list
     */
    private void addClearItem() {
        //---- Clear Recent ----
        JMenuItem clearMapList = new JMenuItem();
        clearMapList.setText("Clear ");
        clearMapList.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                //Clear all items from preferences:
                for (int i = 0; i < m_maxItems; i++) {
                    prefs.remove(m_entry + i);
                }
                //clear the existing items
                removeAll();
                m_items = new ArrayList<String>();
                setEnabled(false);

                //Clean state data:
                if (myType == HiCGlobals.menuType.STATE) {
                    try {
                        BufferedWriter bWriter = new BufferedWriter(new FileWriter(HiCGlobals.stateFile, false));
                        BufferedWriter buffWriter = new BufferedWriter(new FileWriter(JuiceboxStatesXML, false));

                        HiCGlobals.savedStatesList.clear();
                        XMLFileWriter.overwriteXMLFile();

                        buffWriter.close();
                        bWriter.close();
                    } catch (IOException ex) {
                        ex.printStackTrace();
                    }
                }
            }
        });
        addSeparator();
        add(clearMapList);
    }

    /**
     * Add new recent entry, update file and menu
     *
     * @param savedEntryOriginal Name and Value of entry.
     * @param updateFile         also save to file, Constructor call with false - no need to re-write.
     */
    public void addEntry(String savedEntryOriginal, boolean updateFile) {
        //clear the existing items
        this.removeAll();

        m_items.remove(savedEntryOriginal);
        m_items.add(0, savedEntryOriginal);

        //Chop last item if list is over size:
        if (this.m_items.size() > this.m_maxItems) {
            this.m_items.remove(this.m_items.size() - 1);
        }

        //add items back to the menu
        for (String m_item : this.m_items) {
            String delimiter = "@@";
            String[] temp;
            temp = m_item.split(delimiter);

            if (!temp[0].equals("")) {
                String truncatedName = temp[0];
                if (truncatedName.length() > maxLengthEntryName) {
                    truncatedName = truncatedName.substring(0, maxLengthEntryName - 1);
                }
                JMenuItem menuItem = new JMenuItem(truncatedName);
                menuItem.setVisible(true);
                menuItem.setToolTipText(temp[0]);
                menuItem.setActionCommand(m_item);
                //menuItem.setActionMap();
                menuItem.addActionListener(new ActionListener() {
                    public void actionPerformed(ActionEvent actionEvent) {
                        onSelectPosition(actionEvent.getActionCommand());
                    }
                });
                this.add(menuItem);
            }

        }
        //update the file
        if (updateFile) {
            try {
                //todo: null in savedEntryOriginal will cause an exception...
                for (int i = 0; i < this.m_maxItems; i++) {
                    if (i < this.m_items.size()) {
                        prefs.put(this.m_entry + i, this.m_items.get(i));
                    } else {
                        prefs.remove(this.m_entry + i);
                    }
                }
            } catch (Exception x) {
                x.printStackTrace();
            }
        }
        addClearItem();

        //Enable saved states restore, if not already enabled:
        if (!this.isEnabled()) {
            this.setEnabled(true);
        }
    }

    /**
     * Abstract event, fires when recent map is selected.
     *
     * @param mapPath The file that was selected.
     */
    public abstract void onSelectPosition(String mapPath);

    //TODO--- Update recent menu when HiC states are imported
    public void updateNamesFromImport(String importedFile) {
        Document doc;
        DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
        for (int c = 0; c < m_maxItems; c++) {
            prefs.remove(m_entry + c);
        }
        //clear the existing items
        removeAll();
        m_items = new ArrayList<String>();
        //import names to previous states menu
        try {
            DocumentBuilder documentBuilder = dbf.newDocumentBuilder();
            doc = documentBuilder.parse(importedFile);
            Element element = doc.getDocumentElement();
            NodeList nodeList = element.getElementsByTagName("STATE");
            for (int i = 0; i < nodeList.getLength(); i++) {
                String importedMapPath = nodeList.item(i).getAttributes().getNamedItem("SelectedPath").getNodeValue();
                m_items.add(importedMapPath);
            }

            if (this.m_items.size() > this.m_maxItems) {
                this.m_items.remove(this.m_items.size() - 1);
            }
            for (String m_item : this.m_items) {

                if (!m_item.equals("")) {
                    JMenuItem menuItem = new JMenuItem(m_item);
                    menuItem.setVisible(true);
                    menuItem.setToolTipText(m_item);
                    menuItem.setActionCommand(m_item);
                    //menuItem.setActionMap();
                    menuItem.addActionListener(new ActionListener() {
                        public void actionPerformed(ActionEvent actionEvent) {
                            onSelectPosition(actionEvent.getActionCommand());
                        }
                    });
                    //menuItem.addMouseListener(new MouseListener() );
                    this.add(menuItem);
                }

            }

            for (int i = 0; i < this.m_maxItems; i++) {
                if (i < this.m_items.size()) {
                    prefs.put(this.m_entry + i, this.m_items.get(i));
                } else {
                    prefs.remove(this.m_entry + i);
                }
            }

            addClearItem();

        } catch (ParserConfigurationException pce) {
            pce.printStackTrace();
        } catch (SAXException se) {
            se.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }


    }

    public String checkForDuplicateNames(String savedNameOriginal) {
        //check for saved states
        String savedName = savedNameOriginal;

        boolean suitableNameNotFound = true;
        while (suitableNameNotFound) {
            suitableNameNotFound = false;
            boolean repFound = false;
            for (String item : m_items) {
                if (item.equals(savedName)) {
                    repFound = true;
                    break;
                }
            }
            if (repFound) {
                suitableNameNotFound = true;
                int option = JOptionPane.showConfirmDialog(null, "State name: \n" + savedName + "\n" +
                        "already exists. Do you want to overwrite it?", "Confirm", JOptionPane.YES_NO_OPTION);
                if (option == JOptionPane.YES_OPTION) {
                    return savedName;
                } else if (option == JOptionPane.NO_OPTION) {
                    savedName = JOptionPane.showInputDialog(null, "Please enter new name for state.");
                    return savedName;
                } else if (option == JOptionPane.CLOSED_OPTION) {
                    savedName = "";
                    return savedName;
                }
            }
        }
        return savedName;
    }
}
