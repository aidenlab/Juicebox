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

package juicebox.windowui;

import juicebox.HiCGlobals;
import org.broad.igv.Globals;

import javax.swing.*;
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
    private final int m_maxItems;
    private final String m_entry;
    private final Preferences prefs = Preferences.userNodeForPackage(Globals.class);
    private final File currentStates = new File(HiCGlobals.stateFileName);
    private final File JuiceboxStatesXML = new File("JuiceboxStatesXML.txt");
    private List<String> m_items = new ArrayList<String>();

    public RecentMenu(String name, int count, String prefEntry) {
        super(name);

        this.m_maxItems = count;
        this.m_entry = prefEntry;
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
                try {
                    BufferedWriter bWriter = new BufferedWriter(new FileWriter(currentStates, false));
                    BufferedWriter buffWriter = new BufferedWriter(new FileWriter(JuiceboxStatesXML,false));
                    buffWriter.close();
                    bWriter.close();
                } catch(IOException ex){
                    ex.printStackTrace();
                }
            }
        });
        addSeparator();
        add(clearMapList);
    }

    public String getRecentMapName(){
        String recentMapName = "";
        /*String delimeter = "@@";
        String[] temp;*/
        if(m_items.get(0)!=null && !m_items.get(0).equals("")){
            //temp = m_items.get(0).split(delimeter);
            recentMapName += m_items.get(0);
        }
        return recentMapName;
    }
    /**
     * Add new recent entry, update file and menu
     *
     * @param savedEntry Name and Value of entry.
     * @param updateFile also save to file, Constructor call with false - no need to re-write.
     */
    public void addEntry(String savedEntry, boolean updateFile) {

        //clear the existing items
        this.removeAll();

        //Add item, remove previous existing duplicate:
        m_items.remove(savedEntry);
        m_items.add(0, savedEntry);

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
                JMenuItem menuItem = new JMenuItem(temp[0]);
                menuItem.setVisible(true);
                menuItem.setToolTipText(temp[0]);
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
        //update the file
        if (updateFile) {
            try {
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

        //check if this is disabled
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


}
