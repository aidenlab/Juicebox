package juicebox.windowui;

import org.broad.igv.Globals;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.prefs.Preferences;

/**
 * @author Ido Machol
 * @modified Muhammad S Shamim
 */
public abstract class RecentMenu extends JMenu {
    private static final long serialVersionUID = 4685393080959162312L;
    private final int m_maxItems;
    private final String m_entry;
    private final Preferences prefs = Preferences.userNodeForPackage(Globals.class);
    private final List<String> m_items = new ArrayList<String>();

    public RecentMenu(String name, int count,String prefEntry) {
        super(name);

        this.m_maxItems = count;
        this.m_entry = prefEntry;
        String[] recentEntries = new String[count];
        Arrays.fill(recentEntries, "");

        // load recent positions from properties
        for (int i = 0; i < this.m_maxItems; i++) {
            String val = prefs.get(this.m_entry + i, "");
            if (!val.equals("")) {
                addEntry(val, false);
            } else {
                if (i == 0) this.setEnabled(false);
                break;
            }
        }
    }

    /**
     * Add new recent entry, update file and menu
     *
     * @param savedEntry   Name and Value of entry.
     * @param updateFile also save to file, Constructor call with false - no need to re-write.
     */
    public void addEntry(String savedEntry, boolean updateFile) {
        //check if this is disabled
        if (!this.isEnabled()) {
            this.setEnabled(true);
        }

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
    }

    /**
     * Abstract event, fires when recent map is selected.
     *
     * @param mapPath The file that was selected.
     */
    public abstract void onSelectPosition(String mapPath);

}
