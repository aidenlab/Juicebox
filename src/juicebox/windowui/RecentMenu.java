package juicebox.windowui;

import org.broad.igv.Globals;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;
import java.util.prefs.Preferences;

/**
 * @author Ido Machol
 * @modified Muhammad S Shamim
 */
public abstract class RecentMenu extends JMenu {
    final private static String HIC_RECENT = "hicRecent";
    private static final long serialVersionUID = 4685393080959162312L;
    private final String defaultText = "";
    private final int m_maxItems;
    private boolean b_isEnabled = false;
    private final Preferences prefs = Preferences.userNodeForPackage(Globals.class);
    private final List<String> m_items = new ArrayList<String>();

    public RecentMenu(int count) {
        super();
        this.setText("Recent");
        this.setMnemonic('R');
        this.m_maxItems = count;
        //initialize default entries
        String[] recentEntries = new String[count];
        for (int index = 0; index < this.m_maxItems; index++) {
            recentEntries[index] = defaultText;
        }

        // load recent positions from properties
        for (int i = 0; i < this.m_maxItems; i++) {
            String val = prefs.get(HIC_RECENT + i, "");
            if (!val.equals("")) {
                addEntry(val, false);
            } else {
                if (i == 0) {
                    // No items.
                    this.setEnabled(false);
                }
                break;
            }
        }
    }

    /**
     * Add new recent entry, update file and menu
     *
     * @param savedMap   url and title of map.
     * @param updateFile also save to file, Constructor call with false - no need to re-write.
     */
    public void addEntry(String savedMap, boolean updateFile) {
        //check if this is disabled
        if (!this.isEnabled()) {
            this.setEnabled(true);
        }

        //clear the existing items
        this.removeAll();

        //Add item, remove previous existing duplicate:
        m_items.remove(savedMap);
        m_items.add(0, savedMap);

        //Chop last item if list is over size:
        if (this.m_items.size() > this.m_maxItems) {
            this.m_items.remove(this.m_items.size() - 1);
        }

        //add items back to the menu
        for (String m_item : this.m_items) {
            JMenuItem menuItem = new JMenuItem();

            String delimiter = "@@";
            String[] temp;
            temp = m_item.split(delimiter);

            menuItem.setText(temp[0]);
            if (temp[0].equals(defaultText)) {
                menuItem.setVisible(false);
            } else {
                menuItem.setVisible(true);
                menuItem.setToolTipText(temp[0]);
                menuItem.setActionCommand(m_item);
                menuItem.addActionListener(new ActionListener() {
                    public void actionPerformed(ActionEvent actionEvent) {
                        onSelectPosition(actionEvent.getActionCommand());
                    }
                });
            }
            this.add(menuItem);
        }
        //update the file
        if (updateFile) {
            try {
                for (int i = 0; i < this.m_maxItems; i++) {
                    if (i < this.m_items.size()) {
                        prefs.put(HIC_RECENT + i, this.m_items.get(i));
                    } else {
                        prefs.remove(HIC_RECENT + i);
                    }
                }
            } catch (Exception x) {
                x.printStackTrace();
            }
        }
    }

    public boolean isEnabled() {
        return this.b_isEnabled;
    }

    public void setEnabled(boolean b_newState) {
        this.b_isEnabled = b_newState;
    }

    /**
     * Abstract event, fires when recent map is selected.
     *
     * @param mapPath The file that was selected.
     */
    public abstract void onSelectPosition(String mapPath);
}
