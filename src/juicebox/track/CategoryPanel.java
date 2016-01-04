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

package juicebox.track;

import com.jidesoft.swing.ButtonStyle;
import com.jidesoft.swing.JideButton;
import org.broad.igv.util.ResourceLocator;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;

//import javax.swing.border.Border;
//import javax.swing.border.LineBorder;
//import java.awt.geom.Path2D;
//import java.awt.geom.Rectangle2D;
//import java.awt.geom.RoundRectangle2D;

/**
 * @author Jim Robinson
 * @date 5/8/12
 */
class CategoryPanel extends JPanel {

    private static final long serialVersionUID = -729150966236965013L;
    private final JPanel listPanel;
    private final JPanel labelBar;
    int nColumns = 5;
    private boolean expanded;

    public CategoryPanel(String name, List<ResourceLocator> locatorList, Set<String> loadedTrackNames) {

        expanded = true;

        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        setAlignmentX(LEFT_ALIGNMENT);
        //setLayout(null);

        labelBar = new JPanel();
        //labelBar.setBackground(Color.blue);
        labelBar.setLayout(new BoxLayout(labelBar, BoxLayout.X_AXIS));
        labelBar.setBorder(BorderFactory.createRaisedBevelBorder()); //  new LabelBorder(Color.black));
        labelBar.setAlignmentX(LEFT_ALIGNMENT);
        JideButton toggleButton = new JideButton(expanded ? " - " : " + ");
        toggleButton.setButtonStyle(ButtonStyle.HYPERLINK_STYLE);
        labelBar.add(toggleButton);

        labelBar.add(new JLabel(name));
        this.add(labelBar);


        listPanel = new JPanel();
        listPanel.setLayout(new GridLayout(0, 4));
        for (ResourceLocator loc : locatorList) {
            final String trackName = loc.getTrackName();
            JCheckBox cb = new JCheckBox(trackName);
            cb.setSelected(loadedTrackNames.contains(trackName));
            listPanel.add(cb);
        }
        this.add(listPanel);

        toggleButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                expanded = !expanded;
                listPanel.setVisible(expanded);
            }
        });
        labelBar.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent mouseEvent) {
                expanded = !expanded;
                listPanel.setVisible(expanded);
            }
        });

    }

    public Collection<String> getSelectedTracks() {
        List<String> selectedTracks = new ArrayList<String>();
        for (Component c : listPanel.getComponents()) {
            if (c instanceof JCheckBox && ((JCheckBox) c).isSelected()) {
                selectedTracks.add(((JCheckBox) c).getText());

            }
        }
        return selectedTracks;

    }


    /**
     * If the <code>preferredSize</code> has been set to a
     * non-<code>null</code> value just returns it.
     * If the UI delegate's <code>getPreferredSize</code>
     * method returns a non <code>null</code> value then return that;
     * otherwise defer to the component's layout manager.
     *
     * @return the value of the <code>preferredSize</code> property
     * @see #setPreferredSize
     * @see javax.swing.plaf.ComponentUI
     */
    @Override
    public Dimension getPreferredSize() {
        if (listPanel == null) {
            return super.getPreferredSize();
        } else {

            Dimension d = listPanel.getPreferredSize();
            Component p = getRootPane();
            int h = listPanel.isVisible() ? d.height : 0;
            int w = p == null ? d.width : getParent().getWidth();
            return new Dimension(w, 3 + 3 + 30 + h);

        }
    }

    /**
     * If the minimum size has been set to a non-<code>null</code> value
     * just returns it.  If the UI delegate's <code>getMinimumSize</code>
     * method returns a non-<code>null</code> value then return that; otherwise
     * defer to the component's layout manager.
     *
     * @return the value of the <code>minimumSize</code> property
     * @see #setMinimumSize
     * @see javax.swing.plaf.ComponentUI
     */
    @Override
    public Dimension getMinimumSize() {
        return getPreferredSize();
    }

    @Override
    public void doLayout() {
        if (labelBar == null || listPanel == null) return;

        Dimension d = listPanel.getPreferredSize();
        Component p = getParent();
        int w = p == null ? d.width : getParent().getWidth();


        int y = 0;
        labelBar.setBounds(0, y, w, 30);
        y += 30;

        //noinspection SuspiciousNameCombination
        listPanel.setBounds(y, 33, w, d.height);

    }


}
