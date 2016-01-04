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

import javax.accessibility.Accessible;
import javax.accessibility.AccessibleContext;
import javax.accessibility.AccessibleRole;
import javax.swing.*;
import javax.swing.event.ChangeListener;
import javax.swing.plaf.ButtonUI;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;

/**
 * A split button. The user can either click the text, which executes an
 * action, or click the icon, which opens a popup menu.
 *
 * @author eoogbe
 */
class JSplitButton extends AbstractButton implements Accessible {

    private static final long serialVersionUID = 249426086343067971L;
    private static final String uiClassID = "ButtonUI";
    private static final int DEFAULT_POPUP_ICON_LENGTH = 10;
    private static final String ALWAYS_SHOWS_POPUP_CHANGED_PROPERTY =
            "alwaysShowPopup";
    private static final String SPLIT_GAP_CHANGED_PROPERTY = "splitGap";
    private static final String POPUP_ICON_CHANGED_PROPERTY = "popupIcon";
    private static final String DISABLED_POPUP_ICON_CHANGED_PROPERTY =
            "disabledPopupIcon";
    private static final String
            DISABLED_SELECTED_POPUP_ICON_CHANGED_PROPERTY = "disabledSelectedPopupIcon";
    private static final String PRESSED_POPUP_ICON_CHANGED_PROPERTY =
            "pressedPopupIcon";
    private static final String ROLLOVER_POPUP_ICON_CHANGED_PROPERTY =
            "rolloverPopupIcon";
    private static final String
            ROLLOVER_SELECTED_POPUP_ICON_CHANGED_PROPERTY = "rolloverSelectedPopupIcon";
    private static final String SELECTED_POPUP_ICON_CHANGED_PROPERTY =
            "selectedPopupIcon";
    private static final String MAIN_TEXT_CHANGED_PROPERTY = "mainText";
    private final JButton mainButton;
    private final JButton popupButton;
    private PopupAction popupAction;
    private boolean alwaysShowPopup;
    private int splitGap = 5;

    /**
     * Creates a new JSplitButton with no set text.
     */
    private JSplitButton() {
        this("");
    }

    /**
     * Creates a new JSplitButton with initial text.
     *
     * @param text the text displayed on this JSplitButton
     */
    public JSplitButton(String text) {
        mainButton = new JButton(text);
        popupButton = new JButton(createDefaultPopupIcon());
        //   popupButton.setPreferredSize(new Dimension((int)popupButton.getPreferredSize().getWidth()+10,(int)popupButton.getPreferredSize().getHeight()));

        setModel(new DefaultButtonModel());

        mainButton.setBorder(BorderFactory.createEmptyBorder());
        // popupButton.setBorder(BorderFactory.createEmptyBorder());

        mainButton.setContentAreaFilled(false);

        mainButton.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent e) {
                doClick();
            }

        });

        mainButton.addMouseListener(new MouseAdapter() {

            @Override
            public void mouseEntered(MouseEvent e) {
                if (isRolloverEnabled()) model.setRollover(true);
            }

            @Override
            public void mouseExited(MouseEvent e) {
                model.setRollover(false);
            }

        });

        setLayout(new BoxLayout(this, BoxLayout.LINE_AXIS));
        resetComponents();

        init(null, null);
    }

    /**
     * Creates a new JSplitButton with properties taken from the Action
     * supplied.
     *
     * @param action used to specify the properties of this JSplitButton
     */
    public JSplitButton(Action action) {
        this();
        setAction(action);
    }

    private Icon createDefaultPopupIcon() {
        Image image = new BufferedImage(DEFAULT_POPUP_ICON_LENGTH,
                DEFAULT_POPUP_ICON_LENGTH, BufferedImage.TYPE_INT_ARGB);
        Graphics g = image.getGraphics();

        g.setColor(new Color(255, 255, 255, 0));
        g.fillRect(0, 0, DEFAULT_POPUP_ICON_LENGTH,
                DEFAULT_POPUP_ICON_LENGTH);

        g.setColor(Color.BLACK);
        Polygon traingle = new Polygon(
                new int[]{0, DEFAULT_POPUP_ICON_LENGTH,
                        DEFAULT_POPUP_ICON_LENGTH / 2, 0},
                new int[]{0, 0, DEFAULT_POPUP_ICON_LENGTH, 0}, 4
        );
        g.fillPolygon(traingle);

        g.dispose();
        return new ImageIcon(image);
    }

    private void resetComponents() {
        removeAll();
        add(mainButton);
        add(Box.createRigidArea(new Dimension(splitGap, 0)));
        add(new JSeparator(VERTICAL));
        add(Box.createRigidArea(new Dimension(splitGap, 0)));
        add(popupButton);
        revalidate();
    }

    /**
     * Returns true if this JSplitButton shows the popup menu for every
     * click.
     *
     * @return true if this JSplitButton shows the popup menu for every
     * click
     * @see JSplitButton#setAlwaysShowPopup(boolean)
     */
    public boolean isAlwaysShowPopup() {
        return alwaysShowPopup;
    }

    /**
     * Sets whether this JSplitButton shows the popup menu for every click.
     *
     * @param alwaysShowPopup true if this JSplitButton shows the popup menu
     *                        for every click
     * @see JSplitButton#isAlwaysShowPopup()
     */
    public void setAlwaysShowPopup(boolean alwaysShowPopup) {
        boolean oldValue = this.alwaysShowPopup;
        this.alwaysShowPopup = alwaysShowPopup;
        firePropertyChange(ALWAYS_SHOWS_POPUP_CHANGED_PROPERTY, oldValue,
                alwaysShowPopup);

        if (popupAction != null && oldValue != alwaysShowPopup) {
            setComponentPopupMenu(popupAction.getPopupMenu());
        }
    }

    /**
     * Returns the gap between the separator and the labels on each side
     *
     * @return the gap between the separator and the labels on each side
     * @see JSplitButton#setSplitGap(int)
     */
    public int getSplitGap() {
        return splitGap;
    }

    /**
     * Sets the gap between the separator and the labels on each side.
     *
     * @param splitGap the gap set
     * @see JSplitButton#getSplitGap()
     */
    public void setSplitGap(int splitGap) {
        int oldValue = this.splitGap;
        this.splitGap = splitGap;
        firePropertyChange(SPLIT_GAP_CHANGED_PROPERTY, oldValue, splitGap);

        if (oldValue != splitGap) {
            resetComponents();
        }
    }

    /**
     * Returns the Icon used on the popup side. This Icon is also used as the
     * "pressed" and "disabled" Icon if they are not explicitly set.
     *
     * @return the popupIcon property
     * @see AbstractButton#getIcon()
     * @see JSplitButton#setPopupIcon(Icon)
     */
    private Icon getPopupIcon() {
        return popupButton.getIcon();
    }

    /**
     * Sets the Icon used on the popup side. If null, the Icon will be set to
     * a default value. This Icon is also used as the "pressed" and "disabled"
     * Icon if they are not explicitly set.
     *
     * @param icon the Icon set
     * @see AbstractButton#setIcon(Icon)
     * @see JSplitButton#getPopupIcon()
     */
    public void setPopupIcon(Icon icon) {
        Icon oldValue = getPopupIcon();
        if (icon == null) icon = createDefaultPopupIcon();
        firePropertyChange(POPUP_ICON_CHANGED_PROPERTY, oldValue, icon);

        if (!oldValue.equals(icon)) {
            popupButton.setIcon(icon);
        }
    }

    /**
     * Returns the icon used on the popup side when this JSplitButton is
     * disabled. If no disabled icon has been set this will forward the call
     * to the look and feel to construct an appropriate disabled Icon. Some
     * look and feels might not render the disabled Icon, in which case they
     * will ignore this.
     *
     * @return the disabledPopupIcon property
     * @see AbstractButton#getDisabledIcon()
     * @see JSplitButton#setDisabledPopupIcon(Icon)
     */
    private Icon getDisabledPopupIcon() {
        return popupButton.getDisabledIcon();
    }

    /**
     * Sets the icon used on the popup side when this JSplitButton is
     * disabled. Some look and feels might not render the disabled Icon, in
     * which case they will ignore this.
     *
     * @param icon the icon set
     * @see AbstractButton#setDisabledIcon(Icon)
     * @see JSplitButton#getDisabledPopupIcon()
     */
    public void setDisabledPopupIcon(Icon icon) {
        Icon oldValue = getDisabledPopupIcon();
        firePropertyChange(DISABLED_POPUP_ICON_CHANGED_PROPERTY, oldValue,
                icon);

        if (!oldValue.equals(icon)) {
            popupButton.setDisabledIcon(icon);
        }
    }

    /**
     * Returns the icon used on the popup side when this JSplitButton is
     * disabled and selected. If no disabled selected Icon has been set, this
     * will forward the call to the LookAndFeel to construct an appropriate
     * disabled Icon from the selection icon if it has been set and to
     * getDisabledPopupIcon() otherwise.
     *
     * @return the disabledSelctedPopupIcon property
     * @see AbstractButton#getDisabledSelectedIcon()
     * @see JSplitButton#setDisabledSelectedPopupIcon(Icon)
     */
    private Icon getDisabledSelectedPopupIcon() {
        return popupButton.getDisabledSelectedIcon();
    }

    /**
     * Sets the icon used on the popup side when this JSplitButton is
     * disabled and selected.
     *
     * @param icon the icon set
     * @see AbstractButton#setDisabledSelectedIcon(Icon)
     * @see JSplitButton#getDisabledSelectedPopupIcon()
     */
    public void setDisabledSelectedPopupIcon(Icon icon) {
        Icon oldValue = getDisabledSelectedPopupIcon();
        firePropertyChange(DISABLED_SELECTED_POPUP_ICON_CHANGED_PROPERTY,
                oldValue, icon);

        if (!oldValue.equals(icon)) {
            popupButton.setDisabledSelectedIcon(icon);
        }
    }

    /**
     * Returns the icon used on the popup side when this JSpitButton is
     * pressed.
     *
     * @return the pressedPopupIcon property
     * @see AbstractButton#getPressedIcon()
     * @see JSplitButton#setPressedPopupIcon(Icon)
     */
    private Icon getPressedPopupIcon() {
        return popupButton.getPressedIcon();
    }

    /**
     * Sets the icon used on the popup side when this JSplitButton is pressed.
     *
     * @param icon the icon set
     * @see AbstractButton#setPressedIcon(Icon)
     * @see JSplitButton#getPressedPopupIcon()
     */
    public void setPressedPopupIcon(Icon icon) {
        Icon oldValue = getPressedPopupIcon();
        firePropertyChange(PRESSED_POPUP_ICON_CHANGED_PROPERTY, oldValue,
                icon);

        if (!oldValue.equals(icon)) {
            popupButton.setPressedIcon(icon);
        }
    }

    /**
     * Returns the rollover icon used on the popup side.
     *
     * @return the rolloverPopupIcon property
     * @see AbstractButton#getRolloverIcon()
     * @see JSplitButton#setRolloverPopupIcon(Icon)
     */
    private Icon getRolloverPopupIcon() {
        return popupButton.getRolloverIcon();
    }

    /**
     * Sets the rollover icon used on the popup side.
     *
     * @param icon the icon set
     * @see AbstractButton#setRolloverIcon(Icon)
     * @see JSplitButton#getRolloverPopupIcon()
     */
    public void setRolloverPopupIcon(Icon icon) {
        Icon oldValue = getRolloverPopupIcon();
        firePropertyChange(ROLLOVER_POPUP_ICON_CHANGED_PROPERTY, oldValue,
                icon);

        if (!oldValue.equals(icon)) {
            popupButton.setRolloverIcon(icon);
        }
    }

    /**
     * Returns the rollover selection icon used on the popup side.
     *
     * @return the rolloverSelectedPopupIcon property
     * @see AbstractButton#getRolloverSelectedIcon()
     * @see JSplitButton#setRolloverSelectedPopupIcon(Icon)
     */
    private Icon getRolloverSelectedPopupIcon() {
        return popupButton.getRolloverSelectedIcon();
    }

    /**
     * Sets the rollover selection icon used on the popup side.
     *
     * @param icon the icon set
     * @see AbstractButton#setRolloverSelectedIcon(Icon)
     * @see JSplitButton#getRolloverSelectedPopupIcon()
     */
    public void setRolloverSelectedPopupIcon(Icon icon) {
        Icon oldValue = getRolloverSelectedPopupIcon();
        firePropertyChange(ROLLOVER_SELECTED_POPUP_ICON_CHANGED_PROPERTY,
                oldValue, icon);

        if (!oldValue.equals(icon)) {
            popupButton.setRolloverSelectedIcon(icon);
        }
    }

    /**
     * Returns the selected icon used on the popup side.
     *
     * @return the selectedPopupIcon property
     * @see AbstractButton#getSelectedIcon()
     * @see JSplitButton#setSelectedPopupIcon(Icon)
     */
    private Icon getSelectedPopupIcon() {
        return popupButton.getSelectedIcon();
    }

    /**
     * Sets the selected icon used on the popup side.
     *
     * @param icon the icon set
     * @see AbstractButton#setSelectedIcon(Icon)
     * @see JSplitButton#getSelectedPopupIcon()
     */
    public void setSelectedPopupIcon(Icon icon) {
        Icon oldValue = getSelectedPopupIcon();
        firePropertyChange(SELECTED_POPUP_ICON_CHANGED_PROPERTY, oldValue,
                icon);

        if (!oldValue.equals(icon)) {
            popupButton.setSelectedIcon(icon);
        }
    }

    /**
     * Returns the text of the main part of this JSplitButton. Use this
     * method instead of {@link AbstractButton#getText()}.
     *
     * @return the text of the main part of this JSplitButton
     * @see AbstractButton#getText()
     * @see JSplitButton#setMainText(String)
     */
    private String getMainText() {
        return mainButton.getText();
    }

    /**
     * Sets the text of the main part of this JSplitButton. Use this method
     * instead of {@link AbstractButton#setText(String)}.
     *
     * @param text the text set
     */
    public void setMainText(String text) {
        String oldValue = getMainText();
        firePropertyChange(MAIN_TEXT_CHANGED_PROPERTY, oldValue, text);

        if (!oldValue.equals(text)) {
            mainButton.setText(text);
        }
    }

    /* (non-Javadoc)
     * @see javax.swing.AbstractButton#addActionListener(java.awt.event.ActionListener)
     */
    @Override
    public void addActionListener(ActionListener listener) {
        mainButton.addActionListener(listener);
        super.addActionListener(listener);
    }

    /* (non-Javadoc)
     * @see javax.swing.AbstractButton#removeActionListener(java.awt.event.ActionListener)
     */
    @Override
    public void removeActionListener(ActionListener listener) {
        mainButton.removeActionListener(listener);
        super.removeActionListener(listener);
    }

    /* (non-Javadoc)
     * @see javax.swing.AbstractButton#addChangeListener(javax.swing.event.ChangeListener)
     */
    @Override
    public void addChangeListener(ChangeListener listener) {
        mainButton.addChangeListener(listener);
        super.addChangeListener(listener);
    }

    /* (non-Javadoc)
     * @see javax.swing.AbstractButton#removeChangeListener(javax.swing.event.ChangeListener)
     */
    @Override
    public void removeChangeListener(ChangeListener listener) {
        mainButton.removeChangeListener(listener);
        super.removeChangeListener(listener);
    }

    /* (non-Javadoc)
     * @see javax.swing.AbstractButton#addItemListener(java.awt.event.ItemListener)
     */
    @Override
    public void addItemListener(ItemListener listener) {
        mainButton.addItemListener(listener);
        super.addItemListener(listener);
    }

    /* (non-Javadoc)
     * @see javax.swing.AbstractButton#removeItemListener(java.awt.event.ItemListener)
     */
    @Override
    public void removeItemListener(ItemListener listener) {
        mainButton.removeItemListener(listener);
        super.removeItemListener(listener);
    }

    /* (non-Javadoc)
     * @see javax.swing.AbstractButton#getAction()
     */
    @Override
    public Action getAction() {
        return mainButton.getAction();
    }

    /* (non-Javadoc)
     * @see javax.swing.AbstractButton#setAction(javax.swing.Action)
     */
    @Override
    public void setAction(Action action) {
        Action oldValue = getAction();
        firePropertyChange("action", oldValue, action);
        mainButton.setAction(action);
    }

    /* (non-Javadoc)
     * @see javax.swing.JComponent#getAccessibleContext()
     */
    @Override
    public AccessibleContext getAccessibleContext() {
        if (accessibleContext == null) {
            accessibleContext = new AccessibleJSplitButton();
        }

        return accessibleContext;
    }

    /* (non-Javadoc)
     * @see javax.swing.JComponent#getComponentPopupMenu()
     */
    @Override
    public JPopupMenu getComponentPopupMenu() {
        return (popupAction == null) ? null : popupAction.getPopupMenu();
    }

    /* (non-Javadoc)
     * @see javax.swing.JComponent#setComponentPopupMenu(javax.swing.JPopupMenu)
     */
    @Override
    public void setComponentPopupMenu(JPopupMenu popupMenu) {
        if (popupAction != null) {
            super.removeActionListener(popupAction);
            mainButton.removeActionListener(popupAction);
            popupButton.removeActionListener(popupAction);
            popupAction = null;
        }

        if (popupMenu != null) {
            popupAction = new PopupAction(popupMenu);
            popupButton.addActionListener(popupAction);

            if (alwaysShowPopup) {
                super.addActionListener(popupAction);
                mainButton.addActionListener(popupAction);
            }
        }
    }

    /* (non-Javadoc)
     * @see javax.swing.JComponent#getPopupLocation(java.awt.event.MouseEvent)
     */
    @Override
    public Point getPopupLocation(MouseEvent event) {
        return getPopupLocationRelativeTo(event.getComponent());
    }

    /* (non-Javadoc)
     * @see javax.swing.AbstractButton#getSelectedObjects()
     */
    @Override
    public Object[] getSelectedObjects() {
        Object[] result = new Object[1];

        if (mainButton.isSelected()) {
            result[0] = mainButton;
        } else if (popupButton.isSelected()) {
            result[0] = popupButton;
        } else {
            return null;
        }

        return result;
    }

    /* (non-Javadoc)
     * @see javax.swing.AbstractButton#getMnemonic()
     */
    @Override
    public int getMnemonic() {
        return mainButton.getMnemonic();
    }

    /* (non-Javadoc)
     * @see javax.swing.AbstractButton#setMnemonic(int)
     */
    @Override
    public void setMnemonic(int mnemonic) {
        int oldValue = getMnemonic();
        firePropertyChange(MNEMONIC_CHANGED_PROPERTY, oldValue, mnemonic);
        mainButton.setMnemonic(mnemonic);
    }

    /* (non-Javadoc)
     * @see javax.swing.AbstractButton#getDisplayedMnemonicIndex()
     */
    @Override
    public int getDisplayedMnemonicIndex() {
        return mainButton.getDisplayedMnemonicIndex();
    }

    /* (non-Javadoc)
     * @see javax.swing.AbstractButton#setDisplayedMnemonicIndex(int)
     */
    @Override
    public void setDisplayedMnemonicIndex(int index)
            throws IllegalArgumentException {
        int oldValue = getDisplayedMnemonicIndex();
        mainButton.setDisplayedMnemonicIndex(index);
        firePropertyChange("displayedMnemonicIndex", oldValue, index);
    }

    /* (non-Javadoc)
     * @see javax.swing.AbstractButton#setHideActionText(boolean)
     */
    @Override
    public void setHideActionText(boolean hideActionText) {
        mainButton.setHideActionText(hideActionText);
        super.setHideActionText(hideActionText);
    }

    /* (non-Javadoc)
     * @see javax.swing.JComponent#getUIClassID()
     */
    @Override
    public String getUIClassID() {
        return uiClassID;
    }

    /* (non-Javadoc)
     * @see javax.swing.AbstractButton#updateUI()
     */
    @Override
    public void updateUI() {
        setUI((ButtonUI) UIManager.getUI(this));
    }

    private Point getPopupLocationRelativeTo(Component comp) {
        Insets insets = getInsets();
        int height = getHeight();

        if (comp == popupButton) {
            Insets mainButtonInsets = mainButton.getInsets();
            int width = mainButton.getWidth();
            return new Point(-splitGap - width - mainButtonInsets.left - mainButtonInsets.right - 30, height);
        } else {
            Insets mainButtonInsets = mainButton.getInsets();
            int width = mainButton.getWidth() + mainButtonInsets.left +
                    mainButtonInsets.right + splitGap;
            if (comp == this) width += insets.left;
            return new Point(width, height);
        }
    }

    private class AccessibleJSplitButton extends AccessibleAbstractButton {

        private static final long serialVersionUID = 4456023430055534939L;

        /* (non-Javadoc)
         * @see javax.swing.JComponent.
         * AccessibleJComponent#getAccessibleRole()
         */
        @Override
        public AccessibleRole getAccessibleRole() {
            return AccessibleRole.PUSH_BUTTON;
        }

    }

    private class PopupAction implements ActionListener {

        private final JPopupMenu popupMenu;

        public PopupAction(JPopupMenu popupMenu) {
            this.popupMenu = popupMenu;
        }

        /* (non-Javadoc)
         * @see java.awt.event.ActionListener#actionPerformed(
         * java.awt.event.ActionEvent)
         */
        @Override
        public void actionPerformed(ActionEvent e) {
            Component comp = (Component) e.getSource();
            Point popupLocation = getPopupLocationRelativeTo(comp);
            popupMenu.show(comp, popupLocation.x, popupLocation.y);
        }

        public JPopupMenu getPopupMenu() {
            return popupMenu;
        }

    }
}

