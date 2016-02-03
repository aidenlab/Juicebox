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

/**
 * Created by Marie on 6/25/15.
 */

import juicebox.MainWindow;
import juicebox.track.feature.CustomAnnotation;
import juicebox.track.feature.Feature2D;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.util.ArrayList;
import java.util.HashMap;

/* 1.4 example used by DialogDemo.java. */
public class EditFeatureAttributesDialog extends JDialog implements ActionListener, PropertyChangeListener {
    private static final long serialVersionUID = 3432096869822002860L;
    private final String defaultNewAttributeName = "<Attribute Name>";
    private final String defaultNewAttributeValue = "<Attribute Value>";
    private final HashMap<String, JTextField> textFields;
    private final Feature2D feature;
    private final CustomAnnotation customAnnotations;
    private final ArrayList<String> attributeKeys;
    private final JCheckBox echoOption;
    private final JOptionPane optionPane;
    private final String btnString1 = "Enter";
    private String typedText = null;

    /**
     * Creates the reusable dialog.
     */
    public EditFeatureAttributesDialog(MainWindow mainWindow, Feature2D feature, CustomAnnotation customAnnotations) {
        super(mainWindow);

        this.customAnnotations = customAnnotations;
        this.feature = feature;
        setTitle("Attribute Editor");

        textFields = new HashMap<String, JTextField>();

        attributeKeys = feature.getAttributeKeys();
        Object[] array = new Object[2 * (attributeKeys.size() + 2)];
        int count = 0;

        //Create an array of the text and components to be displayed.
        for (String key : attributeKeys) {
            //Register an event handler that puts the text into the option pane.
            JTextField textField = new JTextField(10);
            textField.setText(feature.getAttribute(key));
            textField.addActionListener(this);
            textFields.put(key, textField);
            array[count++] = key + ":";
            array[count++] = textField;
        }

        // Add panes to input new attribute
        attributeKeys.add("New Field Name");
        JTextField textField = new JTextField(10);
        textField.setText(defaultNewAttributeName);
        textField.addActionListener(this);
        textFields.put("New Field Name", textField);
        JTextField textField2 = new JTextField(10);
        textField2.setText(defaultNewAttributeValue);
        textField2.addActionListener(this);
        textFields.put("New Field Value", textField2);
        echoOption = new JCheckBox("Set value as default for new attribute");

        array[count++] = "Add Attribute:";
        array[count++] = textField;
        array[count++] = textField2;
        array[count] = echoOption;


        String btnString2 = "Cancel";
        Object[] options = {btnString1, btnString2};

        //Create the JOptionPane.
        optionPane = new JOptionPane(array,
                JOptionPane.QUESTION_MESSAGE,
                JOptionPane.YES_NO_OPTION,
                null,
                options,
                options[0]);

        //Make this dialog display it.
        setContentPane(new JScrollPane(optionPane));

        //Handle window closing correctly.
        setDefaultCloseOperation(DO_NOTHING_ON_CLOSE);
        addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent we) {
                /*
                 * Instead of directly closing the window,
                 * we're going to change the JOptionPane's
                 * value property.
                 */
                optionPane.setValue(JOptionPane.CLOSED_OPTION);
            }
        });

        //Register an event handler that reacts to option pane state changes.
        optionPane.addPropertyChangeListener(this);
        pack();
        setLocationRelativeTo(null);
        setVisible(true);
    }

    /**
     * Returns null if the typed string was invalid;
     * otherwise, returns the string as the user entered it.
     */
    public String getValidatedText() {
        return typedText;
    }

    /**
     * This method handles events for the text field.
     */
    public void actionPerformed(ActionEvent e) {
        optionPane.setValue(btnString1);
    }

    /**
     * This method reacts to state changes in the option pane.
     */
    public void propertyChange(PropertyChangeEvent e) {
        String prop = e.getPropertyName();

        if (isVisible()
                && (e.getSource() == optionPane)
                && (JOptionPane.VALUE_PROPERTY.equals(prop) ||
                JOptionPane.INPUT_VALUE_PROPERTY.equals(prop))) {
            Object value = optionPane.getValue();

            if (value == JOptionPane.UNINITIALIZED_VALUE) {
                //ignore reset
                return;
            }

            optionPane.setValue(
                    JOptionPane.UNINITIALIZED_VALUE);

            if (btnString1.equals(value)) {
                for (String key : attributeKeys) {
                    typedText = textFields.get(key).getText();
                    if (typedText != null) {
                        // New Attribute
                        if (key.equals("New Field Name")) {
                            String newAttributeText = textFields.get("New Field Value").getText();
                            // TODO meh - if add to all (echo), things after won't work.

                            // If added new attribute with valid field
                            if (!typedText.equals(defaultNewAttributeName) && typedText != null) {
                                if (!newAttributeText.equals(defaultNewAttributeValue) &&
                                        newAttributeText != null) {
                                    if (echoOption.isSelected()) {
                                        customAnnotations.addAllAttributeValues(typedText, newAttributeText);
                                    } else {
                                        customAnnotations.addAllAttributeValues(typedText, "null");
                                        feature.setAttribute(typedText, newAttributeText);
                                    }
                                }
                            }
                            // Update old attribute
                        } else {
                            // If text changed in already existing attributes, change value
                            if (!typedText.equals(feature.getAttribute(key))) {
                                feature.setAttribute(key, typedText);
                            }
                        }
                    }
                }
            }
            typedText = null;
            clearAndHide();
        }
    }

    /**
     * This method clears the dialog and hides it.
     */
    private void clearAndHide() {
        //textField.setText(null);
        setVisible(false);
    }
}

