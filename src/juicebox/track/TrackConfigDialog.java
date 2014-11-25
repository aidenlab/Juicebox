/*
 * Created by JFormDesigner on Thu Aug 08 17:01:58 EDT 2013
 */

package juicebox.track;

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.*;

import org.broad.igv.renderer.DataRange;
import org.broad.igv.track.WindowFunction;
import org.broad.igv.ui.color.*;

/**
 * @author Jim Robinson
 */
public class TrackConfigDialog extends JDialog {

    private static final long serialVersionUID = -1778029293180119209L;
    private boolean canceled = false;
    private final HiCTrack track;

    public TrackConfigDialog(Frame owner, HiCTrack track) {
        super(owner);
        initComponents();

        this.track = track;

        nameField.setText(track.getName());
        posColorChooser.setSelectedColor(track.getPosColor());

        if (track instanceof HiCDataTrack) {
            minYField.setEnabled(true);
            maxYField.setEnabled(true);
            logScaleCB.setEnabled(true);
            altColorChooser.setEnabled(true);
            maxRB.setEnabled(true);
            meanRB.setEnabled(true);

            HiCDataTrack dataTrack = (HiCDataTrack) track;
            minYField.setText(String.valueOf(dataTrack.getDataRange().getMinimum()));
            maxYField.setText(String.valueOf(dataTrack.getDataRange().getMaximum()));
            logScaleCB.setSelected(dataTrack.getDataRange().isLog());
            altColorChooser.setSelectedColor(dataTrack.getAltColor());

            if(!dataTrack.getAvailableWindowFunctions().contains(WindowFunction.max)) {
                maxRB.setEnabled(false);
            }

            if (dataTrack.getWindowFunction() == WindowFunction.max) {
                maxRB.setSelected(true);
            } else {
                meanRB.setSelected(true);
            }

        } else {
            minYField.setEnabled(false);
            maxYField.setEnabled(false);
            logScaleCB.setEnabled(false);
            altColorChooser.setEnabled(false);
            maxRB.setEnabled(false);
            meanRB.setEnabled(false);
        }
    }

    private void minYFieldFocusLost(FocusEvent e) {
        if (track instanceof HiCDataTrack) {
            if (!validateNumeric(minYField.getText())) {
                DataRange dr = ((HiCDataTrack) track).getDataRange();
                minYField.setText(String.valueOf(dr.getMinimum()));
            }
        }
    }

    private void maxYFieldFocusLost(FocusEvent e) {
        if (track instanceof HiCDataTrack) {
            if (!validateNumeric(maxYField.getText())) {
                DataRange dr = ((HiCDataTrack) track).getDataRange();
                minYField.setText(String.valueOf(dr.getMaximum()));
            }
        }
    }

    public boolean isCanceled() {
        return canceled;
    }

    private void okButtonActionPerformed(ActionEvent e) {

        if (validateNumeric(minYField.getText()) && validateNumeric(maxYField.getText())) {

            track.setName(nameField.getText());
            track.setColor(posColorChooser.getSelectedColor());
            if (track instanceof HiCDataTrack) {
                float newMin = Float.parseFloat(minYField.getText());
                float newMax = Float.parseFloat(maxYField.getText());
                DataRange newDataRange = new DataRange(newMin, newMax);
                newDataRange.setType(logScaleCB.isSelected() ? DataRange.Type.LOG : DataRange.Type.LINEAR);
                ((HiCDataTrack) track).setDataRange(newDataRange);
                track.setAltColor(altColorChooser.getSelectedColor());

                WindowFunction wf = maxRB.isSelected() ? WindowFunction.max : WindowFunction.mean;
                ((HiCDataTrack) track).setWindowFunction(wf);
            }
            canceled = false;
            setVisible(false);
        }
    }

    private void cancelButtonActionPerformed(ActionEvent e) {
        canceled = true;
        setVisible(false);
    }

    private void initComponents() {
        // JFormDesigner - Component initialization - DO NOT MODIFY  //GEN-BEGIN:initComponents
        // Generated using JFormDesigner non-commercial license

        nameField = new JTextField();
        logScaleCB = new JCheckBox();
        minYField = new JTextField();
        maxYField = new JTextField();
        posColorChooser = new ColorChooserPanel();
        altColorChooser = new ColorChooserPanel();
        meanRB = new JRadioButton();
        maxRB = new JRadioButton();

        JPanel dialogPane = new JPanel();
        JPanel contentPanel = new JPanel();
        JPanel panel4 = new JPanel();
        JPanel panel1 = new JPanel();
        JLabel label2 = new JLabel();
        JLabel label3 = new JLabel();
        JPanel panel2 = new JPanel();
        JLabel label4 = new JLabel();
        JLabel label5 = new JLabel();
        JPanel panel3 = new JPanel();
        JPanel buttonBar = new JPanel();
        JButton okButton = new JButton();
        JButton cancelButton = new JButton();

        //======== this ========
        setModalityType(Dialog.ModalityType.APPLICATION_MODAL);
        setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        setAlwaysOnTop(true);
        Container contentPane = getContentPane();
        contentPane.setLayout(new BorderLayout());

        //======== dialogPane ========
        {
            dialogPane.setBorder(new EmptyBorder(12, 12, 12, 12));
            dialogPane.setLayout(new BorderLayout());

            //======== contentPanel ========
            {
                contentPanel.setLayout(null);

                //======== panel4 ========
                {
                    panel4.setBorder(new TitledBorder("Track Name"));
                    panel4.setLayout(null);
                    panel4.add(nameField);
                    nameField.setBounds(20, 25, 500, nameField.getPreferredSize().height);

                    { // compute preferred size
                        Dimension preferredSize = new Dimension();
                        for (int i = 0; i < panel4.getComponentCount(); i++) {
                            Rectangle bounds = panel4.getComponent(i).getBounds();
                            preferredSize.width = Math.max(bounds.x + bounds.width, preferredSize.width);
                            preferredSize.height = Math.max(bounds.y + bounds.height, preferredSize.height);
                        }
                        Insets insets = panel4.getInsets();
                        preferredSize.width += insets.right;
                        preferredSize.height += insets.bottom;
                        panel4.setMinimumSize(preferredSize);
                        panel4.setPreferredSize(preferredSize);
                    }
                }
                contentPanel.add(panel4);
                panel4.setBounds(0, 0, 535, 67);

                //======== panel1 ========
                {
                    panel1.setBorder(new TitledBorder("Y Axis"));
                    panel1.setLayout(null);

                    //---- label2 ----
                    label2.setText("Min");
                    panel1.add(label2);
                    label2.setBounds(new Rectangle(new Point(30, 30), label2.getPreferredSize()));

                    //---- label3 ----
                    label3.setText("Max");
                    panel1.add(label3);
                    label3.setBounds(new Rectangle(new Point(30, 70), label3.getPreferredSize()));

                    //---- logScaleCB ----
                    logScaleCB.setText("Log scale");
                    logScaleCB.setEnabled(false);
                    panel1.add(logScaleCB);
                    logScaleCB.setBounds(new Rectangle(new Point(30, 110), logScaleCB.getPreferredSize()));

                    //---- minYField ----
                    minYField.addFocusListener(new FocusAdapter() {
                        @Override
                        public void focusLost(FocusEvent e) {
                            minYFieldFocusLost(e);
                        }
                    });
                    panel1.add(minYField);
                    minYField.setBounds(100, 24, 170, minYField.getPreferredSize().height);

                    //---- maxYField ----
                    maxYField.addFocusListener(new FocusAdapter() {
                        @Override
                        public void focusLost(FocusEvent e) {
                            maxYFieldFocusLost(e);
                        }
                    });
                    panel1.add(maxYField);
                    maxYField.setBounds(100, 64, 170, 28);

                    { // compute preferred size
                        Dimension preferredSize = new Dimension();
                        for (int i = 0; i < panel1.getComponentCount(); i++) {
                            Rectangle bounds = panel1.getComponent(i).getBounds();
                            preferredSize.width = Math.max(bounds.x + bounds.width, preferredSize.width);
                            preferredSize.height = Math.max(bounds.y + bounds.height, preferredSize.height);
                        }
                        Insets insets = panel1.getInsets();
                        preferredSize.width += insets.right;
                        preferredSize.height += insets.bottom;
                        panel1.setMinimumSize(preferredSize);
                        panel1.setPreferredSize(preferredSize);
                    }
                }
                contentPanel.add(panel1);
                panel1.setBounds(0, 69, 535, 167);

                //======== panel2 ========
                {
                    panel2.setBorder(new TitledBorder("Colors"));
                    panel2.setLayout(null);

                    //---- label4 ----
                    label4.setText("Positive values");
                    panel2.add(label4);
                    label4.setBounds(new Rectangle(new Point(25, 30), label4.getPreferredSize()));

                    //---- label5 ----
                    label5.setText("Negative values");
                    panel2.add(label5);
                    label5.setBounds(25, 60, 105, 16);
                    panel2.add(posColorChooser);
                    posColorChooser.setBounds(160, 27, 55, posColorChooser.getPreferredSize().height);
                    panel2.add(altColorChooser);
                    altColorChooser.setBounds(new Rectangle(new Point(160, 57), altColorChooser.getPreferredSize()));

                    { // compute preferred size
                        Dimension preferredSize = new Dimension();
                        for (int i = 0; i < panel2.getComponentCount(); i++) {
                            Rectangle bounds = panel2.getComponent(i).getBounds();
                            preferredSize.width = Math.max(bounds.x + bounds.width, preferredSize.width);
                            preferredSize.height = Math.max(bounds.y + bounds.height, preferredSize.height);
                        }
                        Insets insets = panel2.getInsets();
                        preferredSize.width += insets.right;
                        preferredSize.height += insets.bottom;
                        panel2.setMinimumSize(preferredSize);
                        panel2.setPreferredSize(preferredSize);
                    }
                }
                contentPanel.add(panel2);
                panel2.setBounds(0, 235, 535, 95);

                //======== panel3 ========
                {
                    panel3.setBorder(new TitledBorder("Data Reduction Function"));
                    panel3.setAlignmentX(0.0F);
                    panel3.setLayout(new FlowLayout(FlowLayout.LEFT));

                    //---- meanRB ----
                    meanRB.setText("Mean");
                    panel3.add(meanRB);

                    //---- maxRB ----
                    maxRB.setText("Max");
                    panel3.add(maxRB);
                }
                contentPanel.add(panel3);
                panel3.setBounds(0, 335, 535, 89);

                { // compute preferred size
                    Dimension preferredSize = new Dimension();
                    for (int i = 0; i < contentPanel.getComponentCount(); i++) {
                        Rectangle bounds = contentPanel.getComponent(i).getBounds();
                        preferredSize.width = Math.max(bounds.x + bounds.width, preferredSize.width);
                        preferredSize.height = Math.max(bounds.y + bounds.height, preferredSize.height);
                    }
                    Insets insets = contentPanel.getInsets();
                    preferredSize.width += insets.right;
                    preferredSize.height += insets.bottom;
                    contentPanel.setMinimumSize(preferredSize);
                    contentPanel.setPreferredSize(preferredSize);
                }
            }
            dialogPane.add(contentPanel, BorderLayout.CENTER);

            //======== buttonBar ========
            {
                buttonBar.setBorder(new EmptyBorder(12, 0, 0, 0));
                buttonBar.setLayout(new GridBagLayout());
                ((GridBagLayout) buttonBar.getLayout()).columnWidths = new int[]{0, 85, 80};
                ((GridBagLayout) buttonBar.getLayout()).columnWeights = new double[]{1.0, 0.0, 0.0};

                //---- okButton ----
                okButton.setText("OK");
                okButton.addActionListener(new ActionListener() {
                    @Override
                    public void actionPerformed(ActionEvent e) {
                        okButtonActionPerformed(e);
                    }
                });
                buttonBar.add(okButton, new GridBagConstraints(1, 0, 1, 1, 0.0, 0.0,
                        GridBagConstraints.CENTER, GridBagConstraints.BOTH,
                        new Insets(0, 0, 0, 5), 0, 0));

                //---- cancelButton ----
                cancelButton.setText("Cancel");
                cancelButton.addActionListener(new ActionListener() {
                    @Override
                    public void actionPerformed(ActionEvent e) {
                        cancelButtonActionPerformed(e);
                    }
                });
                buttonBar.add(cancelButton, new GridBagConstraints(2, 0, 1, 1, 0.0, 0.0,
                        GridBagConstraints.CENTER, GridBagConstraints.BOTH,
                        new Insets(0, 0, 0, 0), 0, 0));
            }
            dialogPane.add(buttonBar, BorderLayout.SOUTH);
        }
        contentPane.add(dialogPane, BorderLayout.CENTER);
        setSize(565, 510);
        setLocationRelativeTo(getOwner());

        //---- dataReductionGroup ----
        ButtonGroup dataReductionGroup = new ButtonGroup();
        dataReductionGroup.add(meanRB);
        dataReductionGroup.add(maxRB);
        // JFormDesigner - End of component initialization  //GEN-END:initComponents
    }


    private boolean validateNumeric(String text) {
        try {
            Double.parseDouble(text);
            return true;
        } catch (NumberFormatException e) {
            JOptionPane.showMessageDialog(this, "Error: numeric value required (" + text + ")");
            return false;
        }
    }


    private ColorChooserPanel posColorChooser;
    private ColorChooserPanel altColorChooser;
    private JCheckBox logScaleCB;
    private JTextField minYField;
    private JTextField maxYField;
    private JRadioButton meanRB;
    private JRadioButton maxRB;
    private JTextField nameField;

    // JFormDesigner - Variables declaration - DO NOT MODIFY  //GEN-BEGIN:variables
    // Generated using JFormDesigner non-commercial license
    /*
    private JPanel dialogPane;
    private JPanel contentPanel;
    private JPanel panel4;
    private JPanel panel1;
    private JLabel label2;
    private JLabel label3;
    private JPanel panel2;
    private JLabel label4;
    private JLabel label5;
    private JPanel panel3;
    private JPanel buttonBar;
    private JButton okButton;
    private JButton cancelButton;
    */
    // JFormDesigner - End of variables declaration  //GEN-END:variables
}
