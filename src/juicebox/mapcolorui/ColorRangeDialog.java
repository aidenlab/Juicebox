/*
 * Created by JFormDesigner on Tue Jan 03 22:58:44 EST 2012
 */

package juicebox.mapcolorui;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;
import java.text.DecimalFormat;
import java.text.ParseException;

/**
 * @author Jim Robinson
 */
public class ColorRangeDialog extends JDialog {

    private static final long serialVersionUID = -2570891809264626823L;
    private final RangeSlider colorSlider;
    private final double colorRangeFactor;
    private final DecimalFormat df1;
    private final DecimalFormat df2;
    private final boolean isObserved;
    private JTextField minimumField;
    private JTextField maximumField;

    public ColorRangeDialog(Frame owner, RangeSlider colorSlider, double colorRangeFactor, boolean isObserved) {
        super(owner);
        initComponents(isObserved);
        this.colorSlider = colorSlider;
        if (!isObserved)  colorRangeFactor = 8;
        this.colorRangeFactor = colorRangeFactor;
        this.isObserved = isObserved;


        df1 = new DecimalFormat( "#,###,###,##0" );
        df2 = new DecimalFormat("##.##");

        if (isObserved) {
            minimumField.setText(df1.format(colorSlider.getMinimum() / colorRangeFactor));
            maximumField.setText(df1.format(colorSlider.getMaximum() / colorRangeFactor));
        }
        else {
            minimumField.setText(df2.format(1/(colorSlider.getMaximum()/colorRangeFactor)));
            maximumField.setText(df2.format(colorSlider.getMaximum()/colorRangeFactor));
        }
        //tickSpacingField.setText(df.format(colorSlider.getMajorTickSpacing() / colorRangeFactor));
    }


    private void okButtonActionPerformed(ActionEvent e)  {
        double max = 0;
        double min = 0;

        try {
            if (isObserved) {
                max = df1.parse(maximumField.getText()).doubleValue();
                min = df1.parse(minimumField.getText()).doubleValue();
            }
            else {
                max = df2.parse(maximumField.getText()).doubleValue();
            }
        }
        catch (ParseException error) {
            JOptionPane.showMessageDialog(this, "Must enter a number", "Error", JOptionPane.ERROR_MESSAGE);
            return;
        }

        int iMin;
        int iMax;
        if (isObserved) {
            iMin = (int) (colorRangeFactor * min);
            iMax = (int) (colorRangeFactor * max);
        }
        else {
            iMax = (int) (max * colorRangeFactor);
            iMin = (int) (colorRangeFactor/max);
        }
        colorSlider.setMinimum(iMin);
        colorSlider.setMaximum(iMax);
        setVisible(false);
        //double tickSpacing = Double.parseDouble(tickSpacingField.getText());
        //int iTickSpacing = (int) Math.max(1, (colorRangeFactor * tickSpacing));
        //colorSlider.setMajorTickSpacing(iTickSpacing);
        //colorSlider.setMinorTickSpacing(iTickSpacing);

    }

    private void cancelButtonActionPerformed(ActionEvent e) {
        setVisible(false);
    }

    private void initComponents(final boolean isObserved) {

        JPanel dialogPane = new JPanel();
        JPanel panel3 = new JPanel();
        JLabel label1 = new JLabel();
        JPanel contentPanel = new JPanel();
        JPanel panel2 = new JPanel();
        JLabel label4 = new JLabel();
        minimumField = new JTextField();
        JPanel panel1 = new JPanel();
        JLabel label2 = new JLabel();
        maximumField = new JTextField();

        JPanel buttonBar = new JPanel();
        JButton okButton = new JButton();
        JButton cancelButton = new JButton();

        Container contentPane = getContentPane();
        contentPane.setLayout(new BorderLayout());


        dialogPane.setBorder(new EmptyBorder(12, 12, 12, 12));
        dialogPane.setLayout(new BorderLayout());


        panel3.setLayout(new FlowLayout(FlowLayout.LEFT, 5, 25));

        //---- label1 ----
        label1.setText("Set color slider control range");
        panel3.add(label1);

        dialogPane.add(panel3, BorderLayout.NORTH);


        contentPanel.setLayout(new BoxLayout(contentPanel, BoxLayout.Y_AXIS));

        //======== panel2 ========
        {
            panel2.setLayout(new FlowLayout(FlowLayout.LEFT));

            //---- label4 ----
            label4.setText("Minimum:");
            panel2.add(label4);

            //---- minimumField ----
            minimumField.setMaximumSize(new Dimension(100, 20));
            minimumField.setMinimumSize(new Dimension(100, 20));
            minimumField.setPreferredSize(new Dimension(100, 20));
            panel2.add(minimumField);
        }
        if (!isObserved) {
            minimumField.setEnabled(false);

        }
        contentPanel.add(panel2);

        panel1.setLayout(new FlowLayout(FlowLayout.LEADING));

        //---- label2 ----
        label2.setText("Maximum");
        panel1.add(label2);

        //---- maximumField ----
        maximumField.setMaximumSize(new Dimension(100, 20));
        maximumField.setMinimumSize(new Dimension(100, 20));
        maximumField.setPreferredSize(new Dimension(100, 20));
        panel1.add(maximumField);
        maximumField.addFocusListener(new FocusListener() {
            @Override
            public void focusGained(FocusEvent focusEvent) {
                //To change body of implemented methods use File | Settings | File Templates.
            }

            @Override
            public void focusLost(FocusEvent focusEvent) {
                if (!isObserved) {
                    double max;
                    try {
                        max = df2.parse(maximumField.getText()).doubleValue();
                    } catch (ParseException error) {
                        return;
                    }
                    minimumField.setText(df2.format(1/max));
                }
            }
        });

        contentPanel.add(panel1);

        dialogPane.add(contentPanel, BorderLayout.CENTER);

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

        dialogPane.add(buttonBar, BorderLayout.SOUTH);

        contentPane.add(dialogPane, BorderLayout.CENTER);
        pack();
        setLocationRelativeTo(getOwner());
    }

}
