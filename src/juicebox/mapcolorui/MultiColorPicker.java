package juicebox.mapcolorui;

import juicebox.MainWindow;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.FileOutputStream;
import java.io.PrintStream;

import javax.swing.*;
import javax.swing.colorchooser.AbstractColorChooserPanel;

public class MultiColorPicker extends JDialog implements ActionListener {
    private static final long serialVersionUID = -678567876;
    JFrame generator = new JFrame("Choose Palette");

    //JPanel preview = new JPanel();
    JButton[] bColor = new JButton[24];
    JButton[] bChoose = new JButton[24];
    JButton[] bDelete = new JButton[24];
    JColorChooser chooser = new JColorChooser();

    JPanel preview = new JPanel();
    JPanel prvPanel1 = new JPanel();
    JPanel prvPanel2 = new JPanel();
    JPanel prvPanel3 = new JPanel();

    JPanel chooserPanel = new JPanel();
    JButton bOk = new JButton("OK");
    JButton bCancel = new JButton("Cancel");


    /**
     * @param args
     */
    public static void main(String[] args) {
        // TODO Auto-generated method stub
        MultiColorPicker g = new MultiColorPicker();
    }

    public MultiColorPicker() {

        generator.setResizable(false);
        generator.setLayout(new BoxLayout(generator.getContentPane(), 1));

        chooser.setSize(new Dimension(690, 270));
        chooserPanel.setMaximumSize(new Dimension(690, 270));

        AbstractColorChooserPanel[] accp = chooser.getChooserPanels();
        chooser.removeChooserPanel(accp[0]);
        chooser.removeChooserPanel(accp[1]);
        chooser.removeChooserPanel(accp[2]);
        chooser.removeChooserPanel(accp[4]);

        chooser.setPreviewPanel(new JPanel());

        chooserPanel.add(chooser);

        chooserPanel.setBorder(BorderFactory.createTitledBorder("Color Chooser"));
        //scrollContainer.setBorder(BorderFactory.createTitledBorder("Preview"));

        prvPanel1.add(new JLabel("RGB"));
        prvPanel2.add(new JLabel("Pick"));
        prvPanel3.add(new JLabel("Clear"));

        Color defaultColor = generator.getBackground();

        for (int x = 0; x < 24; x++) {
            bColor[x] = new JButton();
            bColor[x].setBackground(defaultColor);
            bColor[x].setBorder(BorderFactory.createLineBorder(Color.DARK_GRAY, 1));
            bColor[x].setOpaque(true);
            bChoose[x] = new JButton("+");
            bDelete[x] = new JButton("-");

            bColor[x].setPreferredSize(new Dimension(15, 15));
            prvPanel1.add(bColor[x]);

            bChoose[x].setPreferredSize(new Dimension(15, 15));
            bDelete[x].setPreferredSize(new Dimension(15, 15));
            bColor[x].addActionListener(this);
            bChoose[x].addActionListener(this);
            bDelete[x].addActionListener(this);
            bOk.addActionListener(this);
            bCancel.addActionListener(this);
            prvPanel2.add(bChoose[x]);

            prvPanel3.add(bDelete[x]);
        }

        prvPanel1.setPreferredSize(new Dimension(600, 30));
        prvPanel2.setPreferredSize(new Dimension(600, 30));
        prvPanel3.setPreferredSize(new Dimension(600, 30));

        generator.getContentPane().add(chooserPanel);
        preview.add(prvPanel1);
        preview.add(prvPanel2);
        preview.add(prvPanel3);
        generator.add(preview);

        JPanel okCancel = new JPanel();

        okCancel.add(bOk);
        okCancel.add(bCancel);

        generator.add(okCancel);

                generator.setSize(new Dimension(690, 500));
        generator.setVisible(true);
    }

    public void initValue(Color[] colorArray){
        for (int cIdx=0; cIdx < colorArray.length && cIdx < bColor.length;cIdx++)
        {
            bColor[cIdx].setBackground(colorArray[cIdx]);
        }
    }

    public void actionPerformed(ActionEvent e) {

        JButton clkButton = ((JButton) e.getSource());
        Color defaultColor = generator.getBackground();


        if (clkButton == bColor[0]) {chooser.setColor(bColor[0].getBackground());}
        else if (clkButton == bColor[1]) {chooser.setColor(bColor[1].getBackground());}
        else if (clkButton == bColor[2]) {chooser.setColor(bColor[2].getBackground());}
        else if (clkButton == bColor[3]) {chooser.setColor(bColor[3].getBackground());}
        else if (clkButton == bColor[4]) {chooser.setColor(bColor[4].getBackground());}
        else if (clkButton == bColor[5]) {chooser.setColor(bColor[5].getBackground());}
        else if (clkButton == bColor[6]) {chooser.setColor(bColor[6].getBackground());}
        else if (clkButton == bColor[7]) {chooser.setColor(bColor[7].getBackground());}
        else if (clkButton == bColor[8]) {chooser.setColor(bColor[8].getBackground());}
        else if (clkButton == bColor[9]) {chooser.setColor(bColor[9].getBackground());}
        else if (clkButton == bColor[10]) {chooser.setColor(bColor[10].getBackground());}
        else if (clkButton == bColor[11]) {chooser.setColor(bColor[11].getBackground());}
        else if (clkButton == bColor[12]) {chooser.setColor(bColor[12].getBackground());}
        else if (clkButton == bColor[13]) {chooser.setColor(bColor[13].getBackground());}
        else if (clkButton == bColor[14]) {chooser.setColor(bColor[14].getBackground());}
        else if (clkButton == bColor[15]) {chooser.setColor(bColor[15].getBackground());}
        else if (clkButton == bColor[16]) {chooser.setColor(bColor[16].getBackground());}
        else if (clkButton == bColor[17]) {chooser.setColor(bColor[17].getBackground());}
        else if (clkButton == bColor[18]) {chooser.setColor(bColor[18].getBackground());}
        else if (clkButton == bColor[19]) {chooser.setColor(bColor[19].getBackground());}
        else if (clkButton == bColor[20]) {chooser.setColor(bColor[20].getBackground());}
        else if (clkButton == bColor[21]) {chooser.setColor(bColor[21].getBackground());}
        else if (clkButton == bColor[22]) {chooser.setColor(bColor[22].getBackground());}
        else if (clkButton == bColor[23]) {chooser.setColor(bColor[23].getBackground());}

        else if (clkButton == bChoose[0]) {bColor[0].setBackground(chooser.getColor());}
        else if (clkButton == bChoose[1]) {bColor[1].setBackground(chooser.getColor());}
        else if (clkButton == bChoose[2]) {bColor[2].setBackground(chooser.getColor());}
        else if (clkButton == bChoose[3]) {bColor[3].setBackground(chooser.getColor());}
        else if (clkButton == bChoose[4]) {bColor[4].setBackground(chooser.getColor());}
        else if (clkButton == bChoose[5]) {bColor[5].setBackground(chooser.getColor());}
        else if (clkButton == bChoose[6]) {bColor[6].setBackground(chooser.getColor());}
        else if (clkButton == bChoose[7]) {bColor[7].setBackground(chooser.getColor());}
        else if (clkButton == bChoose[8]) {bColor[8].setBackground(chooser.getColor());}
        else if (clkButton == bChoose[9]) {bColor[9].setBackground(chooser.getColor());}
        else if (clkButton == bChoose[10]) {bColor[10].setBackground(chooser.getColor());}
        else if (clkButton == bChoose[11]) {bColor[11].setBackground(chooser.getColor());}
        else if (clkButton == bChoose[12]) {bColor[12].setBackground(chooser.getColor());}
        else if (clkButton == bChoose[13]) {bColor[13].setBackground(chooser.getColor());}
        else if (clkButton == bChoose[14]) {bColor[14].setBackground(chooser.getColor());}
        else if (clkButton == bChoose[15]) {bColor[15].setBackground(chooser.getColor());}
        else if (clkButton == bChoose[16]) {bColor[16].setBackground(chooser.getColor());}
        else if (clkButton == bChoose[17]) {bColor[17].setBackground(chooser.getColor());}
        else if (clkButton == bChoose[18]) {bColor[18].setBackground(chooser.getColor());}
        else if (clkButton == bChoose[19]) {bColor[19].setBackground(chooser.getColor());}
        else if (clkButton == bChoose[20]) {bColor[20].setBackground(chooser.getColor());}
        else if (clkButton == bChoose[21]) {bColor[21].setBackground(chooser.getColor());}
        else if (clkButton == bChoose[22]) {bColor[22].setBackground(chooser.getColor());}
        else if (clkButton == bChoose[23]) {bColor[23].setBackground(chooser.getColor());}

        else if (clkButton == bDelete[0]) {bColor[0].setBackground(defaultColor);}
        else if (clkButton == bDelete[1]) {bColor[1].setBackground(defaultColor);}
        else if (clkButton == bDelete[2]) {bColor[2].setBackground(defaultColor);}
        else if (clkButton == bDelete[3]) {bColor[3].setBackground(defaultColor);}
        else if (clkButton == bDelete[4]) {bColor[4].setBackground(defaultColor);}
        else if (clkButton == bDelete[5]) {bColor[5].setBackground(defaultColor);}
        else if (clkButton == bDelete[6]) {bColor[6].setBackground(defaultColor);}
        else if (clkButton == bDelete[7]) {bColor[7].setBackground(defaultColor);}
        else if (clkButton == bDelete[8]) {bColor[8].setBackground(defaultColor);}
        else if (clkButton == bDelete[9]) {bColor[9].setBackground(defaultColor);}
        else if (clkButton == bDelete[10]) {bColor[10].setBackground(defaultColor);}
        else if (clkButton == bDelete[11]) {bColor[11].setBackground(defaultColor);}
        else if (clkButton == bDelete[12]) {bColor[12].setBackground(defaultColor);}
        else if (clkButton == bDelete[13]) {bColor[13].setBackground(defaultColor);}
        else if (clkButton == bDelete[14]) {bColor[14].setBackground(defaultColor);}
        else if (clkButton == bDelete[15]) {bColor[15].setBackground(defaultColor);}
        else if (clkButton == bDelete[16]) {bColor[16].setBackground(defaultColor);}
        else if (clkButton == bDelete[17]) {bColor[17].setBackground(defaultColor);}
        else if (clkButton == bDelete[18]) {bColor[18].setBackground(defaultColor);}
        else if (clkButton == bDelete[19]) {bColor[19].setBackground(defaultColor);}
        else if (clkButton == bDelete[20]) {bColor[20].setBackground(defaultColor);}
        else if (clkButton == bDelete[21]) {bColor[21].setBackground(defaultColor);}
        else if (clkButton == bDelete[22]) {bColor[22].setBackground(defaultColor);}
        else if (clkButton == bDelete[23]) {bColor[23].setBackground(defaultColor);}
        else if (clkButton == bCancel) {
            //MainWindow.getInstance()...;
            //Exit:
            // setVisible(false);
        }
        else if (clkButton == bOk) {
            Color[] tmpColor = new Color[24];
            for (int idx =0; idx < bColor.length; idx++)
            {
                tmpColor[idx] = bColor[idx].getBackground();
            }
            MainWindow.getInstance().preDefMapColorPalette = tmpColor;
        }
    }
}
