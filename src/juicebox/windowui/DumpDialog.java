/*
 * Copyright (C) 2011-2014 Aiden Lab - All Rights Reserved
 * SOFTWARE COPYRIGHT NOTICE
 * This software and its documentation are the copyright of the Aiden Lab All rights are reserved.
 *
 * This software is supplied without any warranty or guaranteed support whatsoever. Aiden Lab is not responsible for its use, misuse, or functionality.
 *
 * This software is licensed under the terms of the GNU Lesser General Public License (LGPL),
 * Version 2.1 which is available at http://www.opensource.org/licenses/lgpl-2.1.php.
 */

package juicebox.windowui;

import juicebox.DirectoryManager;
import juicebox.HiC;
import juicebox.MainWindow;
import juicebox.data.ExpectedValueFunction;
import juicebox.data.NormalizationVector;
import juicebox.tools.HiCTools;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;
import java.io.PrintWriter;

public class DumpDialog extends JFileChooser {
    static final long serialVersionUID = 42L;
    private JComboBox<String> box;

    public DumpDialog(MainWindow mainWindow, HiC hic) {
        super();
        int result = showSaveDialog(mainWindow);
        if (result == JFileChooser.APPROVE_OPTION) {
            try {
                if (box.getSelectedItem().equals("Matrix")) {
                    if (hic.getDisplayOption() == MatrixType.OBSERVED) {
                        double[] nv1 = null;
                        double[] nv2 = null;
                        if (!(hic.getNormalizationType() == NormalizationType.NONE)) {
                            NormalizationVector nv = hic.getNormalizationVector(hic.getZd().getChr1Idx());
                            nv1 = nv.getData();
                            if (hic.getZd().getChr1Idx() != hic.getZd().getChr2Idx()) {
                                nv = hic.getNormalizationVector(hic.getZd().getChr2Idx());
                                nv2 = nv.getData();
                            } else {
                                nv2 = nv1;
                            }
                        }
                        hic.getZd().dump(new PrintWriter(getSelectedFile()), nv1, nv2);

                    } else if (hic.getDisplayOption() == MatrixType.OE || hic.getDisplayOption() == MatrixType.PEARSON) {
                        final ExpectedValueFunction df = hic.getDataset().getExpectedValues(hic.getZd().getZoom(),
                                hic.getNormalizationType());
                        if (df == null) {
                            JOptionPane.showMessageDialog(this, box.getSelectedItem() + " not available", "Error",
                                    JOptionPane.ERROR_MESSAGE);
                            return;
                        }
                        if (hic.getDisplayOption() == MatrixType.OE) {
                            hic.getZd().dumpOE(df, "oe",
                                    hic.getNormalizationType(), null, new PrintWriter(getSelectedFile()));
                        } else {
                            hic.getZd().dumpOE(df, "pearson",
                                    hic.getNormalizationType(), null, new PrintWriter(getSelectedFile()));
                        }
                    }

                } else if (box.getSelectedItem().equals("Norm vector")) {

                    if (hic.getNormalizationType() == NormalizationType.NONE) {
                        JOptionPane.showMessageDialog(this, "Selected normalization is None, nothing to write",
                                "Error", JOptionPane.ERROR_MESSAGE);
                    } else {
                        NormalizationVector nv = hic.getNormalizationVector(hic.getZd().getChr1Idx());
                        HiCTools.dumpVector(new PrintWriter(getSelectedFile()), nv.getData(), false);
                    }
                } else if (box.getSelectedItem().toString().contains("Expected")) {

                    final ExpectedValueFunction df = hic.getDataset().getExpectedValues(hic.getZd().getZoom(),
                            hic.getNormalizationType());
                    if (df == null) {
                        JOptionPane.showMessageDialog(this, box.getSelectedItem() + " not available", "Error",
                                JOptionPane.ERROR_MESSAGE);
                        return;
                    }

                    if (box.getSelectedItem().equals("Expected vector")) {
                        int length = df.getLength();
                        int c = hic.getZd().getChr1Idx();
                        PrintWriter pw = new PrintWriter(getSelectedFile());
                        for (int i = 0; i < length; i++) {
                            pw.println((float) df.getExpectedValue(c, i));
                        }
                        pw.flush();
                    } else {
                        HiCTools.dumpVector(new PrintWriter(getSelectedFile()), df.getExpectedValues(), false);
                    }
                } else if (box.getSelectedItem().equals("Eigenvector")) {
                    int chrIdx = hic.getZd().getChr1Idx();
                    double[] eigenvector = hic.getEigenvector(chrIdx, 0);

                    if (eigenvector != null) {
                        HiCTools.dumpVector(new PrintWriter(getSelectedFile()), eigenvector, true);
                    }
                }
            } catch (IOException error) {
                JOptionPane.showMessageDialog(this, "Error while writing:\n" + error, "Error", JOptionPane.ERROR_MESSAGE);
            }
        }
    }

    protected JDialog createDialog(Component component) throws HeadlessException {
        JDialog dialog = super.createDialog(component);
        JPanel panel1 = new JPanel();
        JLabel label = new JLabel("Dump ");
        box = new JComboBox<String>(new String[]{"Matrix", "Norm vector", "Expected vector", "Expected genome-wide vector", "Eigenvector"});
        panel1.add(label);
        panel1.add(box);
        dialog.add(panel1, BorderLayout.NORTH);
        setCurrentDirectory(DirectoryManager.getUserDirectory());
        setDialogTitle("Choose location for dump of matrix or vector");
        setFileSelectionMode(JFileChooser.FILES_ONLY);
        return dialog;
    }

}
