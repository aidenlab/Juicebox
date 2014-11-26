package juicebox.windowui;

import juicebox.HiC;
import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyEvent;

public class HiCKeyDispatcher implements KeyEventDispatcher {

    private HiC hic;
    private JComboBox<MatrixType> displayOptionComboBox;

    public HiCKeyDispatcher(HiC hic, JComboBox<MatrixType> displayOptionComboBox){
        super();
        this.hic = hic;
        this.displayOptionComboBox = displayOptionComboBox;
    }

    @Override
    public boolean dispatchKeyEvent(KeyEvent e) {

        if (e.getID() == KeyEvent.KEY_PRESSED && e.getKeyCode() == KeyEvent.VK_F1) {

            if (hic.getControlZd() != null) {
                MatrixType displayOption = (MatrixType) displayOptionComboBox.getSelectedItem();
                if (displayOption == MatrixType.CONTROL) {
                    displayOptionComboBox.setSelectedItem(MatrixType.OBSERVED);

                } else if (displayOption == MatrixType.OBSERVED) {
                    displayOptionComboBox.setSelectedItem(MatrixType.CONTROL);
                }

            }
            return true;
        } else {

            return false;
        }
    }
}
