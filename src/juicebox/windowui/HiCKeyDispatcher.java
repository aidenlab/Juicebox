/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2021 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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
 *  FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.windowui;

import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.data.HiCFileLoader;
import juicebox.gui.SuperAdapter;
import juicebox.tools.dev.Private;
import juicebox.track.feature.AnnotationLayerHandler;
import org.broad.igv.ui.util.MessageUtils;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyEvent;
import java.util.ArrayList;
import java.util.List;

public class HiCKeyDispatcher implements KeyEventDispatcher {

    private final HiC hic;
    private final JComboBox<MatrixType> displayOptionComboBox;
    private final SuperAdapter superAdapter;
    private final List<AnnotationLayerHandler> handlersPreviouslyHidden = new ArrayList<>();

    public HiCKeyDispatcher(SuperAdapter superAdapter, HiC hic, JComboBox<MatrixType> displayOptionComboBox) {
        super();
        this.hic = hic;
        this.superAdapter = superAdapter;
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
        } else if (e.getID() == KeyEvent.KEY_PRESSED && e.getKeyCode() == KeyEvent.VK_F2) {
            if (handlersPreviouslyHidden.size() > 0) {
                for (AnnotationLayerHandler handler : handlersPreviouslyHidden) {
                    handler.setLayerVisibility(true);
                }
                handlersPreviouslyHidden.clear();
            } else {
                for (AnnotationLayerHandler handler : superAdapter.getAllLayers()) {
                    if (handler.getLayerVisibility()) {
                        handler.setLayerVisibility(false);
                        handlersPreviouslyHidden.add(handler);
                    }
                }
            }
            superAdapter.updateMiniAnnotationsLayerPanel();
            superAdapter.updateMainLayersPanel();
            superAdapter.repaint();
            return true;
        } else if (e.getID() == KeyEvent.KEY_PRESSED && e.getKeyCode() == KeyEvent.VK_F3) {
            for (AnnotationLayerHandler handler : superAdapter.getAllLayers()) {
                if (handler.getLayerVisibility()) {
                    handler.setIsEnlarged(!handler.getIsEnlarged());
                }
            }
            superAdapter.repaint();
            return true;
        } else if (e.getID() == KeyEvent.KEY_PRESSED && e.getKeyCode() == KeyEvent.VK_F4) {
            for (AnnotationLayerHandler handler : superAdapter.getAllLayers()) {
                if (handler.getLayerVisibility()) {
                    handler.setIsTransparent(!handler.getIsTransparent());
                }
            }
            superAdapter.repaint();
            return true;
        } else if (e.getID() == KeyEvent.KEY_PRESSED && e.getKeyCode() == KeyEvent.VK_F5) {
            for (AnnotationLayerHandler handler : superAdapter.getAllLayers()) {
                if (handler.getLayerVisibility()) {
                    handler.setIsSparse(!handler.getIsSparse());
                }
            }
            superAdapter.repaint();
            return true;
        } else if (e.getID() == KeyEvent.KEY_PRESSED && e.getKeyCode() == KeyEvent.VK_F6) {
            for (AnnotationLayerHandler handler : superAdapter.getAllLayers()) {
                if (handler.getLayerVisibility()) {
                    handler.togglePlottingStyle();
                }
            }
            superAdapter.repaint();
            return true;
        } else if (e.getID() == KeyEvent.KEY_PRESSED && e.getKeyCode() == KeyEvent.VK_F7) {
            String newURL = MessageUtils.showInputDialog("Specify a new properties file",
                    HiCGlobals.defaultPropertiesURL);
            if (newURL != null) {
                HiCFileLoader.changeJuiceboxPropertiesFile(newURL);
            }
            return true;
        } else if (e.getID() == KeyEvent.KEY_PRESSED && e.getKeyCode() == KeyEvent.VK_F8) {
            Private.launchMapSubsetGUI(superAdapter);
            return true;
        } else if (e.getID() == KeyEvent.KEY_PRESSED && e.getKeyCode() == KeyEvent.VK_F9) {
            superAdapter.togglePanelVisible();
            return true;
        } else if (e.getID() == KeyEvent.KEY_PRESSED && (e.getKeyCode() == KeyEvent.VK_U) && ((e.getModifiers() & Toolkit.getDefaultToolkit().getMenuShortcutKeyMask()) != 0)) {
            if (SuperAdapter.assemblyModeCurrentlyActive && superAdapter.getAssemblyStateTracker().checkUndo()) {
                superAdapter.getAssemblyStateTracker().undo();
                superAdapter.getHeatmapPanel().removeSelection();
                superAdapter.refresh();
            }
            return true;
        } else if (e.getID() == KeyEvent.KEY_PRESSED && e.getExtendedKeyCode() == KeyEvent.VK_R && ((e.getModifiers() & Toolkit.getDefaultToolkit().getMenuShortcutKeyMask()) != 0)) {
            if (SuperAdapter.assemblyModeCurrentlyActive && superAdapter.getAssemblyStateTracker().checkRedo()) {
                superAdapter.getAssemblyStateTracker().redo();
                superAdapter.getHeatmapPanel().removeSelection();
                superAdapter.refresh();
            }
            return true;
        } else {
            return false;
        }
    }
}
