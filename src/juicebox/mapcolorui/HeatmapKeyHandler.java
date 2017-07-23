/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2017 Broad Institute, Aiden Lab
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

package juicebox.mapcolorui;

import juicebox.HiCGlobals;
import juicebox.MainWindow;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.InputEvent;
import java.awt.event.KeyEvent;

/**
 * Created by ranganmostofa on 7/23/17.
 */
public class HeatmapKeyHandler {
    private HeatmapPanel heatmapPanel;
    private InputMap inputMap;
    private ActionMap actionMap;

    public HeatmapKeyHandler(HeatmapPanel heatmapPanel) {
        this.heatmapPanel = heatmapPanel;
        this.inputMap = heatmapPanel.getInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW);
        this.actionMap = heatmapPanel.getActionMap();
    }

    public void createKeyBindings() {
        populateInputMap();
        populateActionMap();
    }

    private void populateInputMap() {
        for (Keys key : Keys.values()) {
            inputMap.put(key.getKeyStroke(), key.getText());
        }
        heatmapPanel.setInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW, inputMap);
    }

    private void populateActionMap() {

        actionMap.put(Keys.ESCAPE.getText(), new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (HiCGlobals.heatmapCaptureModeEnabled) {
                    HiCGlobals.heatmapCaptureModeEnabled = Boolean.FALSE;
                    heatmapPanel.setCursor(Cursor.getDefaultCursor());
//                    System.out.println("Escape");
                }
            }
        });

        actionMap.put(Keys.SHIFT_CTRL_3.getText(), new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (!HiCGlobals.heatmapCaptureModeEnabled) {
                    HiCGlobals.heatmapCaptureModeEnabled = Boolean.TRUE;
                    heatmapPanel.setCursor(MainWindow.screenshotCursor);
//                    System.out.println("3");
                }
            }
        });

        actionMap.put(Keys.SHIFT_CTRL_4.getText(), new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (HiCGlobals.heatmapCaptureModeEnabled) {
                    HiCGlobals.heatmapCaptureModeEnabled = Boolean.FALSE;
                    heatmapPanel.setCursor(Cursor.getDefaultCursor());
//                    System.out.println("4");
                }
                heatmapPanel.captureHeatmapImage(Boolean.TRUE);
            }
        });

        heatmapPanel.setActionMap(actionMap);
    }

    private enum Keys {
        ESCAPE("Escape", KeyStroke.getKeyStroke(KeyEvent.VK_ESCAPE, 0)),
        SHIFT_CTRL_3("Shift-Control-3", KeyStroke.getKeyStroke(KeyEvent.VK_3, InputEvent.SHIFT_DOWN_MASK + InputEvent.CTRL_DOWN_MASK)),
        SHIFT_CTRL_4("Shift-Control-4", KeyStroke.getKeyStroke(KeyEvent.VK_4, InputEvent.SHIFT_DOWN_MASK + InputEvent.CTRL_DOWN_MASK));

        private String text;
        private KeyStroke keyStroke;

        Keys(String text, KeyStroke keyStroke) {
            this.text = text;
            this.keyStroke = keyStroke;
        }

        public String getText() {
            return text;
        }

        public KeyStroke getKeyStroke() {
            return keyStroke;
        }

        @Override
        public String toString() {
            return text;
        }
    }
}
