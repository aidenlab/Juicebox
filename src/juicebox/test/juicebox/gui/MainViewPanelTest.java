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

package juicebox.gui;

import juicebox.HiC;
import juicebox.MainWindow;
import org.broad.igv.feature.Chromosome;
import org.fest.swing.core.BasicRobot;
import org.fest.swing.core.Robot;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import javax.swing.*;
import java.awt.*;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.StringSelection;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.Arrays;
import java.util.concurrent.ConcurrentLinkedQueue;

import static java.awt.Toolkit.getDefaultToolkit;
import static org.junit.Assert.*;

/**
 * Created by mikeehman on 6/14/16.
 */
public class MainViewPanelTest {

    private final static String testURL = "https://hicfiles.s3.amazonaws.com/hiseq/hela/in-situ/combined.hic";
    public static ConcurrentLinkedQueue<Thread> threadQueue;
    private static MainWindow mainWindow;
    private static SuperAdapter superAdapter;

    /**
     * Open the application ready to be tested
     */
    @BeforeClass
    public static void setUp() throws InterruptedException {
        try {
            // start the GUI application
            MainWindow.main(new String[1]);
            mainWindow = (MainWindow) Window.getWindows()[0];
            superAdapter = mainWindow.getSuperAdapter();
            threadQueue = MainWindow.threadQueue;
            superAdapter.safeLoad(Arrays.asList(testURL), false, "test");

            // hopefully this allows all threads to be registered
            while (!threadQueue.isEmpty()) {
                Thread t = threadQueue.poll();
                t.join();
            }

        } catch (IOException e) {
            e.printStackTrace();
        } catch (InvocationTargetException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        } finally {
            assertTrue(threadQueue.isEmpty());

        }

    }

    /**
     * Uses a robot to terminate the application
     *
     * @throws AWTException
     */
    @AfterClass
    public static void tearDown() throws AWTException {
        Robot robot = BasicRobot.robotWithNewAwtHierarchy();
        robot.cleanUp();
    }


    /**
     * Reset the view panel to All by All
     */
    public void setToChromosomesByIndex(int i1, int i2) throws InterruptedException {
        // test whether "All" and "All" are in the combo boxes
        MainViewPanel mvp = superAdapter.getMainViewPanel();
        assertNotNull(mvp);

        JComboBox<Chromosome> chr1Box = mvp.getChrBox1();
        JComboBox<Chromosome> chr2Box = mvp.getChrBox2();
        assertNotNull(chr1Box);
        assertNotNull(chr2Box);

        chr1Box.setSelectedIndex(i1);
        chr2Box.setSelectedIndex(i2);
        superAdapter.safeRefreshButtonActionPerformed();

        while (!threadQueue.isEmpty()) {
            threadQueue.poll().join();
        }

        Chromosome chr1 = (Chromosome) chr1Box.getSelectedItem();
        Chromosome chr2 = (Chromosome) chr2Box.getSelectedItem();

        assertNotNull(chr1.getName());
        assertNotNull(chr2.getName());

    }

    /**
     * Test whether initial load of a .hic file gives All by All view
     *
     * @throws AWTException
     */
    @Test
    public void isWholeGenomeTest() throws AWTException, InterruptedException {

        Robot robot = BasicRobot.robotWithNewAwtHierarchy();

        // just being paranoid
        assertTrue(robot.isActive());
        robot.focus(mainWindow);
        assertNotNull(mainWindow.getHiC());

        // test whether "All" and "All" are in the combo boxes
        MainViewPanel mvp = superAdapter.getMainViewPanel();
        assertNotNull(mvp);

        JComboBox<Chromosome> chr1Box = mvp.getChrBox1();
        JComboBox<Chromosome> chr2Box = mvp.getChrBox2();
        assertNotNull(chr1Box);
        assertNotNull(chr2Box);

        Chromosome chr1 = (Chromosome) chr1Box.getSelectedItem();
        Chromosome chr2 = (Chromosome) chr2Box.getSelectedItem();

        assertNotNull(chr1.getName());
        assertNotNull(chr2.getName());

        assertEquals("Initial view is not All by All", "All", chr1.getName());
        assertEquals("Initial view is not All by All", "All", chr2.getName());
//        assertEquals("Initial view is not All by All", "All", tmp.getText());
        
        robot.cleanUpWithoutDisposingWindows();
    }

    /**
     * Test if the normalization field is being enabled and disabled correctly
     *
     * @throws InterruptedException
     */
    @Test
    public void normalizationFieldTest() throws InterruptedException {
        Robot robot = BasicRobot.robotWithNewAwtHierarchy();

        // first, let's check whether All by All disables normalization field correctly
        try {
            setToChromosomesByIndex(0, 0);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        // must be NOT enabled
        assertFalse(superAdapter.getMainViewPanel().getNormalizationComboBox().isEnabled());


        try {
            setToChromosomesByIndex(1, 1);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        MainViewPanel mvp = superAdapter.getMainViewPanel();
        assertNotNull(mvp);

        JComboBox<Chromosome> chr1Box = mvp.getChrBox1();
        JComboBox<Chromosome> chr2Box = mvp.getChrBox2();
        assertNotNull(chr1Box);
        assertNotNull(chr2Box);

        Chromosome chr1 = (Chromosome) chr1Box.getSelectedItem();
        Chromosome chr2 = (Chromosome) chr2Box.getSelectedItem();

        assertNotNull(chr1.getName());
        assertNotNull(chr2.getName());

        assertEquals("View is not chr1 by chr1", "1", chr1.getName());
        assertEquals("View is not chr1 by chr1", "1", chr2.getName());

        // must be enabled
        assertTrue(superAdapter.getMainViewPanel().getNormalizationComboBox().isEnabled());


        /* Check issue #334: Goto */
        try {
            setToChromosomesByIndex(0, 0);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        robot.moveMouse(mvp.getHeatmapPanel());
        robot.click(mvp.getHeatmapPanel(), new Point(100, 100));
        Thread.sleep(2000);

        // set the Goto x and y boxes
        HiC hic = superAdapter.getHiC();
        StringSelection stringSelection = new StringSelection(hic.getXPosition());
        superAdapter.setPositionChrTop(hic.getXPosition().concat(":").concat(String.valueOf(hic.getXContext().getZoom().getBinSize())));
        superAdapter.setPositionChrLeft(hic.getYPosition().concat(":").concat(String.valueOf(hic.getYContext().getZoom().getBinSize())));
        Clipboard clpbrd = getDefaultToolkit().getSystemClipboard();
        clpbrd.setContents(stringSelection, null);

        assertTrue(mvp.getGoPanel().isEnabled());

        // click the go button
        mvp.getGoPanel().getGoButton().doClick();
        while (!threadQueue.isEmpty()) {
            threadQueue.poll().join();
        }

        // check that the normalization field is enabled
        assertTrue(mvp.getNormalizationComboBox().isEnabled());

        /* Check issue #376: isWholeGenome checking just the names not the state */
        // right now we're zoomed in
        // choose All by All
        chr1Box.setSelectedIndex(0);
        chr2Box.setSelectedIndex(1);
        while (!threadQueue.isEmpty()) {
            threadQueue.poll().join();
        }
        // click on the go button
        mvp.getGoPanel().getGoButton().doClick();
        while (!threadQueue.isEmpty()) {
            threadQueue.poll().join();
        }

        // normalization field must be enabled
        assertTrue(mvp.getNormalizationComboBox().isEnabled());

        // sync issue
        hic.broadcastLocation();
        assertTrue(mvp.getNormalizationComboBox().isEnabled());



        robot.cleanUpWithoutDisposingWindows();
    }

}