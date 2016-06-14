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

import juicebox.MainWindow;
import org.broad.igv.feature.Chromosome;
import org.fest.swing.core.BasicRobot;
import org.fest.swing.core.Robot;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.Arrays;
import java.util.concurrent.ConcurrentLinkedQueue;

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
    public static synchronized void setUp() throws InterruptedException {
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
    public synchronized void resetToAllByAll() throws InterruptedException {
        // test whether "All" and "All" are in the combo boxes
        MainViewPanel mvp = superAdapter.getMainViewPanel();
        assertNotNull(mvp);

        JComboBox<Chromosome> chr1Box = mvp.getChrBox1();
        JComboBox<Chromosome> chr2Box = mvp.getChrBox2();
        assertNotNull(chr1Box);
        assertNotNull(chr2Box);

        chr1Box.setSelectedIndex(0);
        chr2Box.setSelectedIndex(0);
        superAdapter.safeRefreshButtonActionPerformed();

        while (!threadQueue.isEmpty()) {
            threadQueue.poll().join();
        }

        Chromosome chr1 = (Chromosome) chr1Box.getSelectedItem();
        Chromosome chr2 = (Chromosome) chr2Box.getSelectedItem();

        assertNotNull(chr1.getName());
        assertNotNull(chr2.getName());


        assertEquals("View is not All by All", "All", chr1.getName());
        assertEquals("View is not All by All", "All", chr2.getName());
    }

    /**
     * Test whether initial load of a .hic file gives All by All view
     *
     * @throws AWTException
     */
    @Test
    public synchronized void isWholeGenomeTest() throws AWTException, InterruptedException {

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

    @Test
    public synchronized void normalizationFieldTest() {
        Robot robot = BasicRobot.robotWithNewAwtHierarchy();

        // first, let's check whether All by All disables normalization field correctly
        try {
            resetToAllByAll();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        // must be NOT enabled
        assertFalse(superAdapter.getMainViewPanel().getNormalizationComboBox().isEnabled());

        // now, let's change the view to chr1 by chr1


        robot.cleanUpWithoutDisposingWindows();
    }


    @Test
    public void template() {
        Robot robot = BasicRobot.robotWithNewAwtHierarchy();

        robot.cleanUpWithoutDisposingWindows();
    }

}