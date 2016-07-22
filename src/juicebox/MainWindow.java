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

package juicebox;


import juicebox.gui.MainMenuBar;
import juicebox.gui.MainViewPanel;
import juicebox.gui.SuperAdapter;
import juicebox.windowui.DisabledGlassPane;
import juicebox.windowui.FileDropTargetListener;
import org.apache.log4j.Logger;
import org.broad.igv.Globals;
import org.broad.igv.ui.util.IconFactory;

import javax.swing.*;
import java.awt.*;
import java.awt.dnd.DropTarget;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class MainWindow extends JFrame {

    private static final long serialVersionUID = -3654174199024388185L;
    private static final Logger log = Logger.getLogger(MainWindow.class);
    private static final DisabledGlassPane disabledGlassPane = new DisabledGlassPane(Cursor.WAIT_CURSOR);
    private static final SuperAdapter superAdapter = new SuperAdapter();
    public static Cursor fistCursor;
    public static Color hicMapColor = Color.red;
    private static MainWindow theInstance;
    private final ExecutorService threadExecutor = Executors.newFixedThreadPool(1);
    private final HiC hic; // The "model" object containing the state for this instance.

    private MainWindow() {
        setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        HiCGlobals.guiIsCurrentlyActive = true;
        hic = new HiC(superAdapter);
        MainMenuBar mainMenuBar = new MainMenuBar();
        MainViewPanel mainViewPanel = new MainViewPanel();
        superAdapter.setAdapters(this, hic, mainMenuBar, mainViewPanel);

        initComponents();
        createCursors();
        pack();

        DropTarget target = new DropTarget(this, new FileDropTargetListener(superAdapter));
        setDropTarget(target);

        // Tooltip settings
        ToolTipManager.sharedInstance().setDismissDelay(60000);   // 60 seconds
        KeyboardFocusManager.getCurrentKeyboardFocusManager().addKeyEventDispatcher(superAdapter.getNewHiCKeyDispatcher());

        hicMapColor = Color.red;
    }

    private static MainWindow createMainWindow() {
        return new MainWindow();
    }

    public static synchronized MainWindow getInstance() {
        if (theInstance == null) {
            try {
                theInstance = createMainWindow();
            } catch (Exception e) {
                log.error("Error creating main window", e);
            }
        }
        return theInstance;
    }

    public static void main(String[] args) throws IOException, InvocationTargetException, InterruptedException {
        initApplication();
        Runnable runnable = new Runnable() {
            public void run() {
                theInstance = getInstance();
                theInstance.setVisible(true);
                theInstance.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
                CommandListener.start(theInstance.hic);
            }
        };
        SwingUtilities.invokeAndWait(runnable);

    }

    private static void initApplication() {

        DirectoryManager.initializeLog();

        log.debug("Default User Directory: " + DirectoryManager.getUserDirectory());
        System.setProperty("http.agent", Globals.applicationString());
    }

    private void initComponents() {

        addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent e) {
                superAdapter.exitActionPerformed();
            }
        });

        if (HiCGlobals.printVerboseComments) {
            System.out.println("Initializing Components");
        }

        MainWindow.superAdapter.initializeCustomAnnotations();

        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();

        Insets scnMax = Toolkit.getDefaultToolkit().getScreenInsets(getGraphicsConfiguration());
        int taskBarHeight = scnMax.bottom;

        Container contentPane = getContentPane();

        Dimension bigPanelDim = new Dimension(screenSize.width - getWidth() - 230,
                screenSize.height - taskBarHeight - getHeight() - 120);

        Dimension panelDim = new Dimension(screenSize.width - getWidth() - 300,
                screenSize.height - taskBarHeight - getHeight());

        MainWindow.superAdapter.initializeMainView(contentPane, bigPanelDim, panelDim);

        initializeGlassPaneListening();
    }

    private void createCursors() {
        BufferedImage handImage = new BufferedImage(32, 32, BufferedImage.TYPE_INT_ARGB);

        // Make background transparent
        Graphics2D g = handImage.createGraphics();
        g.setComposite(AlphaComposite.getInstance(AlphaComposite.CLEAR, 0.0f));
        Rectangle2D.Double rect = new Rectangle2D.Double(0, 0, 32, 32);
        g.fill(rect);

        // Draw hand image in middle
        g = handImage.createGraphics();
        g.drawImage(IconFactory.getInstance().getIcon(IconFactory.IconID.FIST).getImage(), 0, 0, null);
        fistCursor = getToolkit().createCustomCursor(handImage, new Point(8, 6), "Move");
    }


    public void exitActionPerformed() {
        setVisible(false);
        dispose();
        System.out.println("Exiting Main Window");
        System.exit(0);
    }

    /**
     * Utility function to execute a task in a worker thread.  The method is on MainWindow because the glassPane
     * is used to display a wait cursor and block events.
     *
     * @param runnable Thread
     * @return thread
     */
    public Future<?> executeLongRunningTask(final Runnable runnable, final String caller) {
        return executeLongRunningTask(runnable, caller, "Loading...");
    }

    public Future<?> executeLongRunningTask(final Runnable runnable, final String caller, final String message) {
        if (HiCGlobals.printVerboseComments) {
            System.out.println("long_execute " + caller);
        }
        Callable<Object> wrapper = new Callable<Object>() {
            public Object call() throws Exception {
                MainWindow.this.showDisabledGlassPane(caller, message);
                try {
                    runnable.run();
                    return "done";
                }
                catch (Exception error) {
                    error.printStackTrace();
                    return "error";
                }
                finally {
                    MainWindow.this.hideDisabledGlassPane(caller);
                }
            }
        };

        return threadExecutor.submit(wrapper);
    }

    private void showDisabledGlassPane(String caller, String displayMessage) {
        disabledGlassPane.activate(displayMessage);
        if (HiCGlobals.printVerboseComments) {
            System.out.println("Loading " + caller);
        }
    }

    private void initializeGlassPaneListening() {
        rootPane.setGlassPane(disabledGlassPane);
    }

    private void hideDisabledGlassPane(String caller) {//getRootPane().getContentPane()
        if (HiCGlobals.printVerboseComments) {
            System.out.println("Done loading " + caller);
        }
        disabledGlassPane.deactivate();
    }

    public void updateNamesFromImport(String path) {
        superAdapter.updatePrevStateNameFromImport(path);
    }
}


