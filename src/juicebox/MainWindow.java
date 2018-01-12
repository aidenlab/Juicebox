/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2018 Broad Institute, Aiden Lab
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


import juicebox.gui.MainViewPanel;
import juicebox.gui.SuperAdapter;
import juicebox.windowui.DisabledGlassPane;
import juicebox.windowui.FileDropTargetListener;
import juicebox.windowui.layers.LayersPanel;
import org.broad.igv.Globals;
import org.broad.igv.ui.util.IconFactory;

import javax.swing.*;
import java.awt.*;
import java.awt.dnd.DropTarget;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.io.*;
import java.lang.reflect.InvocationTargetException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLConnection;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class MainWindow extends JFrame {

    private static final long serialVersionUID = -3654174199024388185L;
    private static final DisabledGlassPane disabledGlassPane = new DisabledGlassPane(Cursor.WAIT_CURSOR);
    private static final SuperAdapter superAdapter = new SuperAdapter();
    public static Cursor fistCursor;
    public static Cursor pasteNECursor;
    public static Cursor pasteSWCursor;
    public static Cursor invertNECursor;
    public static Cursor invertSWCursor;
    public static Cursor scissorCursor;
    public static Color hicMapColor = Color.red;
    private static MainWindow theInstance;
    private final ExecutorService threadExecutor = Executors.newFixedThreadPool(1);
    private final HiC hic; // The "model" object containing the state for this instance.

    private MainWindow() {
        setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        HiCGlobals.guiIsCurrentlyActive = true;
        hic = new HiC(superAdapter);
        MainViewPanel mainViewPanel = new MainViewPanel();
        superAdapter.setAdapters(this, hic, mainViewPanel);

        initComponents();
        createCursors();
      setExtendedState(JFrame.MAXIMIZED_BOTH);
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
                System.err.println("Error creating main window " + e.getLocalizedMessage());
            }
        }
        return theInstance;
    }

    public static void main(String[] args) throws InvocationTargetException, InterruptedException {
        initApplication();

     //   Runnable runnable = new Runnable() {
          //  public void run() {
                theInstance = getInstance();
                theInstance.setVisible(true);
                theInstance.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
                CommandListener.start(theInstance.hic);
        //    }
       // };
       // SwingUtilities.invokeAndWait(runnable);
      /*  URL url;
        try {
            url = new URL("https://s3.amazonaws.com/hicfiles.tc4ga.com/juicebox.version");
            URLConnection next = url.openConnection();
            InputStream is = next.getInputStream();
            BufferedReader reader = new BufferedReader(new InputStreamReader(is));
            String latestVersion = reader.readLine();
            String[] latest = latestVersion.split("\\.");
            String[] current = HiCGlobals.versionNum.split("\\.");
            boolean isOutdated = false;

            int iC = Integer.valueOf(current[0]);
            int iL = Integer.valueOf(latest[0]);

            if (iC < iL) {
                isOutdated = true;
            } else if (iC == iL) {
                int jC = Integer.valueOf(current[1]);
                int jL = Integer.valueOf(latest[1]);
                int kC = Integer.valueOf(current[2]);
                int kL = Integer.valueOf(latest[2]);
                if (jC < jL) {
                    isOutdated = true;
                } else if (jC == jL && kC < kL) {
                    isOutdated = true;
                }
            }

            if (isOutdated) {
                JPanel textPanel = new JPanel(new GridLayout(0, 1));
                JLabel label = new JLabel("<html><p> You are using Juicebox " + HiCGlobals.versionNum + "<br>The lastest version is "
                        + latestVersion + "<br>To download the lastest version, go to</p></html>");
                JLabel label2 = new JLabel("<html><a href=\"https://github.com/theaidenlab/juicebox/wiki/Download\"> https://github.com/theaidenlab/juicebox/wiki/Download </a></html>");
                textPanel.add(label);
                textPanel.add(label2);
                label2.setCursor(new Cursor(Cursor.HAND_CURSOR));
                label2.addMouseListener(new MouseAdapter() {
                    @Override
                    public void mouseClicked(MouseEvent e) {
                        try {
                            Desktop.getDesktop().browse(new URI("https://github.com/theaidenlab/juicebox/wiki/Download"));
                        } catch (URISyntaxException | IOException ex) {
                            //It looks like there's a problem
                        }
                    }
                });
                JOptionPane.showMessageDialog(superAdapter.getMainWindow(), textPanel, "Update Information", JOptionPane.PLAIN_MESSAGE);
            }

        } catch (Exception e) {
        }        */

    }

    private static void initApplication() {
        System.err.println("Default User Directory: " + DirectoryManager.getUserDirectory());

        try {
            HiCGlobals.stateFile = new File(DirectoryManager.getHiCDirectory(), "CurrentJuiceboxStates");
            HiCGlobals.xmlSavedStatesFile = new File(DirectoryManager.getHiCDirectory(),
                    "JuiceboxStatesForExport.xml");
        } catch (Exception e) {
            System.err.println(e.getLocalizedMessage());
            if (HiCGlobals.guiIsCurrentlyActive) {
                SuperAdapter.showMessageDialog("Error with state file\n" + e.getLocalizedMessage());
            }
        }

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

        // first annotation layer must get created
        MainWindow.superAdapter.createNewLayer(null);

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
        ImageIcon icon = new ImageIcon(getClass().getResource("/images/juicebox256.png"));
        setIconImage(icon.getImage());
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

        // Additional cursors for assembly
        ImageIcon imageIcon;

        // Insert (paste) prompts
        BufferedImage pasteNEImage = new BufferedImage(32, 32, BufferedImage.TYPE_INT_ARGB);
        g = pasteNEImage.createGraphics();
        g.setComposite(AlphaComposite.getInstance(AlphaComposite.CLEAR, 0.0f));
        rect = new Rectangle2D.Double(0, 0, 32, 32);
        g.fill(rect);
        g = pasteNEImage.createGraphics();
        imageIcon = new ImageIcon(this.getClass().getResource("/images/assembly/small-ne-paste.png"), "paste");
        g.drawImage(imageIcon.getImage(), 0, 0, null);
        pasteNECursor = getToolkit().createCustomCursor(pasteNEImage, new Point(8, 6), "PasteNE");

        // Insert (paste) prompts
        BufferedImage pasteSWImage = new BufferedImage(32, 32, BufferedImage.TYPE_INT_ARGB);
        g = pasteSWImage.createGraphics();
        g.setComposite(AlphaComposite.getInstance(AlphaComposite.CLEAR, 0.0f));
        rect = new Rectangle2D.Double(0, 0, 32, 32);
        g.fill(rect);
        g = pasteSWImage.createGraphics();
        imageIcon = new ImageIcon(this.getClass().getResource("/images/assembly/small-sw-paste.png"), "paste");
        g.drawImage(imageIcon.getImage(), 0, 0, null);
        pasteSWCursor = getToolkit().createCustomCursor(pasteSWImage, new Point(8, 6), "PasteSW");

        // Invert prompts
        BufferedImage invertNEImage = new BufferedImage(32, 32, BufferedImage.TYPE_INT_ARGB);
        g = invertNEImage.createGraphics();
        g.setComposite(AlphaComposite.getInstance(AlphaComposite.CLEAR, 0.0f));
        rect = new Rectangle2D.Double(0, 0, 32, 32);
        g.fill(rect);
        g = invertNEImage.createGraphics();
        imageIcon = new ImageIcon(this.getClass().getResource("/images/assembly/small-ne-invert.png"), "invert");
        g.drawImage(imageIcon.getImage(), 0, 0, null);
        invertNECursor = getToolkit().createCustomCursor(invertNEImage, new Point(8, 6), "InvertNE");

        // Invert prompts
        BufferedImage invertSWImage = new BufferedImage(32, 32, BufferedImage.TYPE_INT_ARGB);
        g = invertSWImage.createGraphics();
        g.setComposite(AlphaComposite.getInstance(AlphaComposite.CLEAR, 0.0f));
        rect = new Rectangle2D.Double(0, 0, 32, 32);
        g.fill(rect);
        g = invertSWImage.createGraphics();
        imageIcon = new ImageIcon(this.getClass().getResource("/images/assembly/small-sw-invert.png"), "invert");
        g.drawImage(imageIcon.getImage(), 0, 0, null);
        invertSWCursor = getToolkit().createCustomCursor(invertSWImage, new Point(8, 6), "InvertSW");

        // Cut prompts
        BufferedImage scissorImage = new BufferedImage(32, 32, BufferedImage.TYPE_INT_ARGB);
        g = scissorImage.createGraphics();
        g.setComposite(AlphaComposite.getInstance(AlphaComposite.CLEAR, 0.0f));
        rect = new Rectangle2D.Double(0, 0, 32, 32);
        g.fill(rect);
        g = scissorImage.createGraphics();
        imageIcon = new ImageIcon(this.getClass().getResource("/images/assembly/small-scissors.png"), "invert");
        g.drawImage(imageIcon.getImage(), 0, 0, null);
        scissorCursor = getToolkit().createCustomCursor(scissorImage, new Point(8, 6), "Scissors");
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
            public Object call() {
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
        LayersPanel.disabledGlassPane.activate(displayMessage);
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
        LayersPanel.disabledGlassPane.deactivate();
    }

    public void updateNamesFromImport(String path) {
        superAdapter.updatePrevStateNameFromImport(path);
    }
}


