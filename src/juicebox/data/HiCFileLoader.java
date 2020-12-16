/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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

package juicebox.data;

import juicebox.HiCGlobals;
import juicebox.gui.SuperAdapter;
import juicebox.windowui.LoadDialog;
import juicebox.windowui.RecentMenu;
import org.broad.igv.Globals;
import org.broad.igv.ui.util.FileDialogUtils;
import org.broad.igv.ui.util.MessageUtils;
import org.broad.igv.util.ParsingUtils;

import javax.swing.*;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.InputStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Properties;
import java.util.prefs.Preferences;

/**
 * Created by muhammadsaadshamim on 8/4/15.
 */
public class HiCFileLoader {

    private static Properties properties;
    private static LoadDialog loadDialog = null;
    private static String propertiesFileURL = null;
    private static final String RECENT_PROPERTIES_FILE = "recentPropertiesFile";

    public static File loadMenuItemActionPerformed(SuperAdapter superAdapter, boolean control, File openHiCPath) {
        FilenameFilter hicFilter = new FilenameFilter() {
            public boolean accept(File dir, String name) {
                return name.toLowerCase().endsWith(".hic");
            }
        };

        File[] files = FileDialogUtils.chooseMultiple("Choose Hi-C file(s)", openHiCPath, hicFilter);
        if (files != null && files.length > 0) {
            List<String> fileNames = new ArrayList<>();
            StringBuilder str = new StringBuilder();
            String path = "";
            for (File f : files) {
                fileNames.add(f.getAbsolutePath());
                str.append(f.getName()).append(" ");
                path = f.getAbsolutePath();
            }
            openHiCPath = new File(path);
            superAdapter.addRecentMapMenuEntry(str.toString().trim() + RecentMenu.delimiter + files[0].getAbsolutePath(), true);
            superAdapter.safeLoad(fileNames, control, str.toString());
        }
        return openHiCPath;
    }

    public static void loadFromRecentActionPerformed(SuperAdapter superAdapter, String url, String title,
                                                     boolean control) {

        if (url != null) {
            superAdapter.addRecentMapMenuEntry(title.trim() + RecentMenu.delimiter + url, true);
            superAdapter.safeLoad(Collections.singletonList(url), control, title);
        }
    }

    public static void loadFromURLActionPerformed(SuperAdapter superAdapter, boolean control) {
        String urlString = JOptionPane.showInputDialog("Enter URLs (separated by commas): ");

        if (urlString != null && urlString.length() > 0) {
            try {
                if (HiCFileTools.isDropboxURL(urlString)) {
                    urlString = HiCFileTools.cleanUpDropboxURL(urlString);
                }
                urlString = urlString.trim();
                String[] urls = urlString.split(",");
                List<String> urlList = new ArrayList<>();
                StringBuilder title = new StringBuilder();
                for (String url : urls) {
                    urlList.add(url);
                    title.append((new URL(url)).getPath()).append(" ");
                }
                superAdapter.safeLoad(urlList, control, title.toString());
            } catch (MalformedURLException e1) {
                superAdapter.launchFileLoadingError(urlString);
            }
        }
    }

    public static void loadFromListActionPerformed(SuperAdapter superAdapter, boolean control) {

        if (loadDialog == null) {
            initProperties();
            loadDialog = superAdapter.launchLoadFileDialog(properties);
            if (!loadDialog.getSuccess()) {
                loadDialog = null;
                return;
            }
        }
        loadDialog.setControl(control);
        loadDialog.setVisible(true);
    }

    private static void initProperties() {
        try {
            if (propertiesFileURL == null) {
                try {
                    Preferences prefs = Preferences.userNodeForPackage(Globals.class);
                    String potentialURL = prefs.get(RECENT_PROPERTIES_FILE, null);
                    if (potentialURL != null && potentialURL.length() > 0 && potentialURL.endsWith(".properties")) {
                        propertiesFileURL = potentialURL;
                    }
                } catch (Exception ignored) {
                }
                if (propertiesFileURL == null) propertiesFileURL = System.getProperty("jnlp.loadMenu");
                if (propertiesFileURL == null) propertiesFileURL = HiCGlobals.defaultPropertiesURL;
            }
            InputStream is = ParsingUtils.openInputStream(propertiesFileURL);
            properties = new Properties();
            if (is != null) {
                properties.load(is);
            }
        } catch (Exception error) {
            boolean fileFailedToLoad = true;
            if (!propertiesFileURL.equals(HiCGlobals.defaultPropertiesURL)) {
                try {
                    loadPropertiesViaURL(HiCGlobals.defaultPropertiesURL);
                    fileFailedToLoad = false;
                } catch (Exception ignored) {
                }
            }
            if (fileFailedToLoad) {
                System.err.println("Can't find properties file for loading list - internet likely disconnected" + error.getLocalizedMessage());
            }
        }
    }

    private static void loadPropertiesViaURL(String url) throws IOException {
        InputStream is = ParsingUtils.openInputStream(url);
        properties = new Properties();
        if (is != null) {
            properties.load(is);
        }
    }

    public static void changeJuiceboxPropertiesFile(String newURL) {
        boolean providedURLIsValid = true;
        try {
            InputStream is = ParsingUtils.openInputStream(newURL);
            properties = new Properties();
            if (is != null) {
                properties.load(is);
            }
        } catch (Exception error) {
            providedURLIsValid = false;
            if (HiCGlobals.guiIsCurrentlyActive) {
                SuperAdapter.showMessageDialog("Can't find/load specified properties file");
            } else {
                MessageUtils.showErrorMessage("Can't find/load specified properties file", error);
            }
        }

        // if no exception has been thrown at this point, the url is a valid one
        if (providedURLIsValid) {
            setPropertiesFileURL(newURL);
            loadDialog = null;
        }
    }

    private static void setPropertiesFileURL(String propertiesFileURL) {
        HiCFileLoader.propertiesFileURL = propertiesFileURL;
        Preferences prefs = Preferences.userNodeForPackage(Globals.class);
        prefs.put(RECENT_PROPERTIES_FILE, propertiesFileURL);
    }
}
