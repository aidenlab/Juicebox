/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2015 Broad Institute, Aiden Lab
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

import juicebox.DirectoryManager;
import juicebox.gui.SuperAdapter;
import juicebox.windowui.LoadDialog;
import org.apache.log4j.Logger;
import org.broad.igv.ui.util.FileDialogUtils;
import org.broad.igv.util.ParsingUtils;

import javax.swing.*;
import java.io.File;
import java.io.FilenameFilter;
import java.io.InputStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;

/**
 * Created by muhammadsaadshamim on 8/4/15.
 */
public class HiCFileLoader {

    private static final Logger log = Logger.getLogger(HiCFileLoader.class);
    private static Properties properties;
    private static LoadDialog loadDialog = null;

    public static void loadMenuItemActionPerformed(SuperAdapter superAdapter, boolean control) {
        FilenameFilter hicFilter = new FilenameFilter() {
            public boolean accept(File dir, String name) {
                return name.toLowerCase().endsWith(".hic");
            }
        };

        File[] files = FileDialogUtils.chooseMultiple("Choose Hi-C file(s)", DirectoryManager.getUserDirectory(),
                hicFilter);
        if (files != null && files.length > 0) {
            List<String> fileNames = new ArrayList<String>();
            String str = "";
            for (File f : files) {
                fileNames.add(f.getAbsolutePath());
                str += f.getName() + " ";
            }
            superAdapter.addRecentMapMenuEntry(str.trim() + "@@" + files[0].getAbsolutePath(), true);
            superAdapter.safeLoad(fileNames, control, str);
        }
    }

    public static void loadFromRecentActionPerformed(SuperAdapter superAdapter, String url, String title,
                                                     boolean control) {

        if (url != null) {
            superAdapter.addRecentMapMenuEntry(title.trim() + "@@" + url, true);
            superAdapter.safeLoad(Arrays.asList(url), control, title);
        }
    }

    public static void loadFromURLActionPerformed(SuperAdapter superAdapter, boolean control) {
        String urlString = JOptionPane.showInputDialog("Enter URLs (seperated by commas): ");
        if (urlString != null) {
            try {
                String[] urls = urlString.split(",");
                List<String> urlList = new ArrayList<String>();
                String title = "";
                for (String url : urls) {
                    urlList.add(url);
                    title += (new URL(url)).getPath() + " ";
                }
                superAdapter.safeLoad(urlList, control, title);
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
            String url = System.getProperty("jnlp.loadMenu");
            if (url == null) {
                url = "http://hicfiles.tc4ga.com/juicebox.properties";
            }
            InputStream is = ParsingUtils.openInputStream(url);
            properties = new Properties();
            if (is != null) {
                properties.load(is);
            }
        } catch (Exception error) {
            log.error("Can't find properties file for loading list", error);
            //    JOptionPane.showMessageDialog(this, "Can't find properties file for loading list", "Error", JOptionPane.ERROR_MESSAGE);
        }
    }
}
