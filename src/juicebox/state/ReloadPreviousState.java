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


package juicebox.state;


import juicebox.CommandExecutor;
import juicebox.HiC;
import juicebox.MainWindow;
import juicebox.windowui.MatrixType;
import juicebox.windowui.NormalizationType;
import org.apache.log4j.Logger;

import javax.swing.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


/**
 * Created by Zulkifl on 6/10/2015.
 */

public class ReloadPreviousState {

    private static final Logger log = Logger.getLogger(CommandExecutor.class);
    private final HiC hic;



    public ReloadPreviousState(HiC hic) {
        this.hic = hic;
    }

    private List<String> getArgs(String[] tokens) {
        List<String> args = new ArrayList<String>(tokens.length);
        for (String s : tokens) {
            if (s.trim().length() > 0) {
                args.add(s.trim());
            }
        }
        return args;
    }


    public String reloadXML(String[] infoForReload) {
        String result = "OK";
        //TODO---USE XML File instead
        String[] initialInfo = new String[5]; //hicURL,xChr,yChr,unitSize
         double[] doubleInfo = new double[7]; //xOrigin, yOrigin, ScaleFactor, minColorVal, lowerColorVal, upperColorVal, maxColorVal
         String[] trackURLsAndNames = new String[2];
        log.debug("Executing: " + infoForReload);
        if (infoForReload.length > 0) {
            int fileSize = infoForReload.length;
            if (infoForReload.length > 14) {
                try {
                    initialInfo[0] = infoForReload[1]; //HiC Map Name
                    initialInfo[1] = infoForReload[2]; //hicURL
                    initialInfo[2] = infoForReload[3]; //xChr
                    initialInfo[3] = infoForReload[4]; //yChr
                    initialInfo[4] = infoForReload[5]; //unitSize
                    int binSize = Integer.parseInt(infoForReload[6]); //binSize
                    doubleInfo[0] = Double.parseDouble(infoForReload[7]); //xOrigin
                    doubleInfo[1] = Double.parseDouble(infoForReload[8]); //yOrigin
                    doubleInfo[2] = Double.parseDouble(infoForReload[9]); //ScaleFactor
                    MatrixType displayOption = MatrixType.valueOf(infoForReload[10].toUpperCase());
                    NormalizationType normType = NormalizationType.valueOf(infoForReload[11].toUpperCase());
                    doubleInfo[3] = Double.parseDouble(infoForReload[12]); //minColorVal
                    doubleInfo[4] = Double.parseDouble(infoForReload[13]); //lowerColorVal
                    doubleInfo[5] = Double.parseDouble(infoForReload[14]); //upperColorVal
                    doubleInfo[6] = Double.parseDouble(infoForReload[15]); //maxColorVal
                    trackURLsAndNames[0] = (infoForReload[16]); //trackURLs
                    trackURLsAndNames[1] = (infoForReload[17]); //trackNames

                    hic.safeSetReloadStateFromXML(initialInfo, binSize, doubleInfo, displayOption, normType, trackURLsAndNames);
                } catch(NumberFormatException nfe){
                        JOptionPane.showMessageDialog(MainWindow.getInstance(), "Error:\n" + nfe.getMessage(), "Error",
                                JOptionPane.ERROR_MESSAGE);
                }
                } else {
                    result = "Not enough parameters";
                }
                } else {
                    result = "Unknown command string";
                }

        return result;

    }
}




