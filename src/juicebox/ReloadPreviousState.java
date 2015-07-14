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


package juicebox;


import juicebox.data.Matrix;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.MatrixType;
import juicebox.windowui.NormalizationType;
import org.apache.log4j.Logger;
import org.broad.igv.util.StringUtils;

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


    public String reload(File currentInfo) {
        String result = "OK";
 //TODO---USE XML File instead
        try {
            BufferedReader buffRead = new BufferedReader(new FileReader(currentInfo));
            String command;
            while((command = buffRead.readLine()) != null) {
                if (command.contains(hic.currentMapName())) {
                    String delimiter = "\\$\\$";
                    String[] stateArray;
                    stateArray = command.split(delimiter);
                    List<String> args = getArgs(stateArray);
                    ArrayList<String> tracks = new ArrayList<String>();
                    log.debug("Executing: " + command);
                    if (args.size() > 0) {
                        int fileSize = args.size();
                        String cmd = args.get(0).toLowerCase();
                        if (cmd.contains("currentstate:")) {
                            if (args.size() > 14) {
                                String hicURL = args.get(1);
                                String chrXName = args.get(2);
                                String chrYName = args.get(3);
                                String unitName = args.get(4);
                                int binSize = Integer.parseInt(args.get(5));
                                double xOrigin = Double.parseDouble(args.get(6));
                                double yOrigin = Double.parseDouble(args.get((7)));
                                double scaleFactor = Double.parseDouble(args.get(8));
                                MatrixType displayOption = MatrixType.valueOf(args.get((9)).toUpperCase());
                                NormalizationType normType = NormalizationType.valueOf(args.get((10)).toUpperCase());
                                double minColorVal = Double.parseDouble(args.get((11)));
                                double lowerColorVal = Double.parseDouble(args.get((12)));
                                double upperColorVal = Double.parseDouble(args.get((13)));
                                double maxColorVal = Double.parseDouble(args.get((14)));
                                for (int i = 15; i < fileSize; i++) {
                                    tracks.add(args.get(i));
                                }
                                hic.setReloadState(hicURL, chrXName, chrYName, unitName, binSize, xOrigin, yOrigin, scaleFactor, displayOption, normType
                                        , minColorVal, lowerColorVal, upperColorVal, maxColorVal, tracks);
                            } else {
                                result = "Not enough parameters";
                            }
                        }
                    } else {
                        result = "Unknown command string";
                    }
                }
            }

        } catch (IOException e) {
            e.printStackTrace();
            //result = "Error: " + e.getMessage();
        }

        return result;

    }
}




