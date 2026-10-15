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

/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package juicebox;

import org.broad.igv.util.StringUtils;

import java.util.ArrayList;
import java.util.List;

class CommandExecutor {

    private final HiC hic;

    public CommandExecutor(HiC hic) {
        this.hic = hic;
    }

    private List<String> getArgs(String[] tokens) {
        List<String> args = new ArrayList<>(tokens.length);
        for (String s : tokens) {
            if (s.trim().length() > 0) {
                args.add(s.trim());
            }
        }
        return args;
    }

    public String execute(String command) {

        List<String> commandString = StringUtils.breakQuotedString(command, ' ');
        List<String> args = getArgs(commandString.toArray(new String[commandString.size()]));

        String result = "OK";
        System.err.println("Executing: " + command);
        try {
            if (args.size() > 0) {

                String cmd = args.get(0).toLowerCase();
                if (cmd.equals("setlocation")) {
                    if (args.size() > 7) {
                        String chrXName = args.get(1);
                        String chrYName = args.get(2);
                        String unitName = args.get(3);
                        HiC.Unit unit = HiC.valueOfUnit(unitName);
                        int binSize = Integer.parseInt(args.get(4));
                        double xOrigin = Double.parseDouble(args.get(5));
                        double yOrigin = Double.parseDouble(args.get(6));
                        double scaleFactor = Double.parseDouble(args.get(7));
                        if (unit == null || binSize <= 0 || !Double.isFinite(xOrigin) || !Double.isFinite(yOrigin)
                                || !Double.isFinite(scaleFactor) || scaleFactor <= 0) {
                            throw new IllegalArgumentException("Invalid location parameters");
                        }
                        double allBinBP = 0, bpX = 0, bpY = 0;
                        String locusX = null, locusY = null;
                        if (args.size() > 8) {
                            if (args.size() != 14 || !args.get(8).equals("all-locus") ||
                                    !chrXName.equalsIgnoreCase("All") || !chrYName.equalsIgnoreCase("All"))
                                throw new IllegalArgumentException("Invalid All sync metadata");
                            allBinBP = Double.parseDouble(args.get(9));
                            locusX = args.get(10); bpX = Double.parseDouble(args.get(11));
                            locusY = args.get(12); bpY = Double.parseDouble(args.get(13));
                            if (!Double.isFinite(allBinBP) || allBinBP <= 0 || !Double.isFinite(bpX) || bpX < 0 ||
                                    !Double.isFinite(bpY) || bpY < 0) throw new IllegalArgumentException("Invalid All sync coordinates");
                        }
                        hic.setSynchronizedLocation(chrXName, chrYName, unit, binSize, xOrigin, yOrigin, scaleFactor,
                                allBinBP, locusX, bpX, locusY, bpY);
                    } else {
                        result = "Not enough parameters";
                    }
                } else {
                    result = "Unknown command string";
                }
            } else {
                result = "Unknown command string";
            }

        } catch (Exception e) {
            System.err.println(e.getLocalizedMessage());
            result = "Error: " + e.getMessage();
        }
        return result;
    }

}
