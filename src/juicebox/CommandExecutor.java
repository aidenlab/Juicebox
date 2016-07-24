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

/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package juicebox;

import org.apache.log4j.Logger;
import org.broad.igv.util.StringUtils;

import java.util.ArrayList;
import java.util.List;

class CommandExecutor {

    private static final Logger log = Logger.getLogger(CommandExecutor.class);
    private final HiC hic;

    public CommandExecutor(HiC hic) {
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

    public String execute(String command) {

        List<String> commandString = StringUtils.breakQuotedString(command, ' ');
        List<String> args = getArgs(commandString.toArray(new String[commandString.size()]));

        String result = "OK";
        log.debug("Executing: " + command);
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
                        hic.setLocation(chrXName, chrYName, unit, binSize, xOrigin, yOrigin, scaleFactor,
                                HiC.ZoomCallType.DIRECT, "Goto Sync", false);
                    } else {
                        result = "Not enough parameters";
                    }
                }
            } else {
                result = "Unknown command string";
            }

        } catch (Exception e) {
            log.error(e);
            result = "Error: " + e.getMessage();
        }
        return result;
    }

}
