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

package juicebox.state;

/**
 * Created by mshamim on 7/31/2015.
 */
class State {
    private static final String MAPPATH = "MapPath";
    private static final String MAP = "Map";
    private static final String MAPURL = "MapURL";
    private static final String CONTROLURL = "ControlURL";
    private static final String XCHR = "XChromosome";
    private static final String YCHR = "YChromosome";
    private static final String UNIT = "UnitName";
    private static final String BINSIZE = "BinSize";
    private static final String XORIGIN = "xOrigin";
    private static final String YORIGIN = "yOrigin";
    private static final String SCALE = "ScaleFactor";
    private static final String DISPLAY = "DisplayOption";
    private static final String NORM = "NormalizationType";
    private static final String MINCOLOR = "MinColorVal";
    private static final String LOWCOLOR = "LowerColorVal";
    private static final String UPPERCOLOR = "UpperColorVal";
    private static final String MAXCOLOR = "MaxColorVal";
    private static final String COLORSCALE = "colrScaleVal";
    private static final String TRACKURLS = "LoadedTrackURLS";
    private static final String TRACKNAMES = "LoadedTrackNames";
    private static final String CONFIGTRACKINFO = "ConfigTrackInfo";

    /* TODO implement switch case on enum
    enum StateVar {}*/
    public static final String[] stateVarNames = new String[]{MAPPATH, MAP, MAPURL, CONTROLURL, XCHR, YCHR, UNIT, BINSIZE,
            XORIGIN, YORIGIN, SCALE, DISPLAY, NORM, MINCOLOR, LOWCOLOR, UPPERCOLOR, MAXCOLOR, COLORSCALE, TRACKURLS, TRACKNAMES, CONFIGTRACKINFO};
}
