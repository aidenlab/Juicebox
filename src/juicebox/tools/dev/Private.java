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

package juicebox.tools.dev;

import juicebox.gui.SuperAdapter;

/**
 * Created by muhammadsaadshamim on 8/15/16.
 */
public class Private {

    public static boolean assessGenomeForRE(String genomeId) {
        return genomeId.equalsIgnoreCase("anasPlat1");
    }

    public static boolean assessGenomeForRE2(String genomeId) {
        return genomeId.equalsIgnoreCase("hg19_contig");
    }

    public static String reForDMEL(int sites) {
        if (sites == 185217) return "MseI";
        return null;
    }

    public static boolean assessGenomeForRE3(String genomeId) {
        return genomeId.equalsIgnoreCase("galGal4");
    }

    public static String reForHG18(int sites) {
        if (sites == 64338) return "HindIII";
        return null;
    }

    public static String reForHG19(int sites) {
        if (sites == 22706) return "Acc65I";
        if (sites == 4217) return "AgeI";
        if (sites == 158473) return "BseYI";
        if (sites == 74263) return "BspHI";
        if (sites == 60834) return "BstUI2";
        if (sites == 2284472) return "CpG";
        if (sites == 139125) return "HinP1I";
        if (sites == 160930) return "HpyCH4IV2";
        if (sites == 1632) return "MluI";
        if (sites == 1428208) return "MseI";
        if (sites == 194423) return "MspI";
        if (sites == 22347) return "NheI";
        if (sites == 1072254) return "NlaIII";
        if (sites == 1128) return "NruI";
        if (sites == 2344) return "SaII";
        if (sites == 1006921) return "StyD4I";
        if (sites == 256163) return "StyI";
        if (sites == 119506) return "TaqI2";
        if (sites == 9958) return "XhoI";
        if (sites == 31942) return "XmaI";
        return null;
    }

    public static String reForMM9(int sites) {
        if (sites == 1157974) return "MseI";
        if (sites == 933321) return "NlaIII";
        return null;
    }

    public static boolean assessGenomeForRE4(String genomeId) {
        return genomeId.equalsIgnoreCase("susScr3");
    }

    public static void launchMapSubsetGUI(SuperAdapter superAdapter) {
        MapSelectionPanel.launchMapSubsetGUI(superAdapter);
    }
}
