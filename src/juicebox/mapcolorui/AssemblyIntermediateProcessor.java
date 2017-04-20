/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2017 Broad Institute, Aiden Lab
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

package juicebox.mapcolorui;

import org.broad.igv.util.Pair;

/**
 * Created by muhammadsaadshamim on 4/17/17.
 */
public class AssemblyIntermediateProcessor {
    public static Pair<Integer, Integer> process(int binX, int binY) {
        if (binX >= 20 && binX < 40) {
            binX += 1400;
        } else if (binX >= 1420 && binX < 1440) {
            binX -= 1400;
        }

        if (binY >= 20 && binY < 40) {
            binY += 1400;
        } else if (binY >= 1420 && binY < 1440) {
            binY -= 1400;
        }

        if (binX > binY) {
            return new Pair<>(binY, binX);
        }
        return new Pair<>(binX, binY);
    }
}
