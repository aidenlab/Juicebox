/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2019 Broad Institute, Aiden Lab
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

package juicebox.tools.utils.juicer.grind;

import juicebox.data.ChromosomeHandler;
import juicebox.data.Dataset;
import juicebox.track.feature.Feature2DList;
import juicebox.windowui.NormalizationType;

import java.io.File;
import java.util.HashSet;
import java.util.Set;

public class ParameterConfigurationContainer {
    public int x, y, z;
    public int grindIterationTypeOption = 0;
    public boolean useObservedOverExpected = false;
    public Dataset ds;
    public boolean useDenseLabelsNotBinary = false;
    public boolean onlyMakePositiveExamples = false;
    public boolean featureDirectionOrientationIsImportant = false;
    public boolean wholeGenome = false;
    public File outputDirectory;
    public Set<Integer> resolutions = new HashSet<>();
    public String featureListPath;
    public int offsetOfCornerFromDiagonal = 0;
    public int stride = 1;
    public String imgFileType = "";
    public boolean useAmorphicPixelLabeling = false;
    public boolean useDiagonal = false;
    public boolean useTxtInsteadOfNPY = false;
    public NormalizationType norm;
    public ChromosomeHandler chromosomeHandler;
    public Feature2DList feature2DList;

    public ParameterConfigurationContainer() {
    }
}
