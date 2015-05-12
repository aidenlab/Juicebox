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

package juicebox.tools.utils.Juicer;

import juicebox.tools.utils.Common.HiCFileTools;
import juicebox.track.Feature2D;
import org.broad.igv.feature.Chromosome;

import java.util.*;

/**
 * Created by muhammadsaadshamim on 5/5/15.
 */
public class LoopContainer {

    private Map<Chromosome, Set<Feature2D>> filteredChrToLoopsMap;
    private Map<Chromosome, Integer[]> numFilteredUniqueTotalLoops;

    public LoopContainer(Map<Chromosome, Set<Feature2D>> filteredChrToLoopsMap,
                         Map<Chromosome, Integer[]> numFilteredUniqueTotalLoops) {
        this.filteredChrToLoopsMap = filteredChrToLoopsMap;
        this.numFilteredUniqueTotalLoops = numFilteredUniqueTotalLoops;
    }

    public Set<Chromosome> getCommonChromosomes(List<Chromosome> chromosomes) {
        return HiCFileTools.getSetIntersection(
                new HashSet<Chromosome>(filteredChrToLoopsMap.keySet()),
                new HashSet<Chromosome>(chromosomes));
    }

    public Set<Feature2D> getUniqueFilteredLoopList(Chromosome chr) {
        return new HashSet<Feature2D>(filteredChrToLoopsMap.get(chr));
    }

    /**
     * [NumUniqueFiltered, NumUnique, NumTotal]
     * @param chr
     * @return
     */
    public Integer[] getUniqueFilteredLoopNumbers(Chromosome chr) {
        return numFilteredUniqueTotalLoops.get(chr);
    }


}
