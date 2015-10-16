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

package juicebox.tools.utils.juicer.arrowhead;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by muhammadsaadshamim on 7/22/15.
 */
class CumulativeBlockResults {

    private final ArrowheadScoreList cumulativeInternalList;
    private final ArrowheadScoreList cumulativeInternalControl;
    private List<HighScore> cumulativeResults = new ArrayList<HighScore>();

    public CumulativeBlockResults(int resolution) {
        cumulativeInternalList = new ArrowheadScoreList(resolution);
        cumulativeInternalControl = new ArrowheadScoreList(resolution);
    }


    public void add(BlockResults blockResults) {
        cumulativeResults.addAll(blockResults.getResults());
        cumulativeInternalControl.addAll(blockResults.getInternalControl());
        cumulativeInternalList.addAll(blockResults.getInternalList());
    }

    public ArrowheadScoreList getCumulativeInternalControl() {
        return cumulativeInternalControl;
    }

    public List<HighScore> getCumulativeResults() {
        return cumulativeResults;
    }

    public void setCumulativeResults(List<HighScore> cumulativeResults) {
        this.cumulativeResults = new ArrayList<HighScore>(cumulativeResults);
    }

    public ArrowheadScoreList getCumulativeInternalList() {
        return cumulativeInternalList;
    }

    public void mergeScores() {
        cumulativeInternalControl.mergeScores();
        cumulativeInternalList.mergeScores();
    }

    public void scaleIndicesByResolution(int resolution) {
        for (HighScore hs : cumulativeResults) {
            hs.scaleIndicesByResolution(resolution);
        }
    }
}
