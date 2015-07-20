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

import juicebox.tools.utils.common.MatrixTools;
import org.apache.commons.math.linear.RealMatrix;
import java.util.HashSet;
import java.util.Set;

/**
 * Created by muhammadsaadshamim on 7/20/15.
 */
public class ArrowheadScoreList {

    private Set<ArrowheadScore> arrowheadScores = new HashSet<ArrowheadScore>();

    public void updateActiveIndexScores(RealMatrix blockScore) {

        for (ArrowheadScore score : arrowheadScores) {
            if (score.isActive) {
                score.setScore(MatrixTools.calculateMax(MatrixTools.getSubMatrix(blockScore, score.indices)));
            }
        }
    }

    public void setActiveListElements(int limStart, int limEnd) {
        for(ArrowheadScore score : arrowheadScores) {
            score.isActive = false;
        }

        for(ArrowheadScore score : arrowheadScores) {
            if (score.isWithin(limStart, limEnd)) {
                score.isActive = true;
            }
        }
    }


    private class ArrowheadScore{
        private int[] indices = new int[4];
        private double score = Double.NaN;
        private boolean isActive = false;

        public void setScore(double score) {
            if(Double.isNaN(this.score))
                this.score = score;
            else if(!Double.isNaN(score))
                this.score = Math.max(score, this.score);
        }

        public boolean isWithin(int limStart, int limEnd) {

            boolean containedInBounds = true;
            for(int index : indices){
                containedInBounds &= index >= limStart && index <= limEnd;
            }

            return containedInBounds;
        }
    }
}


