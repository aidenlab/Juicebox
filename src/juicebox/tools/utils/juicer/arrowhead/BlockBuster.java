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

package juicebox.tools.utils.Juicer.Arrowhead;

import juicebox.data.MatrixZoomData;
import juicebox.tools.utils.Common.HiCFileTools;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.RealMatrix;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by muhammadsaadshamim on 6/3/15.
 */
public class BlockBuster {


    /**
     * should be called separately for each chromosome
     *
     * @return
     */
    public static void blockbuster(MatrixZoomData zd, int chrLength, ArrowheadScoreList list, ArrowheadScoreList control){

        // int chrLength = chromosome.getLength();

        float signThreshold = 0.4f;
        float varThreshold = 1000f;

        List<HighScore> results = callSubBlockbuster(zd, chrLength, varThreshold, signThreshold, list, control);

        while(results.size() == 0 && signThreshold > 0){
            signThreshold = signThreshold - 0.1f;
            results = callSubBlockbuster(zd, chrLength, varThreshold, signThreshold, list, control);
        }

        // high variance threshold, fewer blocks, high confidence
        List<HighScore> highConfidenceResults = callSubBlockbuster(zd, chrLength, 0.2f, 0.5f, null, null);


        // TODO
        //diffBetweenResults(results, highConfidenceResults);

    }

    private static List<HighScore> callSubBlockbuster(MatrixZoomData zd, int chrLength, float varThreshold, float signThreshold,
                                                   ArrowheadScoreList list, ArrowheadScoreList control) {

        List<HighScore> cumulativeResults = new ArrayList<HighScore>();

        for(int limStart = 0; limStart < chrLength; limStart += 1000){
            int limEnd = Math.min(limStart + 2000, chrLength);

            list.setActiveListElements(limStart, limEnd);
            control.setActiveListElements(limStart, limEnd);

            int n = limEnd - limStart + 1;
            RealMatrix observed = HiCFileTools.extractLocalBoundedRegion(zd, limStart, limEnd,
                    limStart, limEnd, n, n, NormalizationType.KR);

            List<HighScore> results = (new BlockResults(observed, varThreshold, signThreshold, list, control)).getResults();
            offsetResultsIndex(results, limStart);

            cumulativeResults.addAll(results);
            System.out.print(".");
        }
        return cumulativeResults;
    }

    private static void offsetResultsIndex(List<HighScore> scores, int offset) {
        for(HighScore score : scores){
            score.offsetIndex(offset);
        }
    }


    // for repeat values - select max score
    // remove repeats
    private void cleanScores(){

    }

}
