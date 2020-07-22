/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab
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

package juicebox.tools.clt.old;

import juicebox.tools.utils.original.FragmentCalculation;

import java.util.HashMap;
import java.util.Map;

public class StatisticsContainer {
    //Variables for calculating statistics
    public final Map<Integer,Integer> hindIII = new HashMap<>();
    public final Map<Integer,Integer> mapQ = new HashMap<>();
    public final Map<Integer,Integer> mapQInter = new HashMap<>();
    public final Map<Integer,Integer> mapQIntra = new HashMap<>();
    public final Map<Integer,Integer> innerM = new HashMap<>();
    public final Map<Integer,Integer> outerM = new HashMap<>();
    public final Map<Integer,Integer> rightM = new HashMap<>();
    public final Map<Integer,Integer> leftM = new HashMap<>();

    public final Map<Integer,Integer> hindIII2 = new HashMap<>();
    public final Map<Integer,Integer> mapQ2 = new HashMap<>();
    public final Map<Integer,Integer> mapQInter2 = new HashMap<>();
    public final Map<Integer,Integer> mapQIntra2 = new HashMap<>();
    public final Map<Integer,Integer> innerM2 = new HashMap<>();
    public final Map<Integer,Integer> outerM2 = new HashMap<>();
    public final Map<Integer,Integer> rightM2 = new HashMap<>();
    public final Map<Integer,Integer> leftM2 = new HashMap<>();

    public int threePrimeEnd = 0;
    public int fivePrimeEnd = 0;
    public int dangling = 0;
    public int ligation = 0;
    public int inner = 0;
    public int outer = 0;
    public int left = 0;
    public int right = 0;
    public int intra = 0;
    public int inter = 0;
    public int small = 0;
    public int large = 0;
    public int verySmall = 0;
    public int verySmallDangling = 0;
    public int smallDangling = 0;
    public int largeDangling = 0;
    public int interDangling = 0;
    public int trueDanglingIntraSmall = 0;
    public int trueDanglingIntraLarge = 0;
    public int trueDanglingInter = 0;
    public int totalCurrent = 0;
    public int underMapQ = 0;
    public int intraFragment = 0;
    public int unique = 0;

    public int threePrimeEnd2 = 0;
    public int fivePrimeEnd2 = 0;
    public int dangling2 = 0;
    public int ligation2 = 0;
    public int inner2 = 0;
    public int outer2 = 0;
    public int left2 = 0;
    public int right2 = 0;
    public int intra2 = 0;
    public int inter2 = 0;
    public int small2 = 0;
    public int large2 = 0;
    public int verySmall2 = 0;
    public int verySmallDangling2 = 0;
    public int smallDangling2 = 0;
    public int largeDangling2 = 0;
    public int interDangling2 = 0;
    public int trueDanglingIntraSmall2 = 0;
    public int trueDanglingIntraLarge2 = 0;
    public int trueDanglingInter2 = 0;
    public int totalCurrent2 = 0;
    public int underMapQ2 = 0;

    public void add(StatisticsContainer individualContainer){
        for (int i = 1; i <= 2000; i++) {
            hindIII.put(i,hindIII.getOrDefault(i,0)+individualContainer.hindIII.getOrDefault(i, 0));
        }
        for (int i = 1; i <= 200; i++) {
            mapQ.put(i,mapQ.getOrDefault(i,0)+individualContainer.mapQ.getOrDefault(i, 0));
            mapQInter.put(i,mapQInter.getOrDefault(i,0)+individualContainer.mapQInter.getOrDefault(i, 0));
            mapQIntra.put(i,mapQIntra.getOrDefault(i,0)+individualContainer.mapQIntra.getOrDefault(i, 0));
        }
        for (int i = 1; i <= 100; i++) {
            innerM.put(i,innerM.getOrDefault(i,0)+individualContainer.innerM.getOrDefault(i, 0));
            outerM.put(i,outerM.getOrDefault(i,0)+individualContainer.outerM.getOrDefault(i, 0));
            rightM.put(i,rightM.getOrDefault(i,0)+individualContainer.rightM.getOrDefault(i, 0));
            leftM.put(i,leftM.getOrDefault(i,0)+individualContainer.leftM.getOrDefault(i, 0));

        }

        for (int i = 1; i <= 2000; i++) {
            hindIII2.put(i,hindIII2.getOrDefault(i,0)+individualContainer.hindIII2.getOrDefault(i, 0));
        }
        for (int i = 1; i <= 200; i++) {
            mapQ2.put(i,mapQ2.getOrDefault(i,0)+individualContainer.mapQ2.getOrDefault(i, 0));
            mapQInter2.put(i,mapQInter2.getOrDefault(i,0)+individualContainer.mapQInter2.getOrDefault(i, 0));
            mapQIntra2.put(i,mapQIntra2.getOrDefault(i,0)+individualContainer.mapQIntra2.getOrDefault(i, 0));
        }
        for (int i = 1; i <= 100; i++) {
            innerM2.put(i,innerM2.getOrDefault(i,0)+individualContainer.innerM2.getOrDefault(i, 0));
            outerM2.put(i,outerM2.getOrDefault(i,0)+individualContainer.outerM2.getOrDefault(i, 0));
            rightM2.put(i,rightM2.getOrDefault(i,0)+individualContainer.rightM2.getOrDefault(i, 0));
            leftM2.put(i,leftM2.getOrDefault(i,0)+individualContainer.leftM2.getOrDefault(i, 0));
        }
        
         intraFragment += individualContainer.intraFragment;
         unique += individualContainer.unique;

         threePrimeEnd += individualContainer.threePrimeEnd;
         fivePrimeEnd += individualContainer.fivePrimeEnd;
         dangling += individualContainer.dangling;
         ligation += individualContainer.ligation;
         inner += individualContainer.inner;
         outer += individualContainer.outer;
         left += individualContainer.left;
         right += individualContainer.right;
         intra += individualContainer.intra;
         inter += individualContainer.inter;
         small += individualContainer.small;
         large += individualContainer.large;
         verySmall += individualContainer.verySmall;
         verySmallDangling += individualContainer.verySmallDangling;
         smallDangling += individualContainer.smallDangling;
         largeDangling += individualContainer.largeDangling;
         interDangling += individualContainer.interDangling;
         trueDanglingIntraSmall += individualContainer.trueDanglingIntraSmall;
         trueDanglingIntraLarge += individualContainer.trueDanglingIntraLarge;
         trueDanglingInter += individualContainer.trueDanglingInter;
         totalCurrent += individualContainer.totalCurrent;
         underMapQ += individualContainer.underMapQ;

         threePrimeEnd2 += individualContainer.threePrimeEnd2;
         fivePrimeEnd2 += individualContainer.fivePrimeEnd2;
         dangling2 += individualContainer.dangling2;
         ligation2 += individualContainer.ligation2;
         inner2 += individualContainer.inner2;
         outer2 += individualContainer.outer2;
         left2 += individualContainer.left2;
         right2 += individualContainer.right2;
         intra2 += individualContainer.intra2;
         inter2 += individualContainer.inter2;
         small2 += individualContainer.small2;
         large2 += individualContainer.large2;
         verySmall2 += individualContainer.verySmall2;
         verySmallDangling2 += individualContainer.verySmallDangling2;
         smallDangling2 += individualContainer.smallDangling2;
         largeDangling2 += individualContainer.largeDangling2;
         interDangling2 += individualContainer.interDangling2;
         trueDanglingIntraSmall2 += individualContainer.trueDanglingIntraSmall2;
         trueDanglingIntraLarge2 += individualContainer.trueDanglingIntraLarge2;
         trueDanglingInter2 += individualContainer.trueDanglingInter2;
         totalCurrent2 += individualContainer.totalCurrent2;
         underMapQ2 += individualContainer.underMapQ2;

    }
}
