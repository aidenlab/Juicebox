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

package juicebox.tools.utils.norm;

import juicebox.HiCGlobals;
import juicebox.data.ContactRecord;
import juicebox.tools.utils.norm.final2.FinalScale;

import java.util.List;

public class ZeroScale {
    public static double[] scale(List<List<ContactRecord>> contactRecordsListOfLists, double[] targetVectorInitial, String key) {
        double[] newVector = FinalScale.scaleToTargetVector(contactRecordsListOfLists, targetVectorInitial);

        if (newVector == null && HiCGlobals.printVerboseComments) {
            System.err.println("Scaling result still null for " + key + "; vector did not converge");
        }
        return newVector;
    }


    public static double[] normalizeVectorByScaleFactor(double[] newNormVector, List<List<ContactRecord>> contactRecordsListOfLists) {

        for (int k = 0; k < newNormVector.length; k++) {
            if (newNormVector[k] <= 0 || Double.isNaN(newNormVector[k])) {
                newNormVector[k] = Double.NaN;
            } else {
                newNormVector[k] = 1 / newNormVector[k];
            }
        }

        double normalizedSumTotal = 0, sumTotal = 0;

        for (List<ContactRecord> records : contactRecordsListOfLists) {
            for (ContactRecord cr : records) {
                int x = cr.getBinX();
                int y = cr.getBinY();
                final float counts = cr.getCounts();

                if (!Double.isNaN(newNormVector[x]) && !Double.isNaN(newNormVector[y])) {
                    double normalizedValue = counts / (newNormVector[x] * newNormVector[y]);
                    normalizedSumTotal += normalizedValue;
                    sumTotal += counts;
                    if (x != y) {
                        normalizedSumTotal += normalizedValue;
                        sumTotal += counts;
                    }
                }
            }
        }

        double scaleFactor = Math.sqrt(normalizedSumTotal / sumTotal);

        for (int k = 0; k < newNormVector.length; k++) {
            if (!Double.isNaN(newNormVector[k])) {
                newNormVector[k] = scaleFactor * newNormVector[k];
            }
        }
        return newNormVector;
    }

    public static double[] mmbaScaleToVector(List<List<ContactRecord>> contactRecords, double[] tempTargetVector) {

        double[] newNormVector = scale(contactRecords, tempTargetVector, "mmsa_scale");
        if (newNormVector != null) {
            newNormVector = normalizeVectorByScaleFactor(newNormVector, contactRecords);
        }

        return newNormVector;
    }

}
