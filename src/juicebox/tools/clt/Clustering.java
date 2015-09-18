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

package juicebox.tools.clt;

/**
 * Created by muhammadsaadshamim on 9/14/15.
 */
public class Clustering {

    /**
     * @param data to cluster - each
     * @param n
     */
    public void cluster(double[][] data, int n) {

/*
        OpdfMultiGaussianFactory factory = new OpdfMultiGaussianFactory(6);
        new ObservationVector(data[0]);
        Hmm<ObservationVector> hmm = new Hmm<ObservationVector>(6, factory);

        List<List<Observation>> sequences = new ArrayList<List<O>>();

        for (int i = 0; i < 200; i++)
            sequences.add(new ObservationVector())
            sequences.add(mg.observationSequence(100));


        BaumWelchLearner bwl = new BaumWelchLearner();
        Hmm<?> learntHmm = bwl.learn(hmm, sequences);

        for (int i = 0; i < 10; i++) {
            learntHmm = bwl.iterate(learntHmm);
        }


        List<List<ObservationVector>> sequences2 = new ArrayList<List<ObservationVector>>();

        KMeansLearner<ObservationVector> kml =
                new KMeansLearner <ObservationVector>(3 , new OpdfMultiGaussianFactory(6) , sequences2);
        Hmm <ObservationVector> initHmm = kml.iterate() ;
    */
    }
}
