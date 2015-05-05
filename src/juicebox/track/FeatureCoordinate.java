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

package juicebox.track;

/**
 * Created by muhammadsaadshamim on 5/4/15.
 */
public class FeatureCoordinate implements Comparable<FeatureCoordinate>{

    private String chromosome;
    private int startPosition;
    private int endPosition;

    public FeatureCoordinate(String chromosome, int startPosition, int endPosition){
        this.chromosome = chromosome;
        this.startPosition = startPosition;
        this.endPosition = endPosition;
    }

    public String getChromosome() {
        return chromosome;
    }

    public int getStartPosition() {
        return startPosition;
    }

    public int getEndPosition() {
        return endPosition;
    }

    @Override
    public int compareTo(FeatureCoordinate otherCoordinate) {
        if(chromosome.equals(otherCoordinate.getChromosome())){
            if(startPosition == otherCoordinate.getStartPosition()){
                return endPosition - otherCoordinate.getEndPosition();
            }
            else{
                return startPosition - otherCoordinate.getStartPosition();
            }
        }
        else{
            return chromosome.compareTo(otherCoordinate.getChromosome());
        }
    }
}
