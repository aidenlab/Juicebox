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

/**
 * Wrapper for arrowhead blockbuster results
 * Created by muhammadsaadshamim on 6/8/15.
 */
class HighScore {
    private int i;
    private int j;
    private final double score;
    private final double uVarScore;
    private final double lVarScore;
    private final double upSign;
    private final double loSign;

    public HighScore(int i, int j, double score, double uVarScore, double lVarScore,
                     double upSign, double loSign) {
        this.i = i;
        this.j = j;
        this.score = score;
        this.uVarScore = uVarScore;
        this.lVarScore = lVarScore;
        this.upSign = upSign;
        this.loSign = loSign;
    }

    public String toString() {
        return "" + i + "\t" + j + "\t" + score + "\t" + uVarScore + "\t" + lVarScore + "\t" + upSign + "\t" + loSign;
    }

    public void offsetIndex(int offset) {
        this.i += offset;
        this.j += offset;
    }

    public int getWidth() {
        return Math.abs(j-i);
    }

    public int getI() {
        return i;
    }

    public int getJ() {
        return j;
    }

    private double getLoSign() {
        return loSign;
    }

    private double getScore() {
        return score;
    }

    private double getuVarScore() {
        return uVarScore;
    }

    private double getlVarScore() {
        return lVarScore;
    }

    private double getUpSign() {
        return upSign;
    }

    @Override
    public boolean equals(Object object){
        if (this == object)
            return true;
        if (object == null)
            return false;
        if (getClass() != object.getClass())
            return false;
        final HighScore o = (HighScore) object;
        return i == o.getI()
                && j == o.getJ()
                && score == o.getScore()
                && uVarScore == o.getuVarScore()
                && lVarScore == o.getlVarScore()
                && upSign == o.getUpSign()
                && loSign == o.loSign;
    }

    @Override
    public int hashCode(){
        return 7*(i+j)*(int)Math.floor(score+uVarScore+lVarScore+upSign+loSign);
    }


}
