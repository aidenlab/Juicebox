/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2017 Broad Institute, Aiden Lab
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

package juicebox.assembly;

import juicebox.track.feature.Feature2D;

/**
 * Created by nathanielmusial on 6/30/17.
 */
public class ContigProperty {

    private String name;
    private int indexId;
    private int length;
    private Feature2D feature2D;

    public ContigProperty(String name, int indexId, int length) {
        this.name = name;
        this.indexId = indexId;
        this.length = length;
        this.feature2D = null;
    }

    public ContigProperty(ContigProperty contigProperty) {
        this.name = contigProperty.name;
        this.indexId = contigProperty.indexId;
        this.length = contigProperty.length;
        if (this.feature2D != null)
            this.feature2D = contigProperty.feature2D.deepCopy();
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getOriginalContigName() {
        if (name.contains(":::")) {
            return name.split(":::")[0];
        } else {
            return name;
        }
    }

    public int getFragmentNumber() {
        if (name.contains(":::")) {
            if (name.contains(":::debris")) {
                String temp = name.split("_")[1];
                return Integer.parseInt(temp.split(":::")[0]);
            } else {
                return Integer.parseInt(name.split("_")[1]); //can just parse int from string
            }
        } else {
            System.err.println("can't find fragment num");
            return -1;

        }
    }

    public int getIndexId() {
        return indexId;
    }

    public void setIndexId(int indexId) {
        this.indexId = indexId;
    }

    public int getLength() {
        return length;
    }

    public Feature2D getFeature2D() {
        return feature2D;
    }

    public void setFeature2D(Feature2D feature2D) {
        this.feature2D = feature2D;
    }

    @Override
    public String toString() {
        return name + " " + indexId + " " + length;
    }

}