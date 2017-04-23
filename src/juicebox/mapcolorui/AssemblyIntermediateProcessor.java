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

package juicebox.mapcolorui;

import juicebox.gui.SuperAdapter;
import juicebox.track.feature.Contig2D;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.util.Pair;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by muhammadsaadshamim on 4/17/17.
 */
public class AssemblyIntermediateProcessor {
    public static Pair<Integer, Integer> process(int binX, int binY) {
        /*if (binX >= 20 && binX < 40) {
            binX += 1700;
        } else if (binX >= 1720 && binX < 1740) {
            binX -= 1700;
        }

        if (binY >= 20 && binY < 40) {
            binY += 1700;
        } else if (binY >= 1720 && binY < 1740) {
            binY -= 1700;
        }
        */

        if (binX >= 120 && binX <= 1240) {
            binX = 120 + (1240 - binX);
        }

        if (binY >= 120 && binY <= 1240) {
            binY = 120 + (1240 - binY);
        }

        if (binX > binY) {
            return new Pair<>(binY, binX);
        }
        return new Pair<>(binX, binY);
    }

    public static void makeChanges(String[] encodedInstructions, SuperAdapter superAdapter) {
        List<Feature2DList> allFeatureLists = superAdapter.getAllLayers().get(0).getAnnotationLayer().getFeatureHandler()
                .getAllVisibleLoopLists();
        Feature2DList features = allFeatureLists.get(0);
        makeAssemblyChanges(features, superAdapter.getHiC().getXContext().getChromosome(), encodedInstructions);
        superAdapter.getAllLayers().get(0).getAnnotationLayer().getFeatureHandler().remakeRTree();
    }


    private static void makeAssemblyChanges(Feature2DList features, Chromosome chromosome, String[] encodedInstructions) {
        final String key = Feature2DList.getKey(chromosome, chromosome);

        List<Feature2D> contigs = new ArrayList<>();
        for (Feature2D entry : features.get(key)) {
            contigs.add(entry.toContig());
        }
        Collections.sort(contigs);

        for (String instruction : encodedInstructions) {
            if (instruction.startsWith("-")) {
                // TODO future
                // invert selections rather than just one contig
                // this involves inverting each of the sub contigs,
                // but also inverting their order

                invertEntryAt(contigs, Math.abs(Integer.parseInt(instruction)));
            } else {
                String[] indices = instruction.split("->");
                int currentIndex = Integer.parseInt(indices[0]);
                int newIndex = Integer.parseInt(indices[1]);
                moveFeatureToNewIndex(contigs, currentIndex, newIndex);
            }
        }

        recalculateAllAlterations(contigs);

        features.setWithKey(key, contigs);
    }

    private static void recalculateAllAlterations(List<Feature2D> contigs) {
        int i = 0;
        for (Feature2D feature2D : contigs) {
            Contig2D contig2D = feature2D.toContig();
            i = contig2D.setNewStart(i);
        }
    }

    private static void moveFeatureToNewIndex(List<Feature2D> contigs, int currentIndex, int newIndex) {
        // http://stackoverflow.com/questions/4938626/moving-items-around-in-an-arraylist
        Feature2D item = contigs.remove(currentIndex);
        contigs.add(newIndex, item);
    }

    private static void invertEntryAt(List<Feature2D> contigs, int index) {
        ((Contig2D) contigs.get(index)).toggleInversion();
    }
}
