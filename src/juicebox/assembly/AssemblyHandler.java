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

import juicebox.HiC;
import juicebox.gui.SuperAdapter;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;

import java.awt.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by nathanielmusial on 6/30/17.
 */
public class AssemblyHandler {

    private List<ContigProperty> contigProperties;
    private List<List<Integer>> scaffoldProperties;
    private Feature2DList contigs;
    private Feature2DList scaffolds;
    private String chromosomeName = "assembly";


    public AssemblyHandler(List<ContigProperty> contigProperties, List<List<Integer>> scaffoldProperties) {
        this.contigProperties = contigProperties;
        this.scaffoldProperties = scaffoldProperties;
        contigs = new Feature2DList();
        scaffolds = new Feature2DList();
        generateContigsAndScaffolds();
    }

    public Feature2DList getContigs() {
        return contigs;
    }

    public Feature2DList getScaffolds() {
        return scaffolds;
    }

    public List<ContigProperty> getContigProperties() {
        return contigProperties;
    }

    public List<List<Integer>> getScaffoldProperties() {
        return scaffoldProperties;
    }

    private void generateContigsAndScaffolds() {
        int contigStartPos = 0;
        int scaffoldStartPos = 0;
        int scaffoldLength = 0;
        for (List<Integer> row : scaffoldProperties) {
            for (Integer contigIndex : row) {
                ContigProperty contigProperty = contigProperties.get(Math.abs(contigIndex) - 1);
                String contigName = contigProperty.getName();
                Integer contigLength = contigProperty.getLength();

                Feature2D contig = new Feature2D(Feature2D.FeatureType.CONTIG, chromosomeName, contigStartPos, (contigStartPos + contigLength),
                        chromosomeName, contigStartPos, (contigStartPos + contigLength),
                        new Color(0, 255, 0), new HashMap<String, String>());
                Map<String, String> attributes = new HashMap<String, String>();
                attributes.put("Scaffold Name", contigName);
                attributes.put("Scaffold Id", contigIndex.toString());
                contigs.add(1, 1, contig);
                contigProperty.setFeature2D(contig);

                contigStartPos += contigLength;
                scaffoldLength += contigLength;
            }
            Feature2D scaffold = new Feature2D(Feature2D.FeatureType.SCAFFOLD, chromosomeName, scaffoldStartPos, (scaffoldStartPos + scaffoldLength),
                    chromosomeName, scaffoldStartPos, (scaffoldStartPos + scaffoldLength),
                    new Color(0, 0, 255), new HashMap<String, String>());
            scaffolds.add(1, 1, scaffold);

            scaffoldStartPos += scaffoldLength;
            scaffoldLength = 0;
        }
    }

    public void splitGroup(List<Feature2D> contigs, int startIndex, int endIndex) {

    }

    public void splitGroup(List<Integer> contigIds, int scaffoldId) {
        List<Integer> tempGroup = scaffoldProperties.get(scaffoldId);
        int startIndex = tempGroup.indexOf(contigIds.get(0));
        int endIndex = tempGroup.indexOf(contigIds.get(contigIds.size() - 1));
        if (startIndex != 0) {
            List<List<Integer>> newGroups = new ArrayList<>();
            newGroups.add(tempGroup.subList(0, startIndex));
            newGroups.add(tempGroup.subList(startIndex, endIndex));
            newGroups.add(tempGroup.subList(endIndex, tempGroup.size() - 1));
            scaffoldProperties.addAll(scaffoldId, newGroups);
            scaffoldProperties.remove(tempGroup);
        } else {
            List<List<Integer>> newGroups = new ArrayList<>();
            newGroups.add(tempGroup.subList(0, endIndex));
            newGroups.add(tempGroup.subList(endIndex, tempGroup.size() - 1));
            scaffoldProperties.addAll(scaffoldId, newGroups);
            scaffoldProperties.remove(tempGroup);
        }

    }

    public void reGroup(List<Feature2D> contigs, int startIndex, int endIndex) {
        //logic for finding two groups
    }

    public void reGroup(int scaffoldId) {
        List<Integer> firstGroup = scaffoldProperties.get(scaffoldId);
        List<Integer> secondGroup = scaffoldProperties.get(scaffoldId + 1);
        firstGroup.addAll(secondGroup);
        scaffoldProperties.remove(secondGroup);
    }

    public void translateSelection(List<Feature2D> contigs, int startIndex, int endIndex) {

    }

    public void invertSelection(List<Feature2D> contigs, int startIndex, int endIndex) {

    }

    public void splitConitg(Feature2D originalContig, Feature2D debrisContig, SuperAdapter superAdapter, HiC hic) {

    }

    public ContigProperty feature2DtoContig(Feature2D feature2D) {
        for (ContigProperty contigProperty : contigProperties) {
            if (contigProperty.getFeature2D() == feature2D) //make sure it is okay
                return contigProperty;
        }
        return null;
    }
}