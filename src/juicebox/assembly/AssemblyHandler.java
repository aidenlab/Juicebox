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
import java.util.*;
import java.util.List;

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

    public AssemblyHandler(AssemblyHandler assemblyHandler) {
        this.contigProperties = assemblyHandler.cloneContigProperties();
        this.scaffoldProperties = new ArrayList(assemblyHandler.scaffoldProperties);
        this.contigs = new Feature2DList();
        this.scaffolds = new Feature2DList();
        generateContigsAndScaffolds();
    }

    public List<ContigProperty> cloneContigProperties() {
        List<ContigProperty> newList = new ArrayList<>();
        for (ContigProperty contigProperty : contigProperties) {
            newList.add(new ContigProperty(contigProperty));
        }
        return newList;
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

    public void generateContigsAndScaffolds() {
        contigs = new Feature2DList();
        scaffolds = new Feature2DList();
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
                Map<String, String> attributes = new HashMap<>();
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

    public void splitGroup(List<Feature2D> contigs) {
        List<Integer> contigIds = contig2DListToIntegerList(contigs);
        int scaffoldRowNum = getScaffoldRow(contigIds);
        splitGroup(contigIds, scaffoldRowNum);
    }

    private List<List<Integer>> splitGroup(List<Integer> contigIds, int scaffoldRow) {
        List<Integer> tempGroup = scaffoldProperties.get(scaffoldRow);
        int startIndex = tempGroup.indexOf(contigIds.get(0));
        int endIndex = tempGroup.indexOf(contigIds.get(contigIds.size() - 1));

        List<List<Integer>> newGroups = new ArrayList<>();
        if (startIndex != 0) {
            if (endIndex == tempGroup.size() - 1) {
                newGroups.add(tempGroup.subList(0, startIndex));
                newGroups.add(tempGroup.subList(startIndex, endIndex));
            } else {
                newGroups.add(tempGroup.subList(0, startIndex));
                newGroups.add(tempGroup.subList(startIndex, endIndex));
                newGroups.add(tempGroup.subList(endIndex, tempGroup.size() - 1));
            }
        } else {
            newGroups.add(tempGroup.subList(0, endIndex));
            newGroups.add(tempGroup.subList(endIndex, tempGroup.size() - 1));
        }
        scaffoldProperties.addAll(scaffoldRow, newGroups);
        scaffoldProperties.remove(tempGroup);

        return newGroups;
    }

    public void mergeGroup(List<Feature2D> contigs) {
        int scaffoldRowNum = findAdjacentGroups(contig2DListToIntegerList(contigs));
        mergeGroup(scaffoldRowNum);
    }

    private void mergeGroup(int scaffoldRowNum) {
        List<Integer> firstGroup = scaffoldProperties.get(scaffoldRowNum);
        List<Integer> secondGroup = scaffoldProperties.get(scaffoldRowNum + 1);
        firstGroup.addAll(secondGroup);
        scaffoldProperties.remove(secondGroup);
    }

    public void translateSelection(List<Feature2D> contigIds, int translateRow) {
        performTranslation(contig2DListToIntegerList(contigIds), translateRow);

    }

    private void performTranslation(List<Integer> contigIds, int translateRow) {
        int originalRowNum = getScaffoldRow(contigIds);
        List<Integer> originalRow = scaffoldProperties.get(originalRowNum);
        List<List<Integer>> newGroups = splitGroup(contigIds, originalRowNum);
        List<Integer> translateList;
        if (newGroups.size() == 3) {
            translateList = newGroups.get(1);// get middle group in split
        } else if (newGroups.size() == 2) {

            if (originalRow.indexOf(contigIds.get(0)) == 0) {
                translateList = newGroups.get(0);   //get first group in split
            } else {
                translateList = newGroups.get(1); //get second gorup in split
            }
        } else {
            System.err.println("error translating group");
            return;
        }
        scaffoldProperties.remove(translateList);
        scaffoldProperties.add(translateRow, translateList);
    }


    public void invertSelection(List<Feature2D> contigs) {
        List<Integer> contigIds = contig2DListToIntegerList(contigs);
        invertSelection(contigIds, getScaffoldRow(contigIds));
    }

    private void invertSelection(List<Integer> contigIds, int scaffoldRow) {
        //split group into three or two, invert selection, regroup
        List<Integer> selectedGroup = scaffoldProperties.get(scaffoldRow);
        int startIndex = selectedGroup.indexOf(contigIds.get(0));
        int endIndex = selectedGroup.indexOf(contigIds.get(contigIds.size() - 1));
        List<Integer> invertGroup = selectedGroup.subList(startIndex, endIndex);
        Collections.reverse(invertGroup);

        for (int i = startIndex; i <= endIndex; i++) {
            selectedGroup.set(i, invertGroup.get(i));
        }

    }

    public int getScaffoldRow(List<Integer> contigIds) {
        int i = 0;
        for (List<Integer> scaffoldRow : scaffoldProperties) {
            if (scaffoldRow.containsAll(contigIds))
                return i;
            i++;
        }
        System.err.println("Can't Find row");
        return -1;
    }

    public List<Integer> IntegerListToScaffoldList(List<Integer> contigIds) {
        List<Integer> absoluteValueContigs = findAbsoluteValuesList(contigIds);
        for (List<Integer> scaffoldRow : scaffoldProperties) {
            List<Integer> absoluteValuesOfScaffoldRow = findAbsoluteValuesList(scaffoldRow);
            if (absoluteValuesOfScaffoldRow.containsAll(absoluteValueContigs)) {
                int startIndex = absoluteValuesOfScaffoldRow.indexOf(absoluteValueContigs.get(0));
                int endIndex = absoluteValuesOfScaffoldRow.indexOf(absoluteValueContigs.get(absoluteValueContigs.size() - 1));
                return scaffoldRow.subList(startIndex, endIndex);
            }
        }
        System.err.println("error Converting Int list to scaffold List");
        return null;
    }

    public List<Integer> findAbsoluteValuesList(List<Integer> list) {
        List<Integer> newList = new ArrayList<>();
        for (int element : list) {
            newList.add(Math.abs(element));
        }
        return newList;
    }

    public int findAdjacentGroups(List<Integer> contigIds) {
        int i = 0;
        for (List<Integer> scaffoldRow : scaffoldProperties) {
            if (scaffoldRow.contains(contigIds.get(0)) && scaffoldProperties.get(i + 1).contains(contigIds.get(contigIds.size() - 1)))
                return i;
            i++;
        }
        return -1;
    }

    public void splitConitg(Feature2D originalContig, Feature2D debrisContig, SuperAdapter superAdapter, HiC hic) {

    }

    public ContigProperty contig2DtoContigProperty(Feature2D feature2D) {
        for (ContigProperty contigProperty : contigProperties) {
            if (contigProperty.getFeature2D() == feature2D) //make sure it is okay
                return contigProperty;
        }
        return null;
    }

    public int contig2DtoIndexInteger(Feature2D feature2D) {
        for (ContigProperty contigProperty : contigProperties) {
            if (contigProperty.getFeature2D().getStart1() == feature2D.getStart1())
                return contigProperty.getIndexId();
        }
        System.err.println("error finding corresponding contig");
        return -1;
    }

    public List<Integer> contig2DListToIntegerList(List<Feature2D> contigs) {
        List<Integer> contigIds = new ArrayList<Integer>();
        for (Feature2D feature2D : contigs) {
            int indexId = contig2DtoIndexInteger(feature2D);
            if (indexId != -1) {
                contigIds.add(indexId);
            }
            else
                System.err.println("error parsing 2DFeature");
        }
        return IntegerListToScaffoldList(contigIds);
    }
}