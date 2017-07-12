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

import juicebox.track.feature.Contig2D;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;

import java.awt.*;
import java.util.*;
import java.util.List;

/**
 * Created by nathanielmusial on 6/30/17.
 */
public class AssemblyFragmentHandler {

    private final String contigName = "Contig Name";
    private final String scaffoldIndexId = "Scaffold Index";
    private final String scaffoldNum = "Scaffold Number";
    private List<ContigProperty> contigProperties;
    private List<List<Integer>> scaffoldProperties;
    private Feature2DList contigs;
    private Feature2DList scaffolds;
    private String chromosomeName = "assembly";
    private OperationType operationType;
    public AssemblyFragmentHandler(List<ContigProperty> contigProperties, List<List<Integer>> scaffoldProperties) {
        this.contigProperties = contigProperties;
        this.scaffoldProperties = scaffoldProperties;
        contigs = new Feature2DList();
        scaffolds = new Feature2DList();
        generateContigsAndScaffolds();
        this.operationType = OperationType.NONE;
    }

    public AssemblyFragmentHandler(AssemblyFragmentHandler assemblyFragmentHandler) {
        this.contigProperties = assemblyFragmentHandler.cloneContigProperties();
        this.scaffoldProperties = assemblyFragmentHandler.cloneScaffoldProperties();
        this.contigs = new Feature2DList();
        this.scaffolds = new Feature2DList();
        generateContigsAndScaffolds();
    }

    public OperationType getOperationType() {
        return this.operationType;
    }

    public List<ContigProperty> cloneContigProperties() {
        List<ContigProperty> newList = new ArrayList<>();
        for (ContigProperty contigProperty : contigProperties) {
            newList.add(new ContigProperty(contigProperty));
        }
        return newList;
    }

    public List<List<Integer>> cloneScaffoldProperties() {
        List<List<Integer>> newList = new ArrayList<>();
        for (List<Integer> scaffoldRow : scaffoldProperties) {
            newList.add(new ArrayList<Integer>(scaffoldRow));
        }
        return newList;
    }

    public Feature2DList getContigs() {
        return contigs;
    }

    public void setContigs(Feature2DList contigs) {
        this.contigs = contigs;
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

    public void generateInitialContigsAndScaffolds() {
        contigs = new Feature2DList();
        scaffolds = new Feature2DList();
        int contigStartPos = 0;
        int scaffoldStartPos = 0;
        int scaffoldLength = 0;
        Integer rowNum = 0;
        for (List<Integer> row : scaffoldProperties) {
            for (Integer contigIndex : row) {
                ContigProperty contigProperty = contigProperties.get(Math.abs(contigIndex) - 1);
                String contigName = contigProperty.getName();
                Integer contigLength = contigProperty.getLength();

                Map<String, String> attributes = new HashMap<String, String>();
                attributes.put(this.contigName, contigName);
                attributes.put(scaffoldIndexId, contigIndex.toString());
                //put attribute here
                Feature2D feature2D = new Feature2D(Feature2D.FeatureType.CONTIG, chromosomeName, contigStartPos, (contigStartPos + contigLength),
                        chromosomeName, contigStartPos, (contigStartPos + contigLength),
                        new Color(0, 255, 0), attributes); //todo

                Contig2D contig = feature2D.toContig();
                if (contigProperty.isInverted()) {
                    contig.toggleInversion(); //assuming initial contig2D inverted = false
                }


                contigProperty.setIntialState(chromosomeName, contigStartPos, (contigStartPos + contigLength));

                contig.setInitialState(contigProperty.getInitialChr(), contigProperty.getInitialStart(), contigProperty.getInitialEnd());


                contigs.add(1, 1, contig);
                contigProperty.setFeature2D(contig);

                contigStartPos += contigLength;
                scaffoldLength += contigLength;
            }
            Map<String, String> attributes = new HashMap<String, String>();
            attributes.put(scaffoldNum, rowNum.toString());

            Feature2D scaffold = new Feature2D(Feature2D.FeatureType.SCAFFOLD, chromosomeName, scaffoldStartPos, (scaffoldStartPos + scaffoldLength),
                    chromosomeName, scaffoldStartPos, (scaffoldStartPos + scaffoldLength),
                    new Color(0, 0, 255), attributes);
            scaffolds.add(1, 1, scaffold);

            scaffoldStartPos += scaffoldLength;
            scaffoldLength = 0;
            rowNum++;
        }
    }

    public void generateContigsAndScaffolds() {
        contigs = new Feature2DList();
        scaffolds = new Feature2DList();
        int contigStartPos = 0;
        int scaffoldStartPos = 0;
        int scaffoldLength = 0;
        Integer rowNum = 0;
        for (List<Integer> row : scaffoldProperties) {
            for (Integer contigIndex : row) {
                ContigProperty contigProperty = contigProperties.get(Math.abs(contigIndex) - 1);
                String contigName = contigProperty.getName();
                Integer contigLength = contigProperty.getLength();

                Map<String, String> attributes = new HashMap<String, String>();
                attributes.put(this.contigName, contigName);
                attributes.put(scaffoldIndexId, contigIndex.toString());
                //put attribute here
                Feature2D feature2D = new Feature2D(Feature2D.FeatureType.CONTIG, chromosomeName, contigStartPos, (contigStartPos + contigLength),
                        chromosomeName, contigStartPos, (contigStartPos + contigLength),
                        new Color(0, 255, 0), attributes); //todo

                Contig2D contig = feature2D.toContig();
                if (contigProperty.isInverted()) {
                    contig.toggleInversion(); //assuming initial contig2D inverted = false
                }

                contig.setInitialState(contigProperty.getInitialChr(), contigProperty.getInitialStart(), contigProperty.getInitialEnd());

                contigs.add(1, 1, contig);
                contigProperty.setFeature2D(contig);

                contigStartPos += contigLength;
                scaffoldLength += contigLength;
            }
            Map<String, String> attributes = new HashMap<String, String>();
            attributes.put(scaffoldNum, rowNum.toString());

            Feature2D scaffold = new Feature2D(Feature2D.FeatureType.SCAFFOLD, chromosomeName, scaffoldStartPos, (scaffoldStartPos + scaffoldLength),
                    chromosomeName, scaffoldStartPos, (scaffoldStartPos + scaffoldLength),
                    new Color(0, 0, 255), attributes);
            scaffolds.add(1, 1, scaffold);

            scaffoldStartPos += scaffoldLength;
            scaffoldLength = 0;
            rowNum++;
        }
    }

    public void editContig(Feature2D originalFeature, Feature2D debrisContig) {
        operationType = OperationType.EDIT;
        ArrayList<Feature2D> contigs = new ArrayList<>();
        contigs.add(originalFeature);
        int scaffoldRowNum = getScaffoldRow(contig2DListToIntegerList(contigs));
        boolean inverted = originalFeature.getAttribute(scaffoldIndexId).contains("-");  //if contains a negative then it is inverted

        ContigProperty originalContig = feature2DtoContigProperty(originalFeature);
        int indexId = Integer.parseInt(originalFeature.getAttribute(scaffoldIndexId));
        List<ContigProperty> splitContigs = splitContig(originalFeature, debrisContig, originalContig);

        for (ContigProperty contigProperty : splitContigs) {
            System.out.println(contigProperty);
        }
        addContigProperties(originalContig, splitContigs);
        addScaffoldProperties(indexId, inverted, splitContigs, scaffoldRowNum, scaffoldProperties.get(scaffoldRowNum).indexOf(indexId));
    }

    public ContigProperty feature2DtoContigProperty(Feature2D feature2D) {
        for (ContigProperty contigProperty : contigProperties) {
            if (contigProperty.getFeature2D().getStart1() == feature2D.getStart1())
                return contigProperty;
        }
        System.err.println("error finding corresponding contig");
        return null;
    }

    public List<ContigProperty> splitContig(Feature2D originalFeature, Feature2D debrisFeature, ContigProperty originalContig) {

        boolean inverted = originalFeature.getAttribute(scaffoldIndexId).contains("-");  //if contains a negative then it is inverted

        if (originalFeature.overlapsWith(debrisFeature)) {
            if (!inverted)  //not inverted
                return generateNormalSplit(originalFeature, debrisFeature, originalContig);
            else  //is inverted so flip order of contigs
                return generateInvertedSplit(originalFeature, debrisFeature, originalContig);
        } else {
            System.out.println("error splitting contigs");
            return null;
        }
    }

    public List<ContigProperty> generateNormalSplit(Feature2D originalFeature, Feature2D debrisFeature, ContigProperty originalContig) {
        List<ContigProperty> splitContig = new ArrayList<>();
        List<String> newContigNames = getNewContigNames(originalContig);

        int originalIndexId = originalContig.getIndexId();
        int originalStart = originalFeature.getStart1();
        int debrisStart = debrisFeature.getStart1();
        int originalEnd = originalFeature.getEnd2();
        int debrisEnd = debrisFeature.getEnd1();
        int length;

        length = debrisStart - originalStart;
        ContigProperty firstFragment = new ContigProperty(newContigNames.get(0), originalIndexId, length);
        splitContig.add(firstFragment);

        length = debrisEnd - debrisStart;
        ContigProperty secondFragment = new ContigProperty(newContigNames.get(1), (originalIndexId + 1), length);
        splitContig.add(secondFragment);

        length = originalEnd - debrisEnd;
        ContigProperty thirdFragment = new ContigProperty(newContigNames.get(2), (originalIndexId + 2), length);
        splitContig.add(thirdFragment);

        setInitialStatesBasedOnOriginalContig(originalContig, splitContig);
        return splitContig;
    }

    public List<ContigProperty> generateInvertedSplit(Feature2D originalFeature, Feature2D debrisFeature, ContigProperty originalContig) {
        List<ContigProperty> splitContig = new ArrayList<>();
        List<String> newContigNames = getNewContigNames(originalContig);

        int originalIndexId = originalContig.getIndexId();
        int originalStart = originalFeature.getStart1();
        int debrisStart = debrisFeature.getStart1();
        int originalEnd = originalFeature.getEnd2();
        int debrisEnd = debrisFeature.getEnd1();
        int length;

        length = originalEnd - debrisEnd;
        ContigProperty firstFragment = new ContigProperty(newContigNames.get(0), (originalIndexId + 2), length);
        splitContig.add(firstFragment);

        length = debrisEnd - debrisStart;
        ContigProperty secondFragment = new ContigProperty(newContigNames.get(1), (originalIndexId + 1), length);
        splitContig.add(secondFragment);

        length = debrisStart - originalStart;
        ContigProperty thirdFragment = new ContigProperty(newContigNames.get(2), originalIndexId, length);
        splitContig.add(thirdFragment);

        setInitialStatesBasedOnOriginalContig(originalContig, splitContig);
        return splitContig;
    }

    public void setInitialStatesBasedOnOriginalContig(ContigProperty originalContig, List<ContigProperty> splitContig) {
        int newInitialStart;
        int newInitialEnd;
        if (originalContig.isInverted()) {
            newInitialEnd = originalContig.getInitialEnd();
            newInitialStart = newInitialEnd;
            for (ContigProperty contigProperty : splitContig) {
                //    50-100, 40-50, 0-40
                newInitialStart = newInitialStart - contigProperty.getLength();
                contigProperty.setIntialState(originalContig.getInitialChr(), newInitialStart, newInitialEnd);
                contigProperty.toggleInversion(); //toggles inversion of split contig
                newInitialEnd = newInitialStart;
            }
        } else { //not inverted
            newInitialStart = originalContig.getInitialStart();
            newInitialEnd = newInitialStart;
            //    0-40, 40-50, 50-100
            for (ContigProperty contigProperty : splitContig) {
                newInitialEnd = newInitialEnd + contigProperty.getLength();
                contigProperty.setIntialState(originalContig.getInitialChr(), newInitialStart, newInitialEnd);
                newInitialStart = newInitialEnd;
            }
        }
    }

    public List<String> getNewContigNames(ContigProperty contigProperty) {
        String fragmentString = ":::fragment_";
        String contigName = contigProperty.getName();
        List<String> newNames = new ArrayList<String>();
        if (contigName.contains(":::")) {         //        Hmel202003:::fragment_2
            if (contigName.contains("debris")) {
                System.err.println("cannot split a debris fragment");
            } else {
                String originalContigName = contigProperty.getOriginalContigName();
                int fragmentNum = contigProperty.getFragmentNumber();
                newNames.add(originalContigName + fragmentString + fragmentNum);
                newNames.add(originalContigName + fragmentString + (fragmentNum + 1) + ":::debris");
                newNames.add(originalContigName + fragmentString + (fragmentNum + 2));
            }
        } else {  //        Hmel202003
            newNames.add(contigName + fragmentString + "1");
            newNames.add(contigName + fragmentString + "2:::debris");
            newNames.add(contigName + fragmentString + "3");
        }
        return newNames;
    }

    public void addScaffoldProperties(int splitIndex, boolean inverted, List<ContigProperty> splitContigs, int rowNum, int posNum) {
        List<Integer> splitContigsIds = new ArrayList<>();
        int multiplier;
        if (inverted)
            multiplier = -1;
        else
            multiplier = 1;

        for (ContigProperty contigProperty : splitContigs) {
            splitContigsIds.add(multiplier * contigProperty.getIndexId());

        }
        for (int i : splitContigsIds) {
            System.out.print(i + "\t");
        }
        System.out.println();
        shiftScaffoldProperties(splitIndex);

        scaffoldProperties.get(rowNum).addAll(posNum, splitContigsIds);
        scaffoldProperties.get(rowNum).remove(posNum + 3);
    }

    public void shiftScaffoldProperties(int splitIndex) {
        int i;

        for (List<Integer> scaffoldRow : scaffoldProperties) {
            i = 0;
            for (Integer indexId : scaffoldRow) {
                if (Math.abs(indexId) > (Math.abs(splitIndex))) {
                    if (indexId < 0)
                        scaffoldRow.set(i, indexId - 2);
                    else
                        scaffoldRow.set(i, indexId + 2);
                }
                i++;
            }
        }
    }

    public void addContigProperties(ContigProperty originalContig, List<ContigProperty> splitContig) {
        int splitContigIndex = contigProperties.indexOf(originalContig);
        shiftContigIndices(splitContigIndex);

        List<ContigProperty> fromInitialContig = findContigsSplitFromInitial(originalContig);
        for (ContigProperty contigProperty : fromInitialContig) {
            System.out.println(contigProperty);
        }
        if (fromInitialContig.indexOf(originalContig) != fromInitialContig.size() - 1) {
            List<ContigProperty> shiftedContigs = fromInitialContig.subList(fromInitialContig.indexOf(originalContig) + 1, fromInitialContig.size());
            for (ContigProperty contigProperty : shiftedContigs) {
                String newContigName = contigProperty.getName();
                if (newContigName.contains(":::debris")) {
                    newContigName = newContigName.replaceFirst("_.*", "_" + (contigProperty.getFragmentNumber() + 2) + ":::debris");
                } else {
                    newContigName = newContigName.replaceFirst("_\\d+", "_" + (contigProperty.getFragmentNumber() + 2));
                }
                contigProperty.setName(newContigName);
                System.out.println(contigProperty);
            }

        } else {

            System.out.println("error adding edited contigs"); //todo fix if statement
        }

        contigProperties.addAll(splitContigIndex, splitContig);
        contigProperties.remove(originalContig);
    }

    public void shiftContigIndices(int splitIndexId) {
        for (ContigProperty contigProperty : contigProperties) {
            if (Math.abs(contigProperty.getIndexId()) > (Math.abs(splitIndexId) + 1)) {
                contigProperty.setIndexId(contigProperty.getIndexId() + 1);
            }
        }
    }

    public List<ContigProperty> findContigsSplitFromInitial(ContigProperty originalContig) {
        List<ContigProperty> contigPropertiesFromSameInitial = new ArrayList<>();
        String originalContigName = originalContig.getOriginalContigName();
        System.out.println(originalContigName);

        for (ContigProperty contigProperty : contigProperties) {
            if (contigProperty.getName().contains(originalContigName))
                contigPropertiesFromSameInitial.add(contigProperty);
        }
        return contigPropertiesFromSameInitial;
    }

    public void splitGroup(List<Feature2D> contigs) {
        operationType = OperationType.GROUP;
        List<Integer> contigIds = contig2DListToIntegerList(contigs);
        int scaffoldRowNum = getScaffoldRow(contigIds);
        splitGroup(contigIds, scaffoldRowNum);
    }

    private List<List<Integer>> splitGroup(List<Integer> contigIds, int scaffoldRow) {
        List<Integer> tempGroup = scaffoldProperties.get(scaffoldRow);
        int startIndex = tempGroup.indexOf(contigIds.get(0));
        int endIndex = startIndex + contigIds.size();

        List<List<Integer>> newGroups = new ArrayList<>();
        if (startIndex != 0) {
            if (endIndex == tempGroup.size()) {
                newGroups.add(tempGroup.subList(0, startIndex));
                newGroups.add(tempGroup.subList(startIndex, endIndex));
            } else {
                newGroups.add(tempGroup.subList(0, startIndex));
                newGroups.add(tempGroup.subList(startIndex, endIndex));
                newGroups.add(tempGroup.subList(endIndex, tempGroup.size()));
            }
        } else {
            newGroups.add(tempGroup.subList(0, endIndex));
            newGroups.add(tempGroup.subList(endIndex, tempGroup.size()));
        }
        scaffoldProperties.addAll(scaffoldRow, newGroups);
        scaffoldProperties.remove(tempGroup);

        return newGroups;
    }

    public void mergeGroup(int startingIndex, List<Feature2D> contigs) {
        operationType = OperationType.GROUP;
        if (contigs.size() > 1) {
            mergeGroup(startingIndex, contigs.size());
        } else
            System.err.println("not enough many selected");

    }

    private void mergeGroup(int scaffoldRowNum, int length) {
        List<Integer> newGroup = new ArrayList<>();
        List<List<Integer>> removeGroup = new ArrayList<>();
        for (int i = 0; i < length; i++) {
            List<Integer> currentGroup = scaffoldProperties.get(scaffoldRowNum + i);
            for (Integer element : currentGroup) {
                newGroup.add(element);
            }
            removeGroup.add(currentGroup);
        }

        scaffoldProperties.set(scaffoldRowNum, newGroup);
        removeGroup.remove(0);
        for (List<Integer> removeList : removeGroup) {
            scaffoldProperties.remove(removeList);
        }
    }

    public void translateSelection(List<Feature2D> selectedFeatures, Feature2D featureDestination) {
        operationType = OperationType.TRANSLATE;
        int destinationRow;
        int destinationPos;
        List<Feature2D> tempList = new ArrayList<Feature2D>();
        tempList.add(featureDestination);
        List<Integer> destinationIndexId = contig2DListToIntegerList(tempList);
        destinationRow = getScaffoldRow(destinationIndexId);
        destinationPos = scaffoldProperties.get(destinationRow).indexOf(destinationIndexId.get(0));
        System.out.println(destinationPos);

        performTranslation(contig2DListToIntegerList(selectedFeatures), destinationRow, destinationPos);

    }

    private void performTranslation(List<Integer> contigIds, int translateRow, int translatePos) {
        int originalRowNum = getScaffoldRow(contigIds);
        List<Integer> originalRow = scaffoldProperties.get(originalRowNum);
        for (Integer integer : contigIds) {
            System.out.println(integer);
        }
        originalRow.removeAll(contigIds);
        scaffoldProperties.get(translateRow).addAll(translatePos, contigIds);
    }

    public void invertSelection(List<Feature2D> contigs) {
        operationType = OperationType.INVERT;
        List<Integer> contigIds = contig2DListToIntegerList(contigs);
        System.out.println("contig 1 " + contigIds);
        //invert selected contig properties
        List<ContigProperty> selectedContigProperties = contig2DListToContigPropertyList(contigs);
        for (ContigProperty contigProperty : selectedContigProperties) {
            contigProperty.toggleInversion();
        }
        invertSelection(contigIds, getScaffoldRow(contigIds));
        System.out.println("test 1");
    }

    private void invertSelection(List<Integer> contigIds, int scaffoldRow) {
        //split group into three or two, invert selection, regroup
        List<Integer> selectedGroup = scaffoldProperties.get(scaffoldRow);
        int startIndex = selectedGroup.indexOf(contigIds.get(0));
        int endIndex = startIndex + contigIds.size();
//        System.out.println("Start Index "+startIndex+" EndIndex "+endIndex);
        List<Integer> invertGroup = selectedGroup.subList(startIndex, endIndex);
//        System.out.println("Test 1 "+invertGroup);
        Collections.reverse(invertGroup);
        int i = 0;
        for (int element : invertGroup) {
            invertGroup.set(i, element * -1);
            i++;
        }

        System.out.println(invertGroup);
    }

    public int getScaffoldRow(List<Integer> contigIds) {
        int i = 0;
        for (List<Integer> scaffoldRow : scaffoldProperties) {
            List<Integer> absoluteRow = findAbsoluteValuesList(scaffoldRow);
            if (absoluteRow.containsAll(findAbsoluteValuesList(contigIds)))
                return i;
            i++;
        }
        System.err.println("Can't Find row");
        return -1;
    }

    public List<Integer> findAbsoluteValuesList(List<Integer> list) {
        List<Integer> newList = new ArrayList<>();
        for (int element : list) {
            newList.add(Math.abs(element));
        }
        return newList;
    }

    public List<Integer> contig2DListToIntegerList(List<Feature2D> contigs) {
        List<Integer> contigIds = new ArrayList<Integer>();
        for (Feature2D feature2D : contigs) {
            contigIds.add(Integer.parseInt(feature2D.getAttribute(scaffoldIndexId)));
        }
        return contigIds;
    }

    public List<ContigProperty> contig2DListToContigPropertyList(List<Feature2D> contigs) {
        List<ContigProperty> newList = new ArrayList<ContigProperty>();
        for (Feature2D feature2D : contigs) {
            newList.add(contig2DToContigProperty(feature2D));
        }
        return newList;
    }

    public ContigProperty contig2DToContigProperty(Feature2D feature2D) {
        for (ContigProperty contigProperty : contigProperties) {
            if (contigProperty.getFeature2D().equals(feature2D)) { //make sure it is okay

                return contigProperty;
            }
        }
        return null;
    }

    public enum OperationType {EDIT, INVERT, TRANSLATE, GROUP, NONE}

}