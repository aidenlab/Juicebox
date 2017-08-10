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
import org.broad.igv.util.Pair;

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
    private final String initiallyInvertedStatus = "Initially Inverted";
    private List<ContigProperty> contigProperties;
    private List<List<Integer>> scaffoldProperties;
    private Feature2DList contigs;
    private Feature2DList scaffolds;
    private String chromosomeName = "assembly";
    private Contig2D guessContig = null;
    private Integer debrisContigIndex;

    public AssemblyFragmentHandler(List<ContigProperty> contigProperties, List<List<Integer>> scaffoldProperties) {
        this.contigProperties = contigProperties;
        this.scaffoldProperties = scaffoldProperties;
        contigs = new Feature2DList();
        scaffolds = new Feature2DList();
        generateContigsAndScaffolds(false, false, this);
        debrisContigIndex = null;
    }

    public AssemblyFragmentHandler(AssemblyFragmentHandler assemblyFragmentHandler) {
        this.contigProperties = assemblyFragmentHandler.cloneContigProperties();
        this.scaffoldProperties = assemblyFragmentHandler.cloneScaffoldProperties();
        this.contigs = new Feature2DList();
        this.scaffolds = new Feature2DList();
        if (assemblyFragmentHandler.debrisContigIndex == null) {
            this.debrisContigIndex = null;
        } else {
            this.debrisContigIndex = new Integer(assemblyFragmentHandler.debrisContigIndex);
        }
        generateContigsAndScaffolds(false, false, assemblyFragmentHandler);
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

    public void generateContigsAndScaffolds(boolean initialGeneration, boolean modifiedGeneration, AssemblyFragmentHandler originalAssemblyFragmentHandler) {
        Map<String, Pair<List<ContigProperty>, List<ContigProperty>>> splitFragments = new HashMap<>();
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

                if (initialGeneration && !modifiedGeneration) {
                    contigProperty.setInitialState(chromosomeName, contigStartPos, (contigStartPos + contigLength), contigProperty.isInverted());
                }
                if (modifiedGeneration) {
                    List<ContigProperty> newList = getOriginalContigProperty(originalAssemblyFragmentHandler, contigProperty);
                    ContigProperty originalContigProperty = newList.get(0);
                    boolean invertedInAsm = false;
                    if (contigProperty.getIndexId() != originalContigProperty.getIndexId()) {
                        invertedInAsm = true;
                    }
                    contigProperty.setInitialState(chromosomeName, originalContigProperty.getInitialStart(), originalContigProperty.getInitialEnd(), invertedInAsm);
                    contigProperty.setInitiallyInverted(originalContigProperty.wasInitiallyInverted());

                    if (contigProperty.getName().contains("fragment")) {
                        addRelevantOriginalContigProperties(newList, splitFragments, contigProperty);
                    }
                }

                Map<String, String> attributes = new HashMap<String, String>();
                attributes.put(this.contigName, contigName);
                attributes.put(scaffoldIndexId, contigIndex.toString());
                attributes.put(initiallyInvertedStatus, Boolean.toString(contigProperty.wasInitiallyInverted()));
                //put attribute here
                Feature2D feature2D = new Feature2D(Feature2D.FeatureType.CONTIG, chromosomeName, contigStartPos, (contigStartPos + contigLength),
                        chromosomeName, contigStartPos, (contigStartPos + contigLength),
                        new Color(0, 255, 0), attributes); //todo

                Contig2D contig = feature2D.toContig();
                if (contigProperty.isInverted()) {
                    contig.toggleInversion(); //assuming initial contig2D inverted = false
                }
                contig.setInitialState(contigProperty.getInitialChr(), contigProperty.getInitialStart(), contigProperty.getInitialEnd(), contigProperty.wasInitiallyInverted());
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
        if (modifiedGeneration && !splitFragments.isEmpty()) {
            fixInitialState(splitFragments);
            printRelevantOriginalContigProperties(splitFragments);
        }
    }

    public List<ContigProperty> getOriginalContigProperty(AssemblyFragmentHandler originalAssemblyFragmentHandler, ContigProperty lookupContigProperty) {
        List<ContigProperty> originalContigProperties = originalAssemblyFragmentHandler.getContigProperties();
        List<ContigProperty> newList = new ArrayList<>();
        for (ContigProperty contigProperty : originalContigProperties) {
            if (lookupContigProperty.getName().contains(contigProperty.getOriginalContigName())) {
                newList.add(contigProperty);
            }
        }
        return newList;
    }

    public void addRelevantOriginalContigProperties(List<ContigProperty> originalContigProperties, Map<String, Pair<List<ContigProperty>, List<ContigProperty>>> splitFragments, ContigProperty lookupContigProperty) {
        String originalContigName = lookupContigProperty.getOriginalContigName();

        for (ContigProperty originalContigProperty : originalContigProperties) {
            Pair<List<ContigProperty>, List<ContigProperty>> newEntry = new Pair<List<ContigProperty>, List<ContigProperty>>(new ArrayList<ContigProperty>(), new ArrayList<ContigProperty>());
            if (!splitFragments.containsKey(originalContigName)) {
                Pair<List<ContigProperty>, List<ContigProperty>> newPair = new Pair<List<ContigProperty>, List<ContigProperty>>(new ArrayList<ContigProperty>(), new ArrayList<ContigProperty>());
                splitFragments.put(originalContigName, newPair);
            }
            newEntry = splitFragments.get(originalContigName);
            if (!newEntry.getFirst().contains(originalContigProperty))
                newEntry.getFirst().add(originalContigProperty);
            if (!newEntry.getSecond().contains(lookupContigProperty))
                newEntry.getSecond().add(lookupContigProperty);
        }
    }

    public void printRelevantOriginalContigProperties(Map<String, Pair<List<ContigProperty>, List<ContigProperty>>> splitFragments) {
        for (String key : splitFragments.keySet()) {
            Pair<List<ContigProperty>, List<ContigProperty>> pair = splitFragments.get(key);
            System.out.println("Old fragments");
            for (ContigProperty contigProperty : pair.getFirst()) {
                System.out.println(contigProperty.getName());
            }
            System.out.println("New fragments");
            for (ContigProperty contigProperty : pair.getSecond()) {
                System.out.println(contigProperty.getName());
            }
            System.out.println();
        }
    }

    public void fixInitialState(Map<String, Pair<List<ContigProperty>, List<ContigProperty>>> newSplitFragments) {
        Comparator<ContigProperty> comparator = new Comparator<ContigProperty>() {
            @Override
            public int compare(final ContigProperty o1, final ContigProperty o2) {
                return o1.getName().compareTo(o2.getName());
            }
        };
        for (String key : newSplitFragments.keySet()) {
            //reconstruct from here
            Pair<List<ContigProperty>, List<ContigProperty>> entry = newSplitFragments.get(key);
            List<ContigProperty> splitList = entry.getSecond();
            List<ContigProperty> originalContigList = entry.getFirst();
            Collections.sort(splitList, comparator);
            Collections.sort(originalContigList, comparator);


            if (splitList.size() == originalContigList.size()) {
                for (int i = 0; i < splitList.size(); i++) {
                    setInitialStatesBasedOnOriginalContig(originalContigList.get(i), splitList.subList(i, i + 1), originalContigList.get(i).wasInitiallyInverted());
                }
            } else {
                int index = 0;
                for (ContigProperty originalContigProperty : originalContigList) {
                    List<ContigProperty> neededFragments = new ArrayList<>();
                    int sumLength = 0;
                    int length = originalContigProperty.getLength();
                    while (sumLength != length) {
                        neededFragments.add(splitList.get(index));
                        sumLength += splitList.get(index).getLength();
                        index++;
                    }
                    if (originalContigProperty.wasInitiallyInverted()) {  //not sure if needed yet depends
                        Collections.reverse(neededFragments);
                    }
                    setInitialStatesBasedOnOriginalContig(originalContigProperty, neededFragments, originalContigProperty.wasInitiallyInverted());
                }
            }
        }
    }
    //**** Splitting ****//

    public void editContig(Feature2D originalFeature, Feature2D debrisContig) {
        ArrayList<Feature2D> contigs = new ArrayList<>();
        contigs.add(originalFeature);
        int scaffoldRowNum = getScaffoldRow(contig2DListToIntegerList(contigs));
        boolean invertedInAsm = originalFeature.getAttribute(scaffoldIndexId).contains("-");  //if contains a negative then it is inverted

        ContigProperty originalContig = feature2DtoContigProperty(originalFeature);
        int indexId = Integer.parseInt(originalFeature.getAttribute(scaffoldIndexId));
        List<ContigProperty> splitContigs = splitContig(invertedInAsm, originalFeature, debrisContig, originalContig);

        addContigProperties(originalContig, splitContigs);
        addScaffoldProperties(indexId, invertedInAsm, splitContigs, scaffoldRowNum, scaffoldProperties.get(scaffoldRowNum).indexOf(indexId));

        debrisContigIndex = splitContigs.get(1).getIndexId();
        debrisContigIndex = findInvertedContigIndex(debrisContigIndex);
//        System.out.println("dindex "+debrisContigIndex);
    }

    public ContigProperty feature2DtoContigProperty(Feature2D feature2D) {
        for (ContigProperty contigProperty : contigProperties) {
            if (contigProperty.getFeature2D().getStart1() == feature2D.getStart1())
                return contigProperty;
        }
        System.err.println("error finding corresponding contig");
        return null;
    }

    public List<ContigProperty> splitContig(boolean invertedInAsm, Feature2D originalFeature, Feature2D debrisFeature, ContigProperty originalContig) {


        if (originalFeature.overlapsWith(debrisFeature)) {
            if (!invertedInAsm)  //not inverted
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
        boolean initiallyInverted = originalContig.wasInitiallyInverted();

        length = debrisStart - originalStart;
        ContigProperty firstFragment = new ContigProperty(newContigNames.get(0), originalIndexId, length, initiallyInverted);

        length = debrisEnd - debrisStart;
        ContigProperty secondFragment = new ContigProperty(newContigNames.get(1), (originalIndexId + 1), length, initiallyInverted);

        length = originalEnd - debrisEnd;
        ContigProperty thirdFragment = new ContigProperty(newContigNames.get(2), (originalIndexId + 2), length, initiallyInverted);

        splitContig.add(firstFragment);
        splitContig.add(secondFragment);
        splitContig.add(thirdFragment);

        setInitialStatesBasedOnOriginalContig(originalContig, splitContig, false);
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
        boolean initiallyInverted = originalContig.wasInitiallyInverted();

        length = originalEnd - debrisEnd;
        ContigProperty firstFragment = new ContigProperty(newContigNames.get(0), (originalIndexId + 2), length, initiallyInverted);

        length = debrisEnd - debrisStart;
        ContigProperty secondFragment = new ContigProperty(newContigNames.get(1), (originalIndexId + 1), length, initiallyInverted);

        length = debrisStart - originalStart;
        ContigProperty thirdFragment = new ContigProperty(newContigNames.get(2), originalIndexId, length, initiallyInverted);

        splitContig.add(thirdFragment);
        splitContig.add(secondFragment);
        splitContig.add(firstFragment);

        setInitialStatesBasedOnOriginalContig(originalContig, splitContig, true);
        return splitContig;
    }

    public void setInitialStatesBasedOnOriginalContig(ContigProperty originalContig, List<ContigProperty> splitContig, boolean invertedInAsm) {
        int newInitialStart;
        int newInitialEnd;
        boolean initiallyInverted = originalContig.wasInitiallyInverted();

        if (invertedInAsm && !initiallyInverted || !invertedInAsm && initiallyInverted) { //inverted in map
            newInitialEnd = originalContig.getInitialEnd();
            newInitialStart = newInitialEnd;
            for (ContigProperty contigProperty : splitContig) {
                //    50-100, 40-50, 0-40
                newInitialStart = newInitialStart - contigProperty.getLength();
                contigProperty.setInitialState(originalContig.getInitialChr(), newInitialStart, newInitialEnd, originalContig.isInverted());

                newInitialEnd = newInitialStart;
            }
        } else { //not inverted in map
            newInitialStart = originalContig.getInitialStart();
            newInitialEnd = newInitialStart;
            //    0-40, 40-50, 50-100
            for (ContigProperty contigProperty : splitContig) {
                newInitialEnd = newInitialEnd + contigProperty.getLength();
                contigProperty.setInitialState(originalContig.getInitialChr(), newInitialStart, newInitialEnd, originalContig.isInverted());
                newInitialStart = newInitialEnd;
            }
        }
    }

    public List<String> getNewContigNames(ContigProperty contigProperty) {
        String fragmentString = ":::fragment_";
        String contigName = contigProperty.getName();
        List<String> newNames = new ArrayList<String>();
        if (contigName.contains(":::")) {
            if (contigName.contains("debris")) {
                System.err.println("cannot split a debris fragment");
            } else {
                String originalContigName = contigProperty.getOriginalContigName();
                int fragmentNum = contigProperty.getFragmentNumber();
                newNames.add(originalContigName + fragmentString + fragmentNum);
                newNames.add(originalContigName + fragmentString + (fragmentNum + 1) + ":::debris");
                newNames.add(originalContigName + fragmentString + (fragmentNum + 2));
            }
        } else {
            newNames.add(contigName + fragmentString + "1");
            newNames.add(contigName + fragmentString + "2:::debris");
            newNames.add(contigName + fragmentString + "3");
        }
        return newNames;
    }

    public void addScaffoldProperties(int splitIndex, boolean invertedInAsm, List<ContigProperty> splitContigs, int rowNum, int posNum) {
        List<Integer> splitContigsIds = new ArrayList<>();
        int multiplier;
        if (invertedInAsm)
            multiplier = -1;
        else
            multiplier = 1;

        for (ContigProperty contigProperty : splitContigs) {
            splitContigsIds.add(multiplier * contigProperty.getIndexId());

        }

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
        if (fromInitialContig.indexOf(originalContig) != fromInitialContig.size() - 1) { //if there are framents past the one you are splitting
            List<ContigProperty> shiftedContigs = fromInitialContig.subList(fromInitialContig.indexOf(originalContig) + 1, fromInitialContig.size());
            for (ContigProperty contigProperty : shiftedContigs) {
                String newContigName = contigProperty.getName();
                if (newContigName.contains(":::debris")) {
                    newContigName = newContigName.replaceFirst("_.*", "_" + (contigProperty.getFragmentNumber() + 2) + ":::debris");
                } else {
                    newContigName = newContigName.replaceFirst("_\\d+", "_" + (contigProperty.getFragmentNumber() + 2));
                }
                contigProperty.setName(newContigName);
            }
        }

        contigProperties.addAll(splitContigIndex, splitContig);
        contigProperties.remove(originalContig);
    }

    public void shiftContigIndices(int splitIndexId) {
        for (ContigProperty contigProperty : contigProperties) {
            if (Math.abs(contigProperty.getIndexId()) > (Math.abs(splitIndexId) + 1)) {
                contigProperty.setIndexId(contigProperty.getIndexId() + 2);
            }
        }
    }

    public List<ContigProperty> findContigsSplitFromInitial(ContigProperty originalContig) {
        List<ContigProperty> contigPropertiesFromSameInitial = new ArrayList<>();
        String originalContigName = originalContig.getOriginalContigName();

        for (ContigProperty contigProperty : contigProperties) {
            if (contigProperty.getName().contains(originalContigName))
                contigPropertiesFromSameInitial.add(contigProperty);
        }
        return contigPropertiesFromSameInitial;
    }

    //**** Inversion ****//

    public void invertSelection(List<Feature2D> contigs) {

        List<Integer> contigIds = contig2DListToIntegerList(contigs);

        int id1 = contigIds.get(0);
        int id2 = contigIds.get(contigIds.size()-1);
        int gid1 = getGroupID(id1);
        int gid2 = getGroupID(id2);

        if (gid1!=gid2 && scaffoldProperties.get(gid1).indexOf(id1)!=0){
            newSplitGroup(gid1, scaffoldProperties.get(gid1).get(scaffoldProperties.get(gid1).indexOf(id1)-1));
            gid1 = getGroupID(id1);
            gid2 = getGroupID(id2);
        }
        if (gid1!=gid2 && scaffoldProperties.get(gid2).indexOf(id2)!=scaffoldProperties.get(gid2).size()-1){
            newSplitGroup(gid2, id2);
            gid1 = getGroupID(id1);
            gid2 = getGroupID(id2);
        }

        //invert selected contig properties
        List<ContigProperty> selectedContigProperties = contig2DListToContigPropertyList(contigs);
        for (ContigProperty contigProperty : selectedContigProperties) {
            contigProperty.toggleInversion();
        }

        if (gid1==gid2){
            Collections.reverse(scaffoldProperties.get(gid1).subList(scaffoldProperties.get(gid1).indexOf(id1),scaffoldProperties.get(gid2).indexOf(id2)+1));
            for(int i=scaffoldProperties.get(gid1).indexOf(id2); i<=scaffoldProperties.get(gid2).indexOf(id1); i++){
                scaffoldProperties.get(gid1).set(i,-1*scaffoldProperties.get(gid1).get(i));
            }
        } else {
            List<List<Integer>> newGroups = new ArrayList<>();
            for (int i=0; i<=scaffoldProperties.size()-1; i++ ){
                if(i>=gid1&&i<=gid2){
                    newGroups.add(scaffoldProperties.get(gid2-i+gid1));
                    Collections.reverse(newGroups.get(i));
                    for(int j=0; j<=newGroups.get(i).size()-1;j++){
                        newGroups.get(i).set(j, -1*newGroups.get(i).get(j));
                    }
                } else{
                    newGroups.add(scaffoldProperties.get(i));
                }
            }
            scaffoldProperties.clear();
            scaffoldProperties.addAll(newGroups);
        }

        return;

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

    //**** Move toggle ****//

    public void moveSelection(List<Feature2D> selectedFeatures, Feature2D upstreamFeature){
        int id1 = Integer.parseInt(selectedFeatures.get(0).getAttribute(scaffoldIndexId));
        int id2 = Integer.parseInt(selectedFeatures.get(selectedFeatures.size()-1).getAttribute(scaffoldIndexId));
        int id3 = Integer.parseInt(upstreamFeature.getAttribute(scaffoldIndexId));
        moveSelection(id1, id2, id3);
    }

    public void moveDebrisToEnd() {
        if (debrisContigIndex != null) {
            int id1 = debrisContigIndex;
            int id2 = debrisContigIndex;
            int counter = 0;

            List<Integer> lastRow = scaffoldProperties.get(scaffoldProperties.size() - 1);

            int id3 = lastRow.get(lastRow.size() - 1);
            moveSelection(id1, id2, id3);
            debrisContigIndex = null;
        }
    }

    public int findInvertedContigIndex(int id1) {
        for (List<Integer> scaffoldRow : scaffoldProperties) {

            for (int index : scaffoldRow) {
                if (Math.abs(index) == Math.abs(id1))
                    return index;
            }
        }
        System.err.println("error finding contigID");
        return -1;
    }

    public void moveSelection(int id1, int id2, int id3) {


        int gid1 = getGroupID(id1);
        int gid2 = getGroupID(id2);
        int gid3 = getGroupID(id3);

        // check if selectedFeatures span multiple groups paste split at destination
        if (gid1!=gid2 & scaffoldProperties.get(gid3).indexOf(id3)!=scaffoldProperties.get(gid3).size()-1){
            newSplitGroup(gid3,id3);
            gid1 = getGroupID(id1);
            gid2 = getGroupID(id2);
            gid3 = getGroupID(id3);
        }

        List<List<Integer>> newGroups = new ArrayList<>();
        List<List<Integer>> tempGroups = new ArrayList<>();
        List<Integer> truncatedGroup = new ArrayList<Integer>();
        int shiftGroup=0;

        for (int i=0; i<=scaffoldProperties.size()-1; i++){
            if (i==gid1 && i==gid2){

                tempGroups.add(scaffoldProperties.get(gid1).subList(scaffoldProperties.get(gid1).indexOf(id1), scaffoldProperties.get(gid2).indexOf(id2)+1));

                if (scaffoldProperties.get(gid1).indexOf(id1)!=0) {
                    truncatedGroup.addAll(scaffoldProperties.get(gid1).subList(0, scaffoldProperties.get(gid1).indexOf(id1)));
                }
                if (scaffoldProperties.get(gid2).indexOf(id2)!=scaffoldProperties.get(gid2).size()-1) {
                    truncatedGroup.addAll(scaffoldProperties.get(gid2).subList(1 + scaffoldProperties.get(gid2).indexOf(id2), scaffoldProperties.get(gid2).size()));
                }

                if (!truncatedGroup.isEmpty()){
                    newGroups.add(truncatedGroup);
                } else {
                    shiftGroup++;
                }

            } else if (gid1!=gid2 && i==gid1){
                tempGroups.add(scaffoldProperties.get(gid1).subList(scaffoldProperties.get(gid1).indexOf(id1), scaffoldProperties.get(gid1).size()));
                if (scaffoldProperties.get(gid1).indexOf(id1)!=0) {
                    newGroups.add(scaffoldProperties.get(gid1).subList(0, scaffoldProperties.get(gid1).indexOf(id1)));
                }else{
                    shiftGroup++;
                }
            } else if (gid1!=gid2 && i > gid1 && i < gid2){
                tempGroups.add(scaffoldProperties.get(i));
                shiftGroup++;
            } else if (gid1!=gid2 && i==gid2){
                tempGroups.add(scaffoldProperties.get(gid2).subList(0, 1 + scaffoldProperties.get(gid2).indexOf(id2)));
                if (scaffoldProperties.get(gid2).indexOf(id2)!=scaffoldProperties.get(gid2).size()-1){
                    newGroups.add(scaffoldProperties.get(gid2).subList(1 + scaffoldProperties.get(gid2).indexOf(id2), scaffoldProperties.get(gid2).size()));
                }else{
                    shiftGroup++;
                }
            } else {
                newGroups.add(scaffoldProperties.get(i));
            }
        }

        int newgid3=gid3;
        if (gid3 > gid2){
            newgid3-=shiftGroup;
        }

        if (scaffoldProperties.get(gid3).indexOf(id3) == scaffoldProperties.get(gid3).size()-1) {
            newGroups.addAll(newgid3 + 1, tempGroups);
        } else {
            int pasteIndex = scaffoldProperties.get(gid3).indexOf(id3);
            if (gid1 == gid3 && gid2 == gid3 && scaffoldProperties.get(gid3).indexOf(id3) > scaffoldProperties.get(gid3).indexOf(id2)){
                pasteIndex-=scaffoldProperties.get(gid3).size()-truncatedGroup.size();
            }
            newGroups.get(newgid3).addAll(pasteIndex+1,tempGroups.get(0));
        }

        scaffoldProperties.clear();
        scaffoldProperties.addAll(newGroups);

        return;
    }

    //**** Group toggle ****//

    public void toggleGroup(Feature2D upstreamFeature2D, Feature2D downstreamFeature2D) {
        int id1=Integer.parseInt(upstreamFeature2D.getAttribute(scaffoldIndexId));
        int id2=Integer.parseInt(downstreamFeature2D.getAttribute(scaffoldIndexId));

        //should not happen, other sanity checks?
        if (id1==id2){
            return;
        }

        int gr1 = getGroupID(id1);
        int gr2 = getGroupID(id2);

//        System.out.println(Arrays.toString(scaffoldProperties.toArray()));
        if (gr1==gr2){
//            System.out.println("calling split");
            newSplitGroup(gr1, id1);
//            System.out.println(Arrays.toString(scaffoldProperties.toArray()));
        }else {

//            System.out.println("calling merge");
            newMergeGroup(gr1, gr2);
//            System.out.println(Arrays.toString(scaffoldProperties.toArray()));
        }

    }

    private void newMergeGroup(int groupId1, int groupId2) {
        List<List<Integer>> newGroups = new ArrayList<>();
        for (int i=0; i<=scaffoldProperties.size()-1; i++){
            if (i == groupId2) {
                newGroups.get(groupId1).addAll(scaffoldProperties.get(groupId2));
            } else {
                newGroups.add(scaffoldProperties.get(i));
            }
        }
        scaffoldProperties.clear();
        scaffoldProperties.addAll(newGroups);
        return;
    }

    private void newSplitGroup(int groupId1, int id1) {
        List<List<Integer>> newGroups = new ArrayList<>();
        for (int i=0; i<=scaffoldProperties.size()-1; i++){
            if (i == groupId1) {
                newGroups.add(scaffoldProperties.get(groupId1).subList(0, 1 + scaffoldProperties.get(groupId1).indexOf(id1)));
                newGroups.add(scaffoldProperties.get(groupId1).subList(1 + scaffoldProperties.get(groupId1).indexOf(id1), scaffoldProperties.get(groupId1).size()));
            } else {
                newGroups.add(scaffoldProperties.get(i));
            }
        }
        scaffoldProperties.clear();
        scaffoldProperties.addAll(newGroups);
        return;
    }

    //**** Utility functions ****//

    private int getGroupID(int id1) {
        int i = 0;
//        System.out.println(id1);
        for (List<Integer> scaffoldRow : scaffoldProperties) {

            for (int index : scaffoldRow) {
                if (Math.abs(index) == Math.abs(id1))
                    return i;
            }
            i++;
        }
        System.err.println("Can't Find row");
        return -1;
    }

    //**** For debugging ****//
    public void printAssembly(){
        System.out.println(Arrays.toString(scaffoldProperties.toArray()));
        return;
    }

    public Contig2D liftAsmCoordinateToFragment(int chrId1, int chrId2, int asmCoordinate) {

        for (Feature2D contig : contigs.get(chrId1, chrId2)) {
            if (contig.getStart1() < asmCoordinate && contig.getEnd1() >= asmCoordinate) {
                return contig.toContig();
            }
        }
        return null;
    }

    public int liftAsmCoordinateToFragmentCoordinate(int chrId1, int chrId2, int asmCoordinate) {
        Contig2D contig = liftAsmCoordinateToFragment(chrId1, chrId2, asmCoordinate);
        if (contig == null) {
            return -1;
        }
        int newCoordinate;
        boolean inverted = contig.getAttribute(scaffoldIndexId).contains("-");
        if (inverted) {
            newCoordinate = contig.getEnd1() - asmCoordinate + 1;
        } else {
            newCoordinate = asmCoordinate - contig.getStart1();
        }
        return newCoordinate;
    }

    // TODO use rtree
    // TODO likely should be renamed - this is a search function?
//    public Contig2D lookupCurrentFragmentForOriginalAsmCoordinate(int chrId1, int chrId2, int asmCoordinate) {
//        return lookupCurrentFragmentForOriginalAsmCoordinate(chrId1, chrId2, asmCoordinate);
//    }

    public Contig2D lookupCurrentFragmentForOriginalAsmCoordinate(int chrId1, int chrId2, int asmCoordinate) {

            for (Feature2D feature : contigs.get(chrId1, chrId2)) {
                Contig2D contig = feature.toContig();
                if (contig.iniContains(asmCoordinate)) {
                    return contig;
                }
            }

        return null;
    }

    public int liftOriginalAsmCoordinateToFragmentCoordinate(int chrId1, int chrId2, int asmCoordinate) {
        Contig2D contig = lookupCurrentFragmentForOriginalAsmCoordinate(chrId1, chrId2, asmCoordinate);
        return liftOriginalAsmCoordinateToFragmentCoordinate(contig, asmCoordinate);
    }

    public int liftOriginalAsmCoordinateToFragmentCoordinate(Contig2D contig, int asmCoordinate) {
        if (contig == null) {
            return -1;
        }
        int newCoordinate;
        boolean invertedInitially = contig.getInitialInvert();

        if (invertedInitially) {
            newCoordinate = contig.getInitialEnd() - asmCoordinate + 1;
        } else {
            newCoordinate = asmCoordinate - contig.getInitialStart();
        }
        return newCoordinate;
    }

    //TODO: add scaling, check +/-1
    public int liftFragmentCoordinateToAsmCoordinate(Contig2D contig, int fragmentCoordinate) {
        if (contig == null) {
            return -1;
        }
        boolean invertedInAsm = contig.getAttribute(scaffoldIndexId).contains("-");  //if contains a negative then it is inverted

        int newCoordinate;
        if (invertedInAsm) {
            newCoordinate = contig.getEnd1() - fragmentCoordinate + 1;
        } else {
            newCoordinate = contig.getStart1() + fragmentCoordinate;
        }
        return newCoordinate;
    }

    @Override
    public String toString() {
        String s = "CPROPS\n";
        for (ContigProperty contigProperty : contigProperties) {
            s += contigProperty.toString() + "\n";
        }
        s += "ASM\n";
        for (List<Integer> row : scaffoldProperties) {
            for (int id : row) {
                s += id + " ";
            }
            s += "\n";
        }
        return s;
    }
}