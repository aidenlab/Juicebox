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

import juicebox.HiCGlobals;
import juicebox.mapcolorui.Feature2DHandler;
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

    //have just one attribute

    private List<FragmentProperty> listOfScaffoldProperties;
    private final String FRAG_TXT = FragmentProperty.FRAG_TXT;
    private List<List<Integer>> listOfSuperscaffolds;
    private final String DBRS_TXT = FragmentProperty.DBRS_TXT;
    private Feature2DList superscaffoldFeature2DList;
    private String chromosomeName = "assembly";
    private final String unsignedScaffoldIdAttributeKey = FragmentProperty.unsignedScaffoldIdAttributeKey;
    private final String signedScaffoldIdAttributeKey = FragmentProperty.signedScaffoldIdAttributeKey;
    private final String superScaffoldIdAttributeKey = "Superscaffold #";
    private Map<Feature2D, FragmentProperty> mapFeature2DtoFragProp = new HashMap<>();
    private Feature2DHandler scaffoldFeature2DListHandler;

    public AssemblyFragmentHandler(List<FragmentProperty> listOfScaffoldProperties, List<List<Integer>> listOfSuperscaffolds) {
        this.listOfScaffoldProperties = listOfScaffoldProperties;
        this.listOfSuperscaffolds = listOfSuperscaffolds;
        updateAssembly();

    }

    public void updateAssembly() {
        setCurrentState();
        populate2DFeatures();
    }

    public List<FragmentProperty> cloneScaffoldProperties() {
        List<FragmentProperty> newListOfScaffoldProperties = new ArrayList<>();
        for (FragmentProperty scaffoldProperty : listOfScaffoldProperties) {
            newListOfScaffoldProperties.add(new FragmentProperty(scaffoldProperty));
        }
        return newListOfScaffoldProperties;
    }

    public List<List<Integer>> cloneSuperscaffolds() {
        List<List<Integer>> newListOfSuperScaffolds = new ArrayList<>();
        for (List<Integer> superscaffold : listOfSuperscaffolds) {
            newListOfSuperScaffolds.add(new ArrayList<Integer>(superscaffold));
        }
        return newListOfSuperScaffolds;
    }

    public AssemblyFragmentHandler(AssemblyFragmentHandler assemblyFragmentHandler) {
        this(assemblyFragmentHandler.cloneScaffoldProperties(), assemblyFragmentHandler.cloneSuperscaffolds());
    }

    public Feature2DList getSuperscaffoldFeature2DList() {
        return superscaffoldFeature2DList;
    }

    public List<FragmentProperty> getListOfScaffoldProperties() {
        return listOfScaffoldProperties;
    }

    public List<List<Integer>> getListOfSuperscaffolds() {
        return listOfSuperscaffolds;
    }


    public void setCurrentState() {
        long shift = 0;
        for (List<Integer> superscaffold : listOfSuperscaffolds) {
            for (Integer entry : superscaffold) {
                int i = Math.abs(entry) - 1;
                FragmentProperty currentScaffoldProperty = listOfScaffoldProperties.get(i);
                currentScaffoldProperty.setCurrentStart(shift);
                currentScaffoldProperty.setInvertedVsInitial(false);
                if (entry < 0 && (!listOfScaffoldProperties.get(i).wasInitiallyInverted()) ||
                        entry > 0 && listOfScaffoldProperties.get(i).wasInitiallyInverted()) {
                    currentScaffoldProperty.setInvertedVsInitial(true);
                }
                shift += currentScaffoldProperty.getLength();
            }
        }
    }

    public Feature2DList getScaffoldFeature2DList() {
        return scaffoldFeature2DListHandler.getFeatureList();
    }

    private void populateMapWithFrags() {
        mapFeature2DtoFragProp.clear();
        for (FragmentProperty property : listOfScaffoldProperties) {
            Feature2D feature2D = property.getFeature2D();
            if (feature2D != null) {
                mapFeature2DtoFragProp.put(feature2D, property);
            }
        }
    }

    public void populate2DFeatures() {
        Feature2DList scaffoldFeature2DList = new Feature2DList();
        for (FragmentProperty scaffoldProperty : listOfScaffoldProperties) {
            Contig2D contig = scaffoldProperty.convertToContig2D(chromosomeName);
            scaffoldFeature2DList.add(1, 1, contig);
            scaffoldProperty.setFeature2D(contig);
        }
        populateMapWithFrags();
        scaffoldFeature2DListHandler = new Feature2DHandler(scaffoldFeature2DList);

        superscaffoldFeature2DList = new Feature2DList();
        long superscaffoldStart = 0;
        for (int superscaffoldID = 0; superscaffoldID < listOfSuperscaffolds.size(); superscaffoldID++) {

            long superscaffoldLength = 0;
            for (int scaffold : listOfSuperscaffolds.get(superscaffoldID)) {
                superscaffoldLength += listOfScaffoldProperties.get(Math.abs(scaffold) - 1).getLength();
            }

            Feature2D superscaffoldFeature2D = createSuperScaffoldFeature2D(chromosomeName, superscaffoldID, superscaffoldStart, superscaffoldLength);
            superscaffoldFeature2DList.add(1, 1, superscaffoldFeature2D);
            superscaffoldStart += superscaffoldLength;
        }
    }

    private Feature2D createSuperScaffoldFeature2D(String chromosomeName, int superscaffoldID,
                                                   long superscaffoldStart, long superscaffoldLength) {
        Map<String, String> attributes = new HashMap<String, String>();
        attributes.put(superScaffoldIdAttributeKey, String.valueOf(superscaffoldID));

        Feature2D superscaffoldFeature2D = new Feature2D(Feature2D.FeatureType.SUPERSCAFFOLD,
                chromosomeName,
                (int) Math.round(superscaffoldStart / HiCGlobals.hicMapScale),
                (int) Math.round((superscaffoldStart + superscaffoldLength) / HiCGlobals.hicMapScale),
                chromosomeName,
                (int) Math.round(superscaffoldStart / HiCGlobals.hicMapScale),
                (int) Math.round((superscaffoldStart + superscaffoldLength) / HiCGlobals.hicMapScale),
                Color.BLUE, attributes);
        return superscaffoldFeature2D;
    }

    //**** Split fragment ****//

    public void editScaffold(Feature2D originalFeature, Feature2D debrisFeature) {
        // find the relevant fragment property
        int i = Integer.parseInt(originalFeature.getAttribute(unsignedScaffoldIdAttributeKey)) - 1;
        FragmentProperty toEditFragmentProperty = listOfScaffoldProperties.get(i);

        // do not allow for splitting debris scaffoldFeature2DList
        // TODO should probably also handle at the level of prompts
        if (toEditFragmentProperty.isDebris()) {
            return;
        }

        // calculate split with respect to fragment
        long startCut;
        long endCut;

        if (toEditFragmentProperty.getSignIndexId() > 0) {
            startCut = (long) (debrisFeature.getStart1() * HiCGlobals.hicMapScale) - toEditFragmentProperty.getCurrentStart();
            endCut = (long) (debrisFeature.getEnd1() * HiCGlobals.hicMapScale) - toEditFragmentProperty.getCurrentStart();
        } else {
            startCut = (long) (toEditFragmentProperty.getCurrentEnd() - debrisFeature.getEnd1() * HiCGlobals.hicMapScale);
            endCut = (long) (toEditFragmentProperty.getCurrentEnd() - debrisFeature.getStart1() * HiCGlobals.hicMapScale);
        }

        editCprops(toEditFragmentProperty, startCut, endCut);
        editAsm(toEditFragmentProperty);
    }

    private void editAsm(FragmentProperty toEditFragmentProperty) {
        List<Integer> debrisSuperscaffold = new ArrayList<>();
        for (int i = 0; i < listOfSuperscaffolds.size(); i++) {
            int fragmentId = toEditFragmentProperty.getIndexId();
            for (int j = 0; j < listOfSuperscaffolds.get(i).size(); j++) {
                listOfSuperscaffolds.get(i).set(j, modifyScaffoldId(listOfSuperscaffolds.get(i).get(j), fragmentId));
            }
            if (listOfSuperscaffolds.get(i).contains(fragmentId)) {
                listOfSuperscaffolds.get(i).add(listOfSuperscaffolds.get(i).indexOf(fragmentId) + 1, fragmentId + 2);
                debrisSuperscaffold.add(fragmentId + 1);
            } else if (listOfSuperscaffolds.get(i).contains(-fragmentId)) {
                listOfSuperscaffolds.get(i).add(listOfSuperscaffolds.get(i).indexOf(-fragmentId), -fragmentId - 2);
                debrisSuperscaffold.add(-fragmentId - 1);
            }
        }
        listOfSuperscaffolds.add(debrisSuperscaffold);
    }

    private int modifyScaffoldId(int index, int cutElementId) {
        if (Math.abs(index) <= cutElementId)
            return index;
        else {
            if (index > 0)
                return index + 2;
            else
                return index - 2;
        }
    }

    private void editCprops(FragmentProperty originalScaffoldProperty, long startCut, long endCut) {
        List<FragmentProperty> newScafProps = new ArrayList<>();
        //List<FragmentProperty> addedProperties = new ArrayList<>();
        int startingFragmentNumber;
        for (FragmentProperty scaffoldProperty : listOfScaffoldProperties) {
            if (scaffoldProperty.getIndexId() < originalScaffoldProperty.getIndexId()) {
                newScafProps.add(scaffoldProperty);
            } else if (scaffoldProperty.getIndexId() == originalScaffoldProperty.getIndexId()) {
                startingFragmentNumber = scaffoldProperty.getFragmentNumber();
                if (startingFragmentNumber == 0) {
                    startingFragmentNumber++;
                } // first ever split
                newScafProps.add(new FragmentProperty(scaffoldProperty.getOriginalScaffoldName() + FRAG_TXT + (startingFragmentNumber), scaffoldProperty.getIndexId(), startCut));
                newScafProps.add(new FragmentProperty(scaffoldProperty.getOriginalScaffoldName() + FRAG_TXT + (startingFragmentNumber + 1) + DBRS_TXT, scaffoldProperty.getIndexId() + 1, endCut - startCut));
                newScafProps.add(new FragmentProperty(scaffoldProperty.getOriginalScaffoldName() + FRAG_TXT + (startingFragmentNumber + 2), scaffoldProperty.getIndexId() + 2, scaffoldProperty.getLength() - endCut));
                // set their initial properties
                int newListSize = newScafProps.size();
                if (!originalScaffoldProperty.wasInitiallyInverted()) {
                    newScafProps.get(newListSize - 3).setInitialStart(originalScaffoldProperty.getInitialStart());
                    newScafProps.get(newListSize - 3).setInitiallyInverted(false);
                    newScafProps.get(newListSize - 2).setInitialStart(originalScaffoldProperty.getInitialStart() + startCut);
                    newScafProps.get(newListSize - 2).setInitiallyInverted(false);
                    newScafProps.get(newListSize - 1).setInitialStart(originalScaffoldProperty.getInitialStart() + endCut);
                    newScafProps.get(newListSize - 1).setInitiallyInverted(false);
                } else {
                    newScafProps.get(newListSize - 1).setInitialStart(originalScaffoldProperty.getInitialStart());
                    newScafProps.get(newListSize - 1).setInitiallyInverted(true);
                    newScafProps.get(newListSize - 2).setInitialStart(originalScaffoldProperty.getInitialEnd() - endCut);
                    newScafProps.get(newListSize - 2).setInitiallyInverted(true);
                    newScafProps.get(newListSize - 3).setInitialStart(originalScaffoldProperty.getInitialEnd() - startCut);
                    newScafProps.get(newListSize - 3).setInitiallyInverted(true);
                }
            } else {
                if (scaffoldProperty.getOriginalScaffoldName().equals(originalScaffoldProperty.getOriginalScaffoldName())) {
                    if (scaffoldProperty.isDebris())
                        scaffoldProperty.setName(scaffoldProperty.getOriginalScaffoldName() + FRAG_TXT + (scaffoldProperty.getFragmentNumber() + 2) + DBRS_TXT);
                    else
                        scaffoldProperty.setName(scaffoldProperty.getOriginalScaffoldName() + FRAG_TXT + (scaffoldProperty.getFragmentNumber() + 2));
                }
                scaffoldProperty.setIndexId(scaffoldProperty.getIndexId() + 2);
                newScafProps.add(scaffoldProperty);
            }
        }

        listOfScaffoldProperties.clear();
        listOfScaffoldProperties.addAll(newScafProps);
        populateMapWithFrags();
    }



    //**** Inversion ****//

    public void invertSelection(List<Feature2D> scaffolds) {
        int id1 = Integer.parseInt(scaffolds.get(0).getAttribute(signedScaffoldIdAttributeKey));
        int id2 = Integer.parseInt(scaffolds.get(scaffolds.size() - 1).getAttribute(signedScaffoldIdAttributeKey));
        int gid1 = getSuperscaffoldId(id1);
        int gid2 = getSuperscaffoldId(id2);

        if (gid1 != gid2 && listOfSuperscaffolds.get(gid1).indexOf(id1) != 0) {
            splitSuperscaffold(gid1, listOfSuperscaffolds.get(gid1).get(listOfSuperscaffolds.get(gid1).indexOf(id1) - 1));
            gid1 = getSuperscaffoldId(id1);
            gid2 = getSuperscaffoldId(id2);
        }
        if (gid1 != gid2 && listOfSuperscaffolds.get(gid2).indexOf(id2) != listOfSuperscaffolds.get(gid2).size() - 1) {
            splitSuperscaffold(gid2, id2);
            gid1 = getSuperscaffoldId(id1);
            gid2 = getSuperscaffoldId(id2);
        }

        //update scaffold properties
        List<FragmentProperty> selectedScaffoldProperties = scaffold2DListToListOfScaffoldProperties(scaffolds);
        for (FragmentProperty scaffoldProperty : selectedScaffoldProperties) {
            scaffoldProperty.toggleInversion();
        }

        if (gid1==gid2){
            Collections.reverse(listOfSuperscaffolds.get(gid1).subList(listOfSuperscaffolds.get(gid1).indexOf(id1), listOfSuperscaffolds.get(gid2).indexOf(id2) + 1));
            for (int i = listOfSuperscaffolds.get(gid1).indexOf(id2); i <= listOfSuperscaffolds.get(gid2).indexOf(id1); i++) {
                listOfSuperscaffolds.get(gid1).set(i, -1 * listOfSuperscaffolds.get(gid1).get(i));
            }
        } else {
            List<List<Integer>> newListOfSuperscaffolds = new ArrayList<>();
            for (int i = 0; i <= listOfSuperscaffolds.size() - 1; i++) {
                if(i>=gid1&&i<=gid2){
                    newListOfSuperscaffolds.add(listOfSuperscaffolds.get(gid2 - i + gid1));
                    Collections.reverse(newListOfSuperscaffolds.get(i));
                    for (int j = 0; j <= newListOfSuperscaffolds.get(i).size() - 1; j++) {
                        newListOfSuperscaffolds.get(i).set(j, -1 * newListOfSuperscaffolds.get(i).get(j));
                    }
                } else{
                    newListOfSuperscaffolds.add(listOfSuperscaffolds.get(i));
                }
            }
            listOfSuperscaffolds.clear();
            listOfSuperscaffolds.addAll(newListOfSuperscaffolds);
        }
    }

    public List<FragmentProperty> scaffold2DListToListOfScaffoldProperties(List<Feature2D> scaffoldFeature2DList) {
        List<FragmentProperty> fragmentProperties = new ArrayList<>();
        for (Feature2D feature2D : scaffoldFeature2DList) {
            if (mapFeature2DtoFragProp.containsKey(feature2D)) {
                fragmentProperties.add(mapFeature2DtoFragProp.get(feature2D));
            }
        }
        return fragmentProperties;
    }

    //**** Move selection ****//

    public void moveSelection(List<Feature2D> selectedFeatures, Feature2D upstreamFeature){
        int id1 = Integer.parseInt(selectedFeatures.get(0).getAttribute(signedScaffoldIdAttributeKey));
        int id2 = Integer.parseInt(selectedFeatures.get(selectedFeatures.size() - 1).getAttribute(signedScaffoldIdAttributeKey));
        int id3 = Integer.parseInt(upstreamFeature.getAttribute(signedScaffoldIdAttributeKey));
        moveSelection(id1, id2, id3);
    }

    //**** Move selection ****//

    public void moveSelection(int id1, int id2, int id3) {

        int gid1 = getSuperscaffoldId(id1);
        int gid2 = getSuperscaffoldId(id2);
        int gid3 = getSuperscaffoldId(id3);

        // check if selectedFeatures span multiple groups paste split at destination
        if (gid1 != gid2 & listOfSuperscaffolds.get(gid3).indexOf(id3) != listOfSuperscaffolds.get(gid3).size() - 1) {
            splitSuperscaffold(gid3, id3);
            gid1 = getSuperscaffoldId(id1);
            gid2 = getSuperscaffoldId(id2);
            gid3 = getSuperscaffoldId(id3);
        }

        List<List<Integer>> newSuperscaffolds = new ArrayList<>();
        List<List<Integer>> tempSuperscaffolds = new ArrayList<>();
        List<Integer> truncatedSuperscaffold = new ArrayList<Integer>();
        int shiftSuperscaffold = 0;

        for (int i = 0; i <= listOfSuperscaffolds.size() - 1; i++) {
            if (i==gid1 && i==gid2){

                tempSuperscaffolds.add(listOfSuperscaffolds.get(gid1).subList(listOfSuperscaffolds.get(gid1).indexOf(id1), listOfSuperscaffolds.get(gid2).indexOf(id2) + 1));

                if (listOfSuperscaffolds.get(gid1).indexOf(id1) != 0) {
                    truncatedSuperscaffold.addAll(listOfSuperscaffolds.get(gid1).subList(0, listOfSuperscaffolds.get(gid1).indexOf(id1)));
                }
                if (listOfSuperscaffolds.get(gid2).indexOf(id2) != listOfSuperscaffolds.get(gid2).size() - 1) {
                    truncatedSuperscaffold.addAll(listOfSuperscaffolds.get(gid2).subList(1 + listOfSuperscaffolds.get(gid2).indexOf(id2), listOfSuperscaffolds.get(gid2).size()));
                }

                if (!truncatedSuperscaffold.isEmpty()) {
                    newSuperscaffolds.add(truncatedSuperscaffold);
                } else {
                    shiftSuperscaffold++;
                }

            } else if (gid1!=gid2 && i==gid1){
                tempSuperscaffolds.add(listOfSuperscaffolds.get(gid1).subList(listOfSuperscaffolds.get(gid1).indexOf(id1), listOfSuperscaffolds.get(gid1).size()));
                if (listOfSuperscaffolds.get(gid1).indexOf(id1) != 0) {
                    newSuperscaffolds.add(listOfSuperscaffolds.get(gid1).subList(0, listOfSuperscaffolds.get(gid1).indexOf(id1)));
                }else{
                    shiftSuperscaffold++;
                }
            } else if (gid1!=gid2 && i > gid1 && i < gid2){
                tempSuperscaffolds.add(listOfSuperscaffolds.get(i));
                shiftSuperscaffold++;
            } else if (gid1!=gid2 && i==gid2){
                tempSuperscaffolds.add(listOfSuperscaffolds.get(gid2).subList(0, 1 + listOfSuperscaffolds.get(gid2).indexOf(id2)));
                if (listOfSuperscaffolds.get(gid2).indexOf(id2) != listOfSuperscaffolds.get(gid2).size() - 1) {
                    newSuperscaffolds.add(listOfSuperscaffolds.get(gid2).subList(1 + listOfSuperscaffolds.get(gid2).indexOf(id2), listOfSuperscaffolds.get(gid2).size()));
                }else{
                    shiftSuperscaffold++;
                }
            } else {
                newSuperscaffolds.add(listOfSuperscaffolds.get(i));
            }
        }

        int newgid3=gid3;
        if (gid3 > gid2){
            newgid3 -= shiftSuperscaffold;
        }

        if (listOfSuperscaffolds.get(gid3).indexOf(id3) == listOfSuperscaffolds.get(gid3).size() - 1) {
            newSuperscaffolds.addAll(newgid3 + 1, tempSuperscaffolds);
        } else {
            int pasteIndex = listOfSuperscaffolds.get(gid3).indexOf(id3);
            if (gid1 == gid3 && gid2 == gid3 && listOfSuperscaffolds.get(gid3).indexOf(id3) > listOfSuperscaffolds.get(gid3).indexOf(id2)) {
                pasteIndex -= listOfSuperscaffolds.get(gid3).size() - truncatedSuperscaffold.size();
            }
            newSuperscaffolds.get(newgid3).addAll(pasteIndex + 1, tempSuperscaffolds.get(0));
        }

        listOfSuperscaffolds.clear();
        listOfSuperscaffolds.addAll(newSuperscaffolds);
    }

    //**** Group toggle ****//

    public void toggleGroup(Feature2D upstreamFeature2D, Feature2D downstreamFeature2D) {
        int id1 = Integer.parseInt(upstreamFeature2D.getAttribute(signedScaffoldIdAttributeKey));
        int id2 = Integer.parseInt(downstreamFeature2D.getAttribute(signedScaffoldIdAttributeKey));

        //should not happen, other sanity checks?
        if (id1==id2){
            return;
        }

        int super1 = getSuperscaffoldId(id1);
        int super2 = getSuperscaffoldId(id2);

        if (super1 == super2) {
            splitSuperscaffold(super1, id1);
        }else {
            mergeSuperscaffolds(super1, super2);
        }

    }


    private void mergeSuperscaffolds(int superscaffoldId1, int superscaffoldId2) {
        listOfSuperscaffolds.get(superscaffoldId1).addAll(listOfSuperscaffolds.get(superscaffoldId2));
        listOfSuperscaffolds.remove(superscaffoldId2);
    }

    private void splitSuperscaffold(int superscaffoldId, int scaffoldId) {
        List<Integer> extractedList = listOfSuperscaffolds.get(superscaffoldId);
        int splitIndex = 1 + extractedList.indexOf(scaffoldId);

        List<Integer> split1 = extractedList.subList(0, splitIndex);
        List<Integer> split2 = extractedList.subList(splitIndex, extractedList.size());

        // remove old entry and add the two new ones
        listOfSuperscaffolds.remove(superscaffoldId);
        listOfSuperscaffolds.add(superscaffoldId, split1);
        listOfSuperscaffolds.add(superscaffoldId + 1, split2);
    }

    //**** Utility functions ****//
    private int getSuperscaffoldId(int scaffoldId) {
        int i = 0;
        for (List<Integer> scaffoldRow : listOfSuperscaffolds) {
            if (scaffoldRow.contains(scaffoldId) || scaffoldRow.contains(-scaffoldId)) {
                return i;
            }
            i++;
        }
        System.err.println("Cannot find superscaffold containing scaffold " + scaffoldId);
        return -1;
    }

    //**** For debugging ****//
    public void printAssembly(){
        System.out.println(Arrays.toString(listOfSuperscaffolds.toArray()));
    }

    public Contig2D lookupContigForBinValue(int chr1Idx, int chr2Idx, int genomicCoordinate, int binSize) {

        int g1 = genomicCoordinate - binSize;
        int g2 = genomicCoordinate + binSize;

        net.sf.jsi.Rectangle currentWindow = new net.sf.jsi.Rectangle(g1, g1, g2, g2);
        List<Feature2D> foundContigs = scaffoldFeature2DListHandler.getIntersectingFeatures(chr1Idx, chr2Idx, currentWindow, true);
        // todo fix delete List<Feature2D> foundContigs = scaffoldFeature2DListHandler.getFeatureList().get(1,1);
        System.err.println("csize = " + foundContigs.size() + " - " + currentWindow);
        for (Feature2D feature : foundContigs) {
            Contig2D contig = feature.toContig();
            if (contig.iniContains(genomicCoordinate)) {
                //System.err.println("fine so far...1");
                return contig;
            }
        }
        //System.err.println("contig is null...1?");
        return null;
    }

    public int liftOriginalAsmCoordinateToFragmentCoordinate(Contig2D contig, int asmCoordinate) {
        if (contig.getInitialInvert()) {
            return contig.getInitialEnd() - asmCoordinate + 1;
        } else {
            return asmCoordinate - contig.getInitialStart();
        }
    }

    //TODO: add scaling, check +/-1
    public int liftFragmentCoordinateToAsmCoordinate(Contig2D contig, int fragmentCoordinate) {
        if (contig.containsAttributeKey(signedScaffoldIdAttributeKey)) {
            boolean invertedInAsm = contig.getAttribute(signedScaffoldIdAttributeKey).contains("-");  //if contains a negative then it is inverted

            if (invertedInAsm) {
                return contig.getEnd1() - fragmentCoordinate + 1;
            } else {
                return contig.getStart1() + fragmentCoordinate;
            }
        }
        return -1;
    }

    @Override
    public String toString() {
        String s = "CPROPS\n";
        for (FragmentProperty scaffoldProperty : listOfScaffoldProperties) {
            s += scaffoldProperty + "\n";
        }
        s += "ASM\n";
        for (List<Integer> superscaffold : listOfSuperscaffolds) {
            for (int id : superscaffold) {
                s += id + " ";
            }
            s += "\n";
        }
        return s;
    }
}