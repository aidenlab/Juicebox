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
    private final String unsignedScaffoldIdAttributeKey = "Scaffold #";
    private final String signedScaffoldIdAttributeKey = "Signed scaffold #";
    private final String scaffoldNameAttributeKey = "Scaffold name";
    private final String superScaffoldIdAttributeKey = "Superscaffold #";


    private List<FragmentProperty> listOfScaffoldProperties;
    private List<List<Integer>> listOfSuperscaffolds;
    private Feature2DList scaffoldFeature2DList;
    private Feature2DList superscaffoldFeature2DList;
    private String chromosomeName = "assembly";

    private List<FragmentProperty> listOfAggregateScaffoldProperties = new ArrayList<>();
    private Feature2DList aggregateScaffoldFeature2DList;
    private Feature2DHandler aggregateFeature2DHandler;


    public AssemblyFragmentHandler(List<FragmentProperty> listOfScaffoldProperties, List<List<Integer>> listOfSuperscaffolds) {
        this.listOfScaffoldProperties = listOfScaffoldProperties;
        this.listOfSuperscaffolds = listOfSuperscaffolds;
        updateAssembly();
    }

    public AssemblyFragmentHandler(AssemblyFragmentHandler assemblyFragmentHandler) {
        this.listOfScaffoldProperties = assemblyFragmentHandler.cloneScaffoldProperties();
        this.listOfSuperscaffolds = assemblyFragmentHandler.cloneSuperscaffolds();
        updateAssembly();
    }

    public void updateAssembly() {
        setCurrentState();
        populate2DFeatures();
        aggregateScaffolds();
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

    public Feature2DList getScaffoldFeature2DList() {
        return scaffoldFeature2DList;
    }

    public void setScaffoldFeature2DList(Feature2DList scaffoldFeature2DList) {
        this.scaffoldFeature2DList = scaffoldFeature2DList;
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

    public void populate2DFeatures() {
        scaffoldFeature2DList = new Feature2DList();
        for (FragmentProperty scaffoldProperty : listOfScaffoldProperties) {

            Map<String, String> attributes = new HashMap<String, String>();

            attributes.put(scaffoldNameAttributeKey, scaffoldProperty.getName());
            attributes.put(signedScaffoldIdAttributeKey, String.valueOf(scaffoldProperty.getSignIndexId()));
            attributes.put(unsignedScaffoldIdAttributeKey, String.valueOf(scaffoldProperty.getIndexId()));
            //attributes.put(initiallyInvertedStatus, Boolean.toString(scaffoldProperty.wasInitiallyInverted()));

            Feature2D scaffoldFeature2D = new Feature2D(Feature2D.FeatureType.SCAFFOLD,
                    chromosomeName,
                    (int) Math.round(scaffoldProperty.getCurrentStart() / HiCGlobals.hicMapScale),
                    (int) Math.round((scaffoldProperty.getCurrentEnd()) / HiCGlobals.hicMapScale),
                    chromosomeName,
                    (int) Math.round(scaffoldProperty.getCurrentStart() / HiCGlobals.hicMapScale),
                    (int) Math.round((scaffoldProperty.getCurrentEnd()) / HiCGlobals.hicMapScale),
                    new Color(0, 255, 0),
                    attributes);
//TODO: get rid of Contig2D, too much confusion, too much overlap
            Contig2D contig = scaffoldFeature2D.toContig();
            if (scaffoldProperty.isInvertedVsInitial()) {
                contig.toggleInversion(); //assuming initial contig2D inverted = false
            }
            contig.setInitialState(scaffoldProperty.getInitialChr(),
                    (int) Math.round(scaffoldProperty.getInitialStart() / HiCGlobals.hicMapScale),
                    (int) Math.round(scaffoldProperty.getInitialEnd() / HiCGlobals.hicMapScale),
                    scaffoldProperty.wasInitiallyInverted());
            scaffoldFeature2DList.add(1, 1, contig);
            scaffoldProperty.setFeature2D(contig);
        }
        superscaffoldFeature2DList = new Feature2DList();
        long superscaffoldStart = 0;
        for (int superscaffold = 0; superscaffold < listOfSuperscaffolds.size(); superscaffold++) {
            Map<String, String> attributes = new HashMap<String, String>();
            attributes.put(superScaffoldIdAttributeKey, String.valueOf(superscaffold));
            long superscaffoldLength = 0;
            for (int scaffold : listOfSuperscaffolds.get(superscaffold)) {
                superscaffoldLength += listOfScaffoldProperties.get(Math.abs(scaffold) - 1).getLength();
            }
            Feature2D superscaffoldFeature2D = new Feature2D(Feature2D.FeatureType.SUPERSCAFFOLD,
                    chromosomeName,
                    (int) Math.round(superscaffoldStart / HiCGlobals.hicMapScale),
                    (int) Math.round((superscaffoldStart + superscaffoldLength) / HiCGlobals.hicMapScale),
                    chromosomeName,
                    (int) Math.round(superscaffoldStart / HiCGlobals.hicMapScale),
                    (int) Math.round((superscaffoldStart + superscaffoldLength) / HiCGlobals.hicMapScale),
                    new Color(0, 0, 255),
                    attributes);
            superscaffoldFeature2DList.add(1, 1, superscaffoldFeature2D);
            superscaffoldStart += superscaffoldLength;
        }
    }

    public void aggregateScaffolds() {
        listOfAggregateScaffoldProperties.clear();
        aggregateScaffoldFeature2DList = new Feature2DList();
        int counter = 1;
        FragmentProperty aggregateScaffoldProperty = new FragmentProperty(listOfScaffoldProperties.get(Math.abs(listOfSuperscaffolds.get(0).get(0)) - 1));
        aggregateScaffoldProperty.setInitiallyInverted(false);

        for (int i = 0; i < listOfSuperscaffolds.size(); i++) {
            for (int j = 0; j < listOfSuperscaffolds.get(i).size(); j++) {

                if (i == 0 && j == 0) {
                    continue;
                }

                FragmentProperty nextScaffoldProperty = listOfScaffoldProperties.get(Math.abs(listOfSuperscaffolds.get(i).get(j)) - 1);
                FragmentProperty temp = aggregateScaffoldProperty.merge(nextScaffoldProperty);
                if (temp == null) {
                    listOfAggregateScaffoldProperties.add(aggregateScaffoldProperty);
                    aggregateScaffoldProperty.getFeature2D().setAttribute(scaffoldNameAttributeKey, String.valueOf(counter));
                    aggregateScaffoldProperty.setInitiallyInverted(false);
                    aggregateScaffoldFeature2DList.add(1, 1, aggregateScaffoldProperty.getFeature2D());
                    counter++;
                    aggregateScaffoldProperty = new FragmentProperty(nextScaffoldProperty);
                } else {
                    aggregateScaffoldProperty = temp;
                }
            }
        }
        listOfAggregateScaffoldProperties.add(aggregateScaffoldProperty);
        aggregateScaffoldProperty.setInitiallyInverted(false);
        aggregateScaffoldProperty.getFeature2D().setAttribute(scaffoldNameAttributeKey, String.valueOf(counter));
        aggregateScaffoldFeature2DList.add(1, 1, aggregateScaffoldProperty.getFeature2D());

        aggregateFeature2DHandler = new Feature2DHandler();
        aggregateFeature2DHandler.loadLoopList(aggregateScaffoldFeature2DList, true);

//        System.out.println("how many aggregates: "+listOfAggregateScaffoldProperties.size());

    }

    //**** Split fragment ****//

    public void editScaffold(Feature2D originalFeature, Feature2D debrisFeature) {
        // find the relevant fragment property
        int i = Integer.parseInt(originalFeature.getAttribute(unsignedScaffoldIdAttributeKey)) - 1;
        FragmentProperty toEditFragmentProperty = listOfScaffoldProperties.get(i);

        // do not allow for splitting debris scaffoldFeature2DList, TODO: should probably also handle at the level of prompts
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
        List<FragmentProperty> newListOfScaffoldProperties = new ArrayList<>();
        //List<FragmentProperty> addedProperties = new ArrayList<>();
        int startingFragmentNumber;
        for (FragmentProperty scaffoldProperty : listOfScaffoldProperties) {
            if (scaffoldProperty.getIndexId() < originalScaffoldProperty.getIndexId()) {
                newListOfScaffoldProperties.add(scaffoldProperty);
            } else if (scaffoldProperty.getIndexId() == originalScaffoldProperty.getIndexId()) {
                startingFragmentNumber = scaffoldProperty.getFragmentNumber();
                if (startingFragmentNumber == 0) {
                    startingFragmentNumber++;
                } // first ever split
                newListOfScaffoldProperties.add(new FragmentProperty(scaffoldProperty.getOriginalScaffoldName() + ":::fragment_" + (startingFragmentNumber), scaffoldProperty.getIndexId(), startCut));
                newListOfScaffoldProperties.add(new FragmentProperty(scaffoldProperty.getOriginalScaffoldName() + ":::fragment_" + (startingFragmentNumber + 1) + ":::debris", scaffoldProperty.getIndexId() + 1, endCut - startCut));
                newListOfScaffoldProperties.add(new FragmentProperty(scaffoldProperty.getOriginalScaffoldName() + ":::fragment_" + (startingFragmentNumber + 2), scaffoldProperty.getIndexId() + 2, scaffoldProperty.getLength() - endCut));
                // set their initial properties
                if (!originalScaffoldProperty.wasInitiallyInverted()) {
                    newListOfScaffoldProperties.get(newListOfScaffoldProperties.size() - 3).setInitialStart(originalScaffoldProperty.getInitialStart());
                    newListOfScaffoldProperties.get(newListOfScaffoldProperties.size() - 3).setInitiallyInverted(false);
                    newListOfScaffoldProperties.get(newListOfScaffoldProperties.size() - 2).setInitialStart(originalScaffoldProperty.getInitialStart() + startCut);
                    newListOfScaffoldProperties.get(newListOfScaffoldProperties.size() - 2).setInitiallyInverted(false);
                    newListOfScaffoldProperties.get(newListOfScaffoldProperties.size() - 1).setInitialStart(originalScaffoldProperty.getInitialStart() + endCut);
                    newListOfScaffoldProperties.get(newListOfScaffoldProperties.size() - 1).setInitiallyInverted(false);
                } else {
                    newListOfScaffoldProperties.get(newListOfScaffoldProperties.size() - 1).setInitialStart(originalScaffoldProperty.getInitialStart());
                    newListOfScaffoldProperties.get(newListOfScaffoldProperties.size() - 1).setInitiallyInverted(true);
                    newListOfScaffoldProperties.get(newListOfScaffoldProperties.size() - 2).setInitialStart(originalScaffoldProperty.getInitialEnd() - endCut);
                    newListOfScaffoldProperties.get(newListOfScaffoldProperties.size() - 2).setInitiallyInverted(true);
                    newListOfScaffoldProperties.get(newListOfScaffoldProperties.size() - 3).setInitialStart(originalScaffoldProperty.getInitialEnd() - startCut);
                    newListOfScaffoldProperties.get(newListOfScaffoldProperties.size() - 3).setInitiallyInverted(true);
                }
            } else {
                if (scaffoldProperty.getOriginalScaffoldName().equals(originalScaffoldProperty.getOriginalScaffoldName())) {
                    if (scaffoldProperty.isDebris())
                        scaffoldProperty.setName(scaffoldProperty.getOriginalScaffoldName() + ":::fragment_" + (scaffoldProperty.getFragmentNumber() + 2) + ":::debris");
                    else
                        scaffoldProperty.setName(scaffoldProperty.getOriginalScaffoldName() + ":::fragment_" + (scaffoldProperty.getFragmentNumber() + 2));
                }
                scaffoldProperty.setIndexId(scaffoldProperty.getIndexId() + 2);
                newListOfScaffoldProperties.add(scaffoldProperty);
            }
        }

        listOfScaffoldProperties.clear();
        listOfScaffoldProperties.addAll(newListOfScaffoldProperties);
    }



    //**** Inversion ****//

    public void invertSelection(List<Feature2D> scaffolds) {

        List<Integer> scaffoldIds = scaffold2DListToIntegerList(scaffolds);

        int id1 = scaffoldIds.get(0);
        int id2 = scaffoldIds.get(scaffoldIds.size() - 1);
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

        return;

    }

    public List<Integer> scaffold2DListToIntegerList(List<Feature2D> scaffoldFeature2DList) {
        List<Integer> scaffoldIds = new ArrayList<Integer>();
        for (Feature2D feature2D : scaffoldFeature2DList) {
            scaffoldIds.add(Integer.parseInt(feature2D.getAttribute(signedScaffoldIdAttributeKey)));
            scaffoldIds.add(Integer.parseInt(feature2D.getAttribute(signedScaffoldIdAttributeKey)));
        }
        return scaffoldIds;
    }

    public List<FragmentProperty> scaffold2DListToListOfScaffoldProperties(List<Feature2D> scaffoldFeature2DList) {
        List<FragmentProperty> listOfScaffoldProperties = new ArrayList<FragmentProperty>();
        for (Feature2D feature2D : scaffoldFeature2DList) {
            listOfScaffoldProperties.add(feature2DToScaffoldProperty(feature2D));
        }
        return listOfScaffoldProperties;
    }

    public FragmentProperty feature2DToScaffoldProperty(Feature2D feature2D) {
        // TODO: redo via attribute
        for (FragmentProperty fragmentProperty : listOfScaffoldProperties) {
            if (fragmentProperty.getFeature2D().equals(feature2D)) { //make sure it is okay

                return fragmentProperty;
            }
        }
        return null;
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
        List<List<Integer>> newSuperscaffolds = new ArrayList<>();
        for (int i = 0; i <= listOfSuperscaffolds.size() - 1; i++) {
            if (i == superscaffoldId2) {
                newSuperscaffolds.get(superscaffoldId1).addAll(listOfSuperscaffolds.get(superscaffoldId2));
            } else {
                newSuperscaffolds.add(listOfSuperscaffolds.get(i));
            }
        }
        listOfSuperscaffolds.clear();
        listOfSuperscaffolds.addAll(newSuperscaffolds);
        return;
    }

    private void splitSuperscaffold(int superscaffoldId, int scaffoldId) {
        List<List<Integer>> newSuperscaffolds = new ArrayList<>();
        for (int i = 0; i <= listOfSuperscaffolds.size() - 1; i++) {
            if (i == superscaffoldId) {
                newSuperscaffolds.add(listOfSuperscaffolds.get(superscaffoldId).subList(0, 1 + listOfSuperscaffolds.get(superscaffoldId).indexOf(scaffoldId)));
                newSuperscaffolds.add(listOfSuperscaffolds.get(superscaffoldId).subList(1 + listOfSuperscaffolds.get(superscaffoldId).indexOf(scaffoldId), listOfSuperscaffolds.get(superscaffoldId).size()));
            } else {
                newSuperscaffolds.add(listOfSuperscaffolds.get(i));
            }
        }
        listOfSuperscaffolds.clear();
        listOfSuperscaffolds.addAll(newSuperscaffolds);
        return;
    }

    //**** Utility functions ****//

    private int getSuperscaffoldId(int scaffoldId) {
        int i = 0;
        for (List<Integer> scaffoldRow : listOfSuperscaffolds) {

            for (int index : scaffoldRow) {
                if (Math.abs(index) == Math.abs(scaffoldId))
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
        return;
    }

    public Contig2D liftAsmCoordinateToFragment(int chrId1, int chrId2, int asmCoordinate) {

        for (Feature2D contig : scaffoldFeature2DList.get(chrId1, chrId2)) {
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
        boolean inverted = contig.getAttribute(signedScaffoldIdAttributeKey).contains("-");
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

        for (Feature2D feature : scaffoldFeature2DList.get(chrId1, chrId2)) {
                Contig2D contig = feature.toContig();
                if (contig.iniContains(asmCoordinate)) {
                    return contig;
                }
            }

        return null;
    }

    public FragmentProperty newLookupCurrentFragmentForOriginalAsmCoordinate(int chrId1, int chrId2, long asmCoordinate) {

        for (FragmentProperty aggregateFragmentProperty : listOfAggregateScaffoldProperties) {
            if (asmCoordinate > aggregateFragmentProperty.getInitialStart()
                    && asmCoordinate <= aggregateFragmentProperty.getInitialEnd()) {
                return aggregateFragmentProperty;
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

    public long newLiftOriginalAsmCoordinateToFragmentCoordinate(FragmentProperty fragmentProperty, long asmCoordinate) {
        if (fragmentProperty == null) {
            return -1;
        }
        long newCoordinate;
        boolean invertedInitially = fragmentProperty.wasInitiallyInverted();

        if (invertedInitially) {
            newCoordinate = fragmentProperty.getInitialEnd() - asmCoordinate + 1;
        } else {
            newCoordinate = asmCoordinate - fragmentProperty.getInitialStart();
        }
        return newCoordinate;
    }

    //TODO: add scaling, check +/-1
    public int liftFragmentCoordinateToAsmCoordinate(Contig2D contig, int fragmentCoordinate) {
        if (contig == null) {
            return -1;
        }
        boolean invertedInAsm = contig.getAttribute(signedScaffoldIdAttributeKey).contains("-");  //if contains a negative then it is inverted

        int newCoordinate;
        if (invertedInAsm) {
            newCoordinate = contig.getEnd1() - fragmentCoordinate + 1;
        } else {
            newCoordinate = contig.getStart1() + fragmentCoordinate;
        }
        return newCoordinate;
    }

    public long newLiftFragmentCoordinateToAsmCoordinate(FragmentProperty fragmentProperty, long fragmentCoordinate) {
        if (fragmentProperty == null) {
            return -1;
        }
        boolean invertedInAsm = (fragmentProperty.getSignIndexId() < 0);  //if contains a negative then it is inverted

        long newCoordinate;
        if (invertedInAsm) {
            newCoordinate = fragmentProperty.getCurrentEnd() - fragmentCoordinate + 1;
        } else {
            newCoordinate = fragmentProperty.getCurrentStart() + fragmentCoordinate;
        }
        return newCoordinate;
    }

    @Override
    public String toString() {
        String s = "CPROPS\n";
        for (FragmentProperty scaffoldProperty : listOfScaffoldProperties) {
            s += scaffoldProperty.toString() + "\n";
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

    public Feature2DList getAggregateFeature2DList() {
        return aggregateScaffoldFeature2DList;
    }

    public Feature2DHandler getAggregateFeature2DHandler() {
        return aggregateFeature2DHandler;
    }
}