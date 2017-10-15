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
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;

import java.awt.*;
import java.util.*;
import java.util.List;

/**
 * Created by nathanielmusial on 6/30/17.
 */
public class AssemblyScaffoldHandler {

    //constants, have just one attribute
    private final String unsignedScaffoldIdAttributeKey = "Scaffold #";
    private final String signedScaffoldIdAttributeKey = "Signed scaffold #";
    private final String scaffoldNameAttributeKey = "Scaffold name";
    private final String superScaffoldIdAttributeKey = "Superscaffold #";

    //scaffolds and superscaffolds
    private List<Scaffold> listOfScaffolds = new ArrayList<>();
    private List<List<Integer>> listOfSuperscaffolds = new ArrayList<>();

    // aggregates
    private List<Scaffold> listOfAggregateScaffolds = new ArrayList<>();

    private Feature2DHandler scaffoldFeature2DHandler;
    private Feature2DHandler superscaffoldFeature2DHandler;

    // should change this to BinarySearchTree
    private Feature2DHandler originalAggregateFeature2DHandler;
    private Feature2DHandler currentAggregateFeature2DHandler;

    // formalities
    private int chrIndex = 1;
    private String chrName = "assembly";


    public AssemblyScaffoldHandler(List<Scaffold> listOfScaffolds, List<List<Integer>> listOfSuperscaffolds) {
        this.listOfScaffolds = listOfScaffolds;
        this.listOfSuperscaffolds = listOfSuperscaffolds;
        updateAssembly(true);
    }

    // do we still need this constructor?
    public AssemblyScaffoldHandler(AssemblyScaffoldHandler assemblyScaffoldHandler) {
        this.listOfScaffolds = assemblyScaffoldHandler.cloneScaffolds();
        this.listOfSuperscaffolds = assemblyScaffoldHandler.cloneSuperscaffolds();
        //updateAssembly(true);
    }

    public List<Scaffold> cloneScaffolds() {
        List<Scaffold> newListOfScaffolds = new ArrayList<>();
        for (Scaffold scaffold : listOfScaffolds) {
            newListOfScaffolds.add(new Scaffold(scaffold));
        }
        return newListOfScaffolds;
    }

    public List<List<Integer>> cloneSuperscaffolds() {
        List<List<Integer>> newListOfSuperScaffolds = new ArrayList<>();
        for (List<Integer> superscaffold : listOfSuperscaffolds) {
            newListOfSuperScaffolds.add(new ArrayList<Integer>(superscaffold));
        }
        return newListOfSuperScaffolds;
    }

    public List<Scaffold> getListOfScaffolds() {
        return listOfScaffolds;
    }

    public List<List<Integer>> getListOfSuperscaffolds() {
        return listOfSuperscaffolds;
    }

    public void updateAssembly(boolean refreshMap) {

        if (!refreshMap) {
            updateSuperscaffolds();
            return;
        }

        int signScafId;
        long scaffoldShift = 0;
        long superscaffoldShift = 0;
        int aggregateScaffoldCounter = 1;


        Scaffold nextScaffold = null;
        Scaffold aggregateScaffold = null;
        Scaffold tempAggregateScaffold = null;

        Feature2DList scaffoldFeature2DList = new Feature2DList();
        Feature2DList superscaffoldFeature2DList = new Feature2DList();

        listOfAggregateScaffolds.clear();

        for (int i = 0; i < listOfSuperscaffolds.size(); i++) {
            for (int j = 0; j < listOfSuperscaffolds.get(i).size(); j++) {

                if (i == 0 && j == 0) {

                    // first scaffold
                    signScafId = listOfSuperscaffolds.get(i).get(j);
                    nextScaffold = listOfScaffolds.get(Math.abs(signScafId) - 1);
                    nextScaffold.setCurrentStart(scaffoldShift);
                    nextScaffold.setInvertedVsInitial(false);
                    if (signScafId < 0 && (!listOfScaffolds.get(Math.abs(signScafId) - 1).getOriginallyInverted()) ||
                            signScafId > 0 && listOfScaffolds.get(Math.abs(signScafId) - 1).getOriginallyInverted()) {
                        nextScaffold.setInvertedVsInitial(true);
                    }
                    scaffoldFeature2DList.add(chrIndex, chrIndex, nextScaffold.getCurrentFeature2D());
                    scaffoldShift += nextScaffold.getLength();

                    // first aggregate
                    aggregateScaffold = new Scaffold(nextScaffold);
                    aggregateScaffold.setIndexId(aggregateScaffoldCounter);
                    aggregateScaffold.setOriginallyInverted(false);

                    continue;
                }

                // rest of scaffolds
                signScafId = listOfSuperscaffolds.get(i).get(j);
                nextScaffold = listOfScaffolds.get(Math.abs(signScafId) - 1);
                nextScaffold.setCurrentStart(scaffoldShift);
                nextScaffold.setInvertedVsInitial(false);
                if (signScafId < 0 && (!listOfScaffolds.get(Math.abs(signScafId) - 1).getOriginallyInverted()) ||
                        signScafId > 0 && listOfScaffolds.get(Math.abs(signScafId) - 1).getOriginallyInverted()) {
                    nextScaffold.setInvertedVsInitial(true);
                }
                scaffoldFeature2DList.add(chrIndex, chrIndex, nextScaffold.getCurrentFeature2D());
                scaffoldShift += nextScaffold.getLength();

                // try to aggregate
                tempAggregateScaffold = aggregateScaffold.mergeWith(nextScaffold);
                if (tempAggregateScaffold == null) {

                    // if merge failed dump current aggregate
                    listOfAggregateScaffolds.add(aggregateScaffold);

                    // start next aggregate
                    aggregateScaffoldCounter++;
                    aggregateScaffold = new Scaffold(nextScaffold);
                    aggregateScaffold.setIndexId(aggregateScaffoldCounter);
                    aggregateScaffold.setOriginallyInverted(false);

                } else {
                    aggregateScaffold = tempAggregateScaffold;
                }
            }
            Feature2D superscaffoldFeature2D = populateSuperscaffoldFeature2D(superscaffoldShift, scaffoldShift, i);
            superscaffoldFeature2DList.add(chrIndex, chrIndex, superscaffoldFeature2D);
            superscaffoldShift = scaffoldShift;
        }

        listOfAggregateScaffolds.add(aggregateScaffold);

        // create scaffold feature handler
        scaffoldFeature2DHandler = new Feature2DHandler();
        scaffoldFeature2DHandler.loadLoopList(scaffoldFeature2DList, true);
        //scaffoldFeature2DHandler.setSparsePlottingEnabled(true);

        // create superscaffold feature handler
        superscaffoldFeature2DHandler = new Feature2DHandler();
        superscaffoldFeature2DHandler.loadLoopList(superscaffoldFeature2DList, false);

        AssemblyHeatmapHandler.setListOfOSortedAggregateScaffolds(listOfAggregateScaffolds);
        // aggregate list is already sorted, no need to sort again

    }

    private Feature2D populateSuperscaffoldFeature2D(long start, long end, int i) {
        Map<String, String> attributes = new HashMap<String, String>();
        attributes.put(superScaffoldIdAttributeKey, String.valueOf(i + 1));
        return new Feature2D(Feature2D.FeatureType.SUPERSCAFFOLD,
                chrName,
                (int) Math.round(start / HiCGlobals.hicMapScale),
                (int) Math.round(end / HiCGlobals.hicMapScale),
                chrName,
                (int) Math.round(start / HiCGlobals.hicMapScale),
                (int) Math.round(end / HiCGlobals.hicMapScale),
                new Color(0, 0, 255),
                attributes);
    }

    private void updateSuperscaffolds() {
        Feature2DList superscaffoldFeature2DList = new Feature2DList();
        long superscaffoldStart = 0;
        for (int superscaffold = 0; superscaffold < listOfSuperscaffolds.size(); superscaffold++) {
            long superscaffoldLength = 0;
            for (int scaffold : listOfSuperscaffolds.get(superscaffold)) {
                superscaffoldLength += listOfScaffolds.get(Math.abs(scaffold) - 1).getLength();
            }
            Feature2D superscaffoldFeature2D = populateSuperscaffoldFeature2D(superscaffoldStart, superscaffoldStart + superscaffoldLength, superscaffold);
            superscaffoldFeature2DList.add(chrIndex, chrIndex, superscaffoldFeature2D);
            superscaffoldStart += superscaffoldLength;
        }

        superscaffoldFeature2DHandler = new Feature2DHandler();
        superscaffoldFeature2DHandler.loadLoopList(superscaffoldFeature2DList, false);
    }


    //************************** Communication with Feature2D ***************************//

    private int getSignedIndexFromScaffoldFeature2D(Feature2D scaffoldFeature2D) {
        if (scaffoldFeature2D.getFeatureType() == Feature2D.FeatureType.SCAFFOLD) {
            return Integer.parseInt(scaffoldFeature2D.getAttribute(signedScaffoldIdAttributeKey));
        } else {
            return 0;
        }
    }

    private int getUnSignedIndexFromScaffoldFeature2D(Feature2D scaffoldFeature2D) {
        if (scaffoldFeature2D.getFeatureType() == Feature2D.FeatureType.SCAFFOLD) {
            return Integer.parseInt(scaffoldFeature2D.getAttribute(unsignedScaffoldIdAttributeKey));
        } else {
            return 0;
        }
    }

    public Scaffold getScaffoldFromFeature(Feature2D scaffoldFeature2D) {
        if (scaffoldFeature2D.getFeatureType() == Feature2D.FeatureType.SCAFFOLD) {
            int i = getUnSignedIndexFromScaffoldFeature2D(scaffoldFeature2D) - 1;
            return listOfScaffolds.get(i);
        }
        return null;
    }

    public Scaffold getAggegateScaffoldFromFeature(Feature2D aggregateScaffoldFeature2D) {
        if (aggregateScaffoldFeature2D.getFeatureType() == Feature2D.FeatureType.SCAFFOLD) {
            int i = getUnSignedIndexFromScaffoldFeature2D(aggregateScaffoldFeature2D) - 1;
            return listOfAggregateScaffolds.get(i);
        }
        return null;
    }


    //************************************** Actions **************************************//



    //**** Split fragment ****//

    public void editScaffold(Feature2D targetFeature, Feature2D debrisFeature) {

        Scaffold targetScaffold = getScaffoldFromFeature(targetFeature);
        // do not allow for splitting debris scaffoldFeature2DList, TODO: should probably also handle at the level of prompts
        if (targetScaffold.isDebris()) {
            return;
        }

        // calculate split with respect to fragment
        long startCut;
        long endCut;

        if (targetScaffold.getSignIndexId() > 0) {
            startCut = (long) (debrisFeature.getStart1() * HiCGlobals.hicMapScale) - targetScaffold.getCurrentStart();
            endCut = (long) (debrisFeature.getEnd1() * HiCGlobals.hicMapScale) - targetScaffold.getCurrentStart();
        } else {
            startCut = (long) (targetScaffold.getCurrentEnd() - debrisFeature.getEnd1() * HiCGlobals.hicMapScale);
            endCut = (long) (targetScaffold.getCurrentEnd() - debrisFeature.getStart1() * HiCGlobals.hicMapScale);
        }

        editCprops(targetScaffold, startCut, endCut);
        editAsm(targetScaffold);
    }

    private void editAsm(Scaffold scaffold) {
        List<Integer> debrisSuperscaffold = new ArrayList<>();
        for (int i = 0; i < listOfSuperscaffolds.size(); i++) {
            int scaffoldId = scaffold.getIndexId();
            for (int j = 0; j < listOfSuperscaffolds.get(i).size(); j++) {
                listOfSuperscaffolds.get(i).set(j, modifyScaffoldId(listOfSuperscaffolds.get(i).get(j), scaffoldId));
            }
            if (listOfSuperscaffolds.get(i).contains(scaffoldId)) {
                listOfSuperscaffolds.get(i).add(listOfSuperscaffolds.get(i).indexOf(scaffoldId) + 1, scaffoldId + 2);
                debrisSuperscaffold.add(scaffoldId + 1);
            } else if (listOfSuperscaffolds.get(i).contains(-scaffoldId)) {
                listOfSuperscaffolds.get(i).add(listOfSuperscaffolds.get(i).indexOf(-scaffoldId), -scaffoldId - 2);
                debrisSuperscaffold.add(-scaffoldId - 1);
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

    private void editCprops(Scaffold targetScaffold, long startCut, long endCut) {
        List<Scaffold> newListOfScaffolds = new ArrayList<>();
        //List<FragmentProperty> addedProperties = new ArrayList<>();
        int startingFragmentNumber;
        for (Scaffold scaffoldProperty : listOfScaffolds) {
            if (scaffoldProperty.getIndexId() < targetScaffold.getIndexId()) {
                newListOfScaffolds.add(scaffoldProperty);
            } else if (scaffoldProperty.getIndexId() == targetScaffold.getIndexId()) {
                startingFragmentNumber = scaffoldProperty.getFragmentNumber();
                if (startingFragmentNumber == 0) {
                    startingFragmentNumber++;
                } // first ever split
                newListOfScaffolds.add(new Scaffold(scaffoldProperty.getOriginalScaffoldName() + ":::fragment_" + (startingFragmentNumber), scaffoldProperty.getIndexId(), startCut));
                newListOfScaffolds.add(new Scaffold(scaffoldProperty.getOriginalScaffoldName() + ":::fragment_" + (startingFragmentNumber + 1) + ":::debris", scaffoldProperty.getIndexId() + 1, endCut - startCut));
                newListOfScaffolds.add(new Scaffold(scaffoldProperty.getOriginalScaffoldName() + ":::fragment_" + (startingFragmentNumber + 2), scaffoldProperty.getIndexId() + 2, scaffoldProperty.getLength() - endCut));
                // set their initial properties
                if (!targetScaffold.getOriginallyInverted()) {
                    newListOfScaffolds.get(newListOfScaffolds.size() - 3).setOriginalStart(targetScaffold.getOriginalStart());
                    newListOfScaffolds.get(newListOfScaffolds.size() - 3).setOriginallyInverted(false);
                    newListOfScaffolds.get(newListOfScaffolds.size() - 2).setOriginalStart(targetScaffold.getOriginalStart() + startCut);
                    newListOfScaffolds.get(newListOfScaffolds.size() - 2).setOriginallyInverted(false);
                    newListOfScaffolds.get(newListOfScaffolds.size() - 1).setOriginalStart(targetScaffold.getOriginalStart() + endCut);
                    newListOfScaffolds.get(newListOfScaffolds.size() - 1).setOriginallyInverted(false);
                } else {
                    newListOfScaffolds.get(newListOfScaffolds.size() - 1).setOriginalStart(targetScaffold.getOriginalStart());
                    newListOfScaffolds.get(newListOfScaffolds.size() - 1).setOriginallyInverted(true);
                    newListOfScaffolds.get(newListOfScaffolds.size() - 2).setOriginalStart(targetScaffold.getOriginalStart() - endCut);
                    newListOfScaffolds.get(newListOfScaffolds.size() - 2).setOriginallyInverted(true);
                    newListOfScaffolds.get(newListOfScaffolds.size() - 3).setOriginalStart(targetScaffold.getOriginalStart() - startCut);
                    newListOfScaffolds.get(newListOfScaffolds.size() - 3).setOriginallyInverted(true);
                }
            } else {
                if (scaffoldProperty.getOriginalScaffoldName().equals(targetScaffold.getOriginalScaffoldName())) {
                    if (scaffoldProperty.isDebris())
                        scaffoldProperty.setName(scaffoldProperty.getOriginalScaffoldName() + ":::fragment_" + (scaffoldProperty.getFragmentNumber() + 2) + ":::debris");
                    else
                        scaffoldProperty.setName(scaffoldProperty.getOriginalScaffoldName() + ":::fragment_" + (scaffoldProperty.getFragmentNumber() + 2));
                }
                scaffoldProperty.setIndexId(scaffoldProperty.getIndexId() + 2);
                newListOfScaffolds.add(scaffoldProperty);
            }
        }

        listOfScaffolds.clear();
        listOfScaffolds.addAll(newListOfScaffolds);
    }




    //**** Inversion ****//

    public void invertSelection(List<Feature2D> scaffoldFeatures) {

        int id1 = getSignedIndexFromScaffoldFeature2D(scaffoldFeatures.get(0));
        int id2 = getSignedIndexFromScaffoldFeature2D(scaffoldFeatures.get(scaffoldFeatures.size() - 1));
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

        //update scaffold inversion status
        for (Feature2D feature : scaffoldFeatures) {
            getScaffoldFromFeature(feature).toggleInversion();
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

    //**** Move selection ****//

    public void moveSelection(List<Feature2D> selectedFeatures, Feature2D upstreamFeature) {
        // note assumes sorted
        int id1 = getSignedIndexFromScaffoldFeature2D(selectedFeatures.get(0));
        int id2 = getSignedIndexFromScaffoldFeature2D(selectedFeatures.get(selectedFeatures.size() - 1));
        int id3 = getSignedIndexFromScaffoldFeature2D(upstreamFeature);

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
        int id1 = getSignedIndexFromScaffoldFeature2D(upstreamFeature2D);
        int id2 = getSignedIndexFromScaffoldFeature2D(downstreamFeature2D);

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

    @Override
    public String toString() {
        String s = Arrays.toString(listOfSuperscaffolds.toArray());
        return s;
    }


    public Feature2DHandler getOriginalAggregateFeature2DHandler() {
        return originalAggregateFeature2DHandler;
    }

    public Feature2DHandler getCurrentAggregateFeature2DHandler() {
        return currentAggregateFeature2DHandler;
    }

    public Feature2DHandler getScaffoldFeature2DHandler() {
        return scaffoldFeature2DHandler;
    }

    public Feature2DHandler getSuperscaffoldFeature2DHandler() {
        return superscaffoldFeature2DHandler;
    }

    public Scaffold lookUpCurrentAggregateScaffold(long genomicPos) {
        Scaffold tmp = new Scaffold("tmp", 1, 1);
        tmp.setCurrentStart(genomicPos);
        int idx = Collections.binarySearch(listOfAggregateScaffolds, tmp);
        return listOfAggregateScaffolds.get(idx);
    }

    public List<Scaffold> getListOfAggregateScaffolds() {
        return listOfAggregateScaffolds;
    }

    public List<Scaffold> getIntersectingAggregateFeatures(long genomicPos1, long genomicPos2) {
        Scaffold tmp = new Scaffold("tmp", 1, 1);
        tmp.setCurrentStart(genomicPos1);
        int idx1 = Collections.binarySearch(listOfAggregateScaffolds, tmp);
        if (-idx1 - 2 < 0) {
            idx1 = 0;
        } else {
            idx1 = -idx1 - 2;
        }
        tmp = new Scaffold("tmp", 1, 1);
        tmp.setCurrentStart(genomicPos2);
        int idx2 = Collections.binarySearch(listOfAggregateScaffolds, tmp);
        if (-idx2 - 2 <= 0) {
            idx2 = listOfAggregateScaffolds.size() - 1;
        } else {
            idx2 = -idx2 - 2;
        }
        return listOfAggregateScaffolds.subList(idx1, idx2 + 1);
    }
}