/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2021 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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
 *  FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
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
import java.util.List;
import java.util.*;

/**
 * Created by nathanielmusial on 6/30/17.
 */
public class AssemblyScaffoldHandler {

    //constants, have just one attribute
    private final String unsignedScaffoldIdAttributeKey = "Scaffold #";
    private final String signedScaffoldIdAttributeKey = "Signed scaffold #";
    private final String scaffoldNameAttributeKey = "Scaffold name";
    private final String superScaffoldIdAttributeKey = "Superscaffold #";
    private final String superScaffoldNameAttributeKey = "Superscaffold name";
    // formalities
    private final int chrIndex = 1;
    private final String chrName = "assembly";
    //scaffolds and superscaffolds
    private List<Scaffold> listOfScaffolds = new ArrayList<>();
    private List<List<Integer>> listOfSuperscaffolds = new ArrayList<>();
    // aggregates
    private List<Scaffold> listOfAggregateScaffolds = new ArrayList<>();
    private List<String> listOfBundledScaffolds = new ArrayList<>();
    private Feature2DHandler scaffoldFeature2DHandler;
    private Feature2DHandler superscaffoldFeature2DHandler;


    public AssemblyScaffoldHandler(List<Scaffold> listOfScaffolds, List<List<Integer>> listOfSuperscaffolds) {
        this.listOfScaffolds = listOfScaffolds;
        this.listOfSuperscaffolds = listOfSuperscaffolds;
        updateAssembly(true);
    }

    public AssemblyScaffoldHandler(List<Scaffold> listOfScaffolds, List<List<Integer>> listOfSuperscaffolds, List<String> bundled) {
        this.listOfScaffolds = listOfScaffolds;
        this.listOfSuperscaffolds = listOfSuperscaffolds;
        this.listOfBundledScaffolds = bundled;
        updateAssembly(true);
    }

    public AssemblyScaffoldHandler(AssemblyScaffoldHandler assemblyScaffoldHandler) {
        this.listOfScaffolds = assemblyScaffoldHandler.cloneScaffolds();
        this.listOfSuperscaffolds = assemblyScaffoldHandler.cloneSuperscaffolds();
        this.listOfBundledScaffolds = assemblyScaffoldHandler.getListOfBundledScaffolds();
        this.scaffoldFeature2DHandler = assemblyScaffoldHandler.getScaffoldFeature2DHandler();
        this.superscaffoldFeature2DHandler = assemblyScaffoldHandler.getSuperscaffoldFeature2DHandler();
        this.listOfAggregateScaffolds = assemblyScaffoldHandler.getListOfAggregateScaffolds();
        //updateAssembly(true);
    }

    private List<Scaffold> cloneScaffolds() {
        List<Scaffold> newListOfScaffolds = new ArrayList<>();
        for (Scaffold scaffold : listOfScaffolds) {
            newListOfScaffolds.add(new Scaffold(scaffold));
        }
        return newListOfScaffolds;
    }

    private List<List<Integer>> cloneSuperscaffolds() {
        List<List<Integer>> newListOfSuperScaffolds = new ArrayList<>();
        for (List<Integer> superscaffold : listOfSuperscaffolds) {
            newListOfSuperScaffolds.add(new ArrayList<>(superscaffold));
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

        listOfAggregateScaffolds = new ArrayList<>();

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
        scaffoldFeature2DHandler = new Feature2DHandler(scaffoldFeature2DList);
        //scaffoldFeature2DHandler.setSparsePlottingEnabled(true);

//        Uncomment to visualize aggregate scaffold boundaries
//        for (Scaffold temp : listOfAggregateScaffolds){
//            temp.setAssociatedFeatureColor(Color.black);
//            superscaffoldFeature2DList.add(chrIndex, chrIndex, temp.getCurrentFeature2D());
//        }

        // create superscaffold feature handler
        superscaffoldFeature2DHandler = new Feature2DHandler(superscaffoldFeature2DList);
        AssemblyHeatmapHandler.setListOfOSortedAggregateScaffolds(listOfAggregateScaffolds);
        // aggregate list is already sorted, no need to sort again
//        System.out.println("update assembly: " + listOfAggregateScaffolds.size());
    }

    private Feature2D populateSuperscaffoldFeature2D(long start, long end, int i) {
        Map<String, String> attributes = new HashMap<>();
        attributes.put(superScaffoldIdAttributeKey, String.valueOf(i + 1));
        attributes.put(superScaffoldNameAttributeKey, listOfScaffolds.get(Math.abs(listOfSuperscaffolds.get(i).get(0)) - 1).getName());
        return new Feature2D(Feature2D.FeatureType.SUPERSCAFFOLD,
                chrName,
                (int) (start / HiCGlobals.hicMapScale),
                (int) (end / HiCGlobals.hicMapScale),
                chrName,
                (int) (start / HiCGlobals.hicMapScale),
                (int) (end / HiCGlobals.hicMapScale),
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
            Feature2D
                    superscaffoldFeature2D =
                    populateSuperscaffoldFeature2D(superscaffoldStart, superscaffoldStart + superscaffoldLength, superscaffold);
            superscaffoldFeature2DList.add(chrIndex, chrIndex, superscaffoldFeature2D);
            superscaffoldStart += superscaffoldLength;
        }

        superscaffoldFeature2DHandler = new Feature2DHandler(superscaffoldFeature2DList);
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
        for (List<Integer> listOfSuperscaffold : listOfSuperscaffolds) {
            int scaffoldId = scaffold.getIndexId();
            for (int j = 0; j < listOfSuperscaffold.size(); j++) {
                listOfSuperscaffold.set(j, modifyScaffoldId(listOfSuperscaffold.get(j), scaffoldId));
            }
            if (listOfSuperscaffold.contains(scaffoldId)) {
                listOfSuperscaffold.add(listOfSuperscaffold.indexOf(scaffoldId) + 1, scaffoldId + 2);
                debrisSuperscaffold.add(scaffoldId + 1);
            } else if (listOfSuperscaffold.contains(-scaffoldId)) {
                listOfSuperscaffold.add(listOfSuperscaffold.indexOf(-scaffoldId), -scaffoldId - 2);
                debrisSuperscaffold.add(-scaffoldId - 1);
            }
        }
        listOfSuperscaffolds.add(debrisSuperscaffold);
    }

    private int modifyScaffoldId(int index, int cutElementId) {
        if (Math.abs(index) <= cutElementId) {
            return index;
        } else {
            if (index > 0) {
                return index + 2;
            } else {
                return index - 2;
            }
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
                newListOfScaffolds.add(new Scaffold(scaffoldProperty.getOriginalScaffoldName() +
                        ":::fragment_" +
                        (startingFragmentNumber), scaffoldProperty.getIndexId(), startCut));
                newListOfScaffolds.add(new Scaffold(scaffoldProperty.getOriginalScaffoldName() +
                        ":::fragment_" +
                        (startingFragmentNumber + 1) +
                        ":::debris", scaffoldProperty.getIndexId() + 1, endCut - startCut));
                newListOfScaffolds.add(new Scaffold(scaffoldProperty.getOriginalScaffoldName() +
                        ":::fragment_" +
                        (startingFragmentNumber + 2), scaffoldProperty.getIndexId() + 2, scaffoldProperty.getLength() - endCut));
                // set their initial properties
                int lastIndexScaffolds = newListOfScaffolds.size();
                if (!targetScaffold.getOriginallyInverted()) {
                    newListOfScaffolds.get(lastIndexScaffolds - 3).setOriginalStart(targetScaffold.getOriginalStart());
                    newListOfScaffolds.get(lastIndexScaffolds - 3).setOriginallyInverted(false);
                    newListOfScaffolds.get(lastIndexScaffolds - 2)
                            .setOriginalStart(targetScaffold.getOriginalStart() + startCut);
                    newListOfScaffolds.get(lastIndexScaffolds - 2).setOriginallyInverted(false);
                    newListOfScaffolds.get(lastIndexScaffolds - 1)
                            .setOriginalStart(targetScaffold.getOriginalStart() + endCut);
                    newListOfScaffolds.get(lastIndexScaffolds - 1).setOriginallyInverted(false);
                } else {
                    newListOfScaffolds.get(lastIndexScaffolds - 1).setOriginalStart(targetScaffold.getOriginalStart());
                    newListOfScaffolds.get(lastIndexScaffolds - 1).setOriginallyInverted(true);
                    newListOfScaffolds.get(lastIndexScaffolds - 2)
                            .setOriginalStart(targetScaffold.getOriginalEnd() - endCut - 1);
                    newListOfScaffolds.get(lastIndexScaffolds - 2).setOriginallyInverted(true);
                    newListOfScaffolds.get(lastIndexScaffolds - 3)
                            .setOriginalStart(targetScaffold.getOriginalEnd() - startCut - 1);
                    newListOfScaffolds.get(lastIndexScaffolds - 3).setOriginallyInverted(true);
                }
            } else {
                if (scaffoldProperty.getOriginalScaffoldName().equals(targetScaffold.getOriginalScaffoldName())) {
                    if (scaffoldProperty.isDebris()) {
                        scaffoldProperty.setName(scaffoldProperty.getOriginalScaffoldName() +
                                ":::fragment_" +
                                (scaffoldProperty.getFragmentNumber() + 2) +
                                ":::debris");
                    } else {
                        scaffoldProperty.setName(scaffoldProperty.getOriginalScaffoldName() +
                                ":::fragment_" +
                                (scaffoldProperty.getFragmentNumber() + 2));
                    }
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

        if (gid1 == gid2) {
            Collections.reverse(listOfSuperscaffolds.get(gid1)
                    .subList(listOfSuperscaffolds.get(gid1).indexOf(id1), listOfSuperscaffolds.get(gid2).indexOf(id2) + 1));
            for (int i = listOfSuperscaffolds.get(gid1).indexOf(id2); i <= listOfSuperscaffolds.get(gid2).indexOf(id1); i++) {
                listOfSuperscaffolds.get(gid1).set(i, -1 * listOfSuperscaffolds.get(gid1).get(i));
            }
        } else {
            List<List<Integer>> newListOfSuperscaffolds = new ArrayList<>();
            for (int i = 0; i <= listOfSuperscaffolds.size() - 1; i++) {
                if (i >= gid1 && i <= gid2) {
                    newListOfSuperscaffolds.add(listOfSuperscaffolds.get(gid2 - i + gid1));
                    Collections.reverse(newListOfSuperscaffolds.get(i));
                    for (int j = 0; j <= newListOfSuperscaffolds.get(i).size() - 1; j++) {
                        newListOfSuperscaffolds.get(i).set(j, -1 * newListOfSuperscaffolds.get(i).get(j));
                    }
                } else {
                    newListOfSuperscaffolds.add(listOfSuperscaffolds.get(i));
                }
            }
            listOfSuperscaffolds.clear();
            listOfSuperscaffolds.addAll(newListOfSuperscaffolds);
        }
    }


    //**** Move selection ****//

    void moveSelection(List<Feature2D> selectedFeatures, Feature2D upstreamFeature) {
        // note assumes sorted
        int id1 = getSignedIndexFromScaffoldFeature2D(selectedFeatures.get(0));
        int id2 = getSignedIndexFromScaffoldFeature2D(selectedFeatures.get(selectedFeatures.size() - 1));
        int gid1 = getSuperscaffoldId(id1);
        int gid2 = getSuperscaffoldId(id2);

        // initialize id3 and gid3
        int id3 = 0;
        int gid3 = -1;
        if (upstreamFeature != null) {

            // initialize id3 and gid3 if not inserting to top
            id3 = getSignedIndexFromScaffoldFeature2D(upstreamFeature);
            gid3 = getSuperscaffoldId(id3);

            // check if selectedFeatures span multiple groups paste split at destination
            if (gid1 != gid2 & listOfSuperscaffolds.get(gid3).indexOf(id3) != listOfSuperscaffolds.get(gid3).size() - 1) {

                // handles pasting scaffolds from multiple groups into another group
                splitSuperscaffold(gid3, id3);
                gid1 = getSuperscaffoldId(id1);
                gid2 = getSuperscaffoldId(id2);
                gid3 = getSuperscaffoldId(id3);
            }
        }

        List<List<Integer>> newSuperscaffolds = new ArrayList<>();
        List<List<Integer>> tempSuperscaffolds = new ArrayList<>();
        List<Integer> truncatedSuperscaffold = new ArrayList<>();
        int shiftSuperscaffold = 0;
        for (int i = 0; i <= listOfSuperscaffolds.size() - 1; i++) {
            if (i == gid1 && i == gid2) {

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

            } else if (gid1 != gid2 && i == gid1) {
                tempSuperscaffolds.add(listOfSuperscaffolds.get(gid1).subList(listOfSuperscaffolds.get(gid1).indexOf(id1), listOfSuperscaffolds.get(gid1).size()));
                if (listOfSuperscaffolds.get(gid1).indexOf(id1) != 0) {
                    newSuperscaffolds.add(listOfSuperscaffolds.get(gid1).subList(0, listOfSuperscaffolds.get(gid1).indexOf(id1)));
                } else {
                    shiftSuperscaffold++;
                }
            } else if (i > gid1 && i < gid2) { //gid1 != gid2 is covered/inherent to the condition
                tempSuperscaffolds.add(listOfSuperscaffolds.get(i));
                shiftSuperscaffold++;
            } else if (gid1 != gid2 && i == gid2) {
                tempSuperscaffolds.add(listOfSuperscaffolds.get(gid2).subList(0, 1 + listOfSuperscaffolds.get(gid2).indexOf(id2)));
                if (listOfSuperscaffolds.get(gid2).indexOf(id2) != listOfSuperscaffolds.get(gid2).size() - 1) {
                    newSuperscaffolds.add(listOfSuperscaffolds.get(gid2).subList(1 + listOfSuperscaffolds.get(gid2).indexOf(id2), listOfSuperscaffolds.get(gid2).size()));
                } else {
                    shiftSuperscaffold++;
                }
            } else {
                newSuperscaffolds.add(listOfSuperscaffolds.get(i));
            }
        }

        // check if inserting to top through PASTETOP
        if (upstreamFeature != null) {
            int newgid3 = gid3;
            if (gid3 > gid2) {
                // if moving a superscaffold downstream
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
        } else
            // if inserting to top, add all scaffolds to index 0
            newSuperscaffolds.addAll(0, tempSuperscaffolds);

        listOfSuperscaffolds.clear();
        listOfSuperscaffolds.addAll(newSuperscaffolds);
    }
    //**** Group toggle ****//

    public void toggleGroup(Feature2D upstreamFeature2D, Feature2D downstreamFeature2D) {
        int id1 = getSignedIndexFromScaffoldFeature2D(upstreamFeature2D);
        int id2 = getSignedIndexFromScaffoldFeature2D(downstreamFeature2D);

        //should not happen, other sanity checks?
        if (id1 == id2) {
            return;
        }

        int super1 = getSuperscaffoldId(id1);
        int super2 = getSuperscaffoldId(id2);

        if (super1 == super2) {
            splitSuperscaffold(super1, id1);
        } else {
            mergeSuperScaffolds(super1, super2);
        }
    }

    public void multiMerge(Feature2D firstFeature, Feature2D lastFeature) {
        int super1 = getSuperscaffoldId(getSignedIndexFromScaffoldFeature2D(firstFeature));
        int super2 = getSuperscaffoldId(getSignedIndexFromScaffoldFeature2D(lastFeature));

        mergeSuperScaffolds(super1, super2);
    }


//    public void phaseMerge(List<Feature2D> selectedFeatures) {
//
//        List<List<Integer>> newSuperscaffolds = new ArrayList<>();
//
//        List<Integer> newSuperscaffold = new ArrayList<>();
//        List<Integer> newAltSuperscaffold = new ArrayList<>();
//
//        for (Feature2D feature : selectedFeatures) {
//            int id = getSignedIndexFromScaffoldFeature2D(feature);
//            newSuperscaffold.add(id);
//        }
//        Collections.sort(newSuperscaffold);
//        for (int i : newSuperscaffold) {
//            if (i % 2 == 0)
//                newAltSuperscaffold.add(i - 1);
//            else
//                newAltSuperscaffold.add(i + 1);
//        }
//
//        boolean test = true;
//        for (List<Integer> superscaffold : listOfSuperscaffolds) {
//            // if (Collections.disjoint(listOfSuperscaffolds.get(i), newSuperscaffold)) {
//            if (!newSuperscaffold.contains(superscaffold.get(0)) && !newAltSuperscaffold.contains(superscaffold.get(0))) {
//                newSuperscaffolds.add(superscaffold);
//            } else {
//                if (test) {
//                    if (getSuperscaffoldId(newSuperscaffold.get(0)) < getSuperscaffoldId(newAltSuperscaffold.get(0))) {
//                        newSuperscaffolds.add(newSuperscaffold);
//                        newSuperscaffolds.add(newAltSuperscaffold);
//                    } else {
//                        newSuperscaffolds.add(newAltSuperscaffold);
//                        newSuperscaffolds.add(newSuperscaffold);
//                    }
//                    test = false;
//                }
//                continue;
//            }
//        }
//
//        listOfSuperscaffolds.clear();
//        listOfSuperscaffolds.addAll(newSuperscaffolds);
//    }

    public void multiSplit(List<Feature2D> selectedFeatures) {
        int id1 = getSignedIndexFromScaffoldFeature2D(selectedFeatures.get(0));
        int id2 = getSignedIndexFromScaffoldFeature2D(selectedFeatures.get(selectedFeatures.size() - 1));
        int super1 = getSuperscaffoldId(id1);
        int super2 = getSuperscaffoldId(id2);

        multiSplitSuperscaffolds(id1, id2, super1, super2);
    }

    // SuperScaffold manipulations
  private void mergeSuperScaffolds (int super1, int super2){
      if (HiCGlobals.phasing) {
          List<Integer> idListToMerge = new ArrayList<>();
          idListToMerge.add(super1);
          idListToMerge.add(super2);
          phaseMerge(idListToMerge);
          return;
      }
    List<List<Integer>> newSuperscaffolds = new ArrayList<>();
    for (int i = 0; i < listOfSuperscaffolds.size(); i++) {
      if(i>super1 && i<=super2){
        newSuperscaffolds.get(super1).addAll(listOfSuperscaffolds.get(i));
      }else{
        newSuperscaffolds.add(listOfSuperscaffolds.get(i));
      }
    }
    listOfSuperscaffolds.clear();
    listOfSuperscaffolds.addAll(newSuperscaffolds);
  }

    public void phaseMerge(List<Integer> idListToMerge) {

        List<List<Integer>> newSuperscaffolds = new ArrayList<>();
        List<Integer> newSuperscaffold = new ArrayList<>();
        List<Integer> newAltSuperscaffold = new ArrayList<>();
        List<Integer> altIdListToMerge = new ArrayList<>();

        for (int i : idListToMerge) {
            newSuperscaffold.addAll(listOfSuperscaffolds.get(i));
            if (i % 2 == 0) {
                altIdListToMerge.add(i + 1);
            } else {
                altIdListToMerge.add(i - 1);
            }
        }
        idListToMerge.addAll(altIdListToMerge);
        Collections.sort(idListToMerge);

        boolean altGoesFirst = false;
        if (altIdListToMerge.contains(idListToMerge.get(0))) {
            altGoesFirst = true;
        }

        if (!HiCGlobals.noSortInPhasing)
            Collections.sort(newSuperscaffold);

        for (int i : newSuperscaffold) {
            if (i % 2 == 0)
                newAltSuperscaffold.add(i - 1);
            else {
                newAltSuperscaffold.add(i + 1);
            }
        }

        for (int i = 0; i < listOfSuperscaffolds.size(); i++) {
            List<Integer> superscaffold = listOfSuperscaffolds.get(i);
            if (!idListToMerge.contains(Integer.valueOf(i))) {
                newSuperscaffolds.add(superscaffold);
            } else if (i == idListToMerge.get(0)) {
                if (!altGoesFirst) {
                    newSuperscaffolds.add(newSuperscaffold);
                    newSuperscaffolds.add(newAltSuperscaffold);
                } else {
                    newSuperscaffolds.add(newAltSuperscaffold);
                    newSuperscaffolds.add(newSuperscaffold);
                }
            }
        }

        listOfSuperscaffolds.clear();
        listOfSuperscaffolds.addAll(newSuperscaffolds);
    }

    private void splitSuperscaffold(int superscaffoldId, int scaffoldId) {
        if (HiCGlobals.phasing) {
            if (superscaffoldId % 2 != 0) {
                superscaffoldId -= 1;
                if (scaffoldId % 2 == 0) {
                    scaffoldId -= 1;
                } else {
                    scaffoldId += 1;
                }
            }
        }

        List<List<Integer>> newSuperscaffolds = new ArrayList<>();
        for (int i = 0; i < listOfSuperscaffolds.size(); i++) {
            if (i == superscaffoldId) {
                int breakPointIndex = listOfSuperscaffolds.get(superscaffoldId).indexOf(scaffoldId);
                newSuperscaffolds.add(listOfSuperscaffolds.get(superscaffoldId)
                        .subList(0, 1 + breakPointIndex));
                if (HiCGlobals.phasing) {
                    newSuperscaffolds.add(listOfSuperscaffolds.get(superscaffoldId + 1)
                            .subList(0, 1 + breakPointIndex));
                }
                newSuperscaffolds.add(listOfSuperscaffolds.get(superscaffoldId)
                        .subList(1 + breakPointIndex,
                                listOfSuperscaffolds.get(superscaffoldId).size()));
                if (HiCGlobals.phasing) {
                    newSuperscaffolds.add(listOfSuperscaffolds.get(superscaffoldId + 1)
                            .subList(1 + breakPointIndex,
                                    listOfSuperscaffolds.get(superscaffoldId + 1).size()));
                    i++;
                }
            } else {
                newSuperscaffolds.add(listOfSuperscaffolds.get(i));
            }
        }
        listOfSuperscaffolds.clear();
        listOfSuperscaffolds.addAll(newSuperscaffolds);


    }


    private void multiSplitSuperscaffolds(int id1, int id2, int super1, int super2) {
        List<List<Integer>> newSuperscaffolds = new ArrayList<>();
        int startPoint = listOfSuperscaffolds.get(super1).indexOf(id1);
        int endPoint = listOfSuperscaffolds.get(super2).indexOf(id2);
        int jstart, jend;
        boolean addEndScaff = false;

        newSuperscaffolds.addAll(listOfSuperscaffolds.subList(0, super1));

        for (int i = super1; i <= super2; i++) {
            jstart = 0;
            jend = listOfSuperscaffolds.get(i).size() - 1;

            // If at first superscaffold and selected start scaffold not at beginning of current superscaffold
            if (i == super1 && startPoint != 0) {
                jstart = startPoint;

                // Add rest of superscaffold to its own superscaffold
                newSuperscaffolds.add(listOfSuperscaffolds.get(i).subList(0, jstart));
            }
            // If at last superscaffold and selected end scaffold not at end of current superscaffold
            if (i == super2 && endPoint != jend) {
                jend = endPoint;
                addEndScaff = true;
            }

            // Add each inner scaffold to its own superscaffold group
            for (int j = jstart; j <= jend; j++) {
                newSuperscaffolds.add(Arrays.asList(listOfSuperscaffolds.get(i).get(j)));
            }

            // If did not end at last scaffold in last superscaffold selected
            if (addEndScaff) {
                // Add rest of superscaffold to its own superscaffold
                newSuperscaffolds.add(listOfSuperscaffolds.get(i).subList(jend + 1, listOfSuperscaffolds.get(i).size()));
            }
        }

        newSuperscaffolds.addAll(listOfSuperscaffolds.subList(super2 + 1, listOfSuperscaffolds.size()));

        listOfSuperscaffolds.clear();
        listOfSuperscaffolds.addAll(newSuperscaffolds);
    }

    //**** Utility functions ****//

    private int getSuperscaffoldId(int scaffoldId) {
        int i = 0;
        for (List<Integer> scaffoldRow : listOfSuperscaffolds) {

            for (int index : scaffoldRow) {
                if (Math.abs(index) == Math.abs(scaffoldId)) {
                    return i;
                }
            }
            i++;
        }
        System.err.println("Cannot find superscaffold containing scaffold " + scaffoldId);
        return -1;
    }

    @Override
    public String toString() {
        return Arrays.toString(listOfSuperscaffolds.toArray());
    }


    public Feature2DHandler getScaffoldFeature2DHandler() {
        return scaffoldFeature2DHandler;
    }

    public Feature2DHandler getSuperscaffoldFeature2DHandler() {
        return superscaffoldFeature2DHandler;
    }

    public List<Scaffold> getListOfAggregateScaffolds() {
        return listOfAggregateScaffolds;
    }

    public List<Scaffold> getIntersectingAggregateFeatures(long genomicPos1, long genomicPos2) {
        Scaffold tmp = new Scaffold("tmp", 1, 1);
        tmp.setCurrentStart(genomicPos1);
        int idx1 = Collections.binarySearch(listOfAggregateScaffolds, tmp);
        idx1 = Math.max(-idx1 - 2, 0);
        tmp.setCurrentStart(genomicPos2);
        int idx2 = Collections.binarySearch(listOfAggregateScaffolds, tmp);
        if (-idx2 - 2 < 0) {
            idx2 = listOfAggregateScaffolds.size() - 1;
        } else {
            idx2 = -idx2 - 2;
        }
        return listOfAggregateScaffolds.subList(idx1, idx2 + 1);
    }

    public List<String> getListOfBundledScaffolds() {
        return listOfBundledScaffolds;
    }
}