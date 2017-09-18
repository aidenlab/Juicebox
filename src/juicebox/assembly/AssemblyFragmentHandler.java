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

    //deprecated
    private final String contigName = "Contig Name";
    private final String scaffoldIndexId = "Scaffold Index";
    private final String scaffoldNum = "Scaffold Number";
    private final String initiallyInvertedStatus = "Initially Inverted";

    private final String fragmentId = "Fragment ID";
    private List<FragmentProperty> contigProperties;
    private List<List<Integer>> scaffoldProperties;
    private Feature2DList contigs;
    private Feature2DList scaffolds;
    private String chromosomeName = "assembly";

    public AssemblyFragmentHandler(List<FragmentProperty> contigProperties, List<List<Integer>> scaffoldProperties) {
        this.contigProperties = contigProperties;
        this.scaffoldProperties = scaffoldProperties;
        updateAssembly();
    }

    public AssemblyFragmentHandler(AssemblyFragmentHandler assemblyFragmentHandler) {
        this.contigProperties = assemblyFragmentHandler.cloneContigProperties();
        this.scaffoldProperties = assemblyFragmentHandler.cloneScaffoldProperties();
        updateAssembly();
    }

    public void updateAssembly() {
        setCurrentState();
        populate2DFeatures();
    }

    public List<FragmentProperty> cloneContigProperties() {
        List<FragmentProperty> newList = new ArrayList<>();
        for (FragmentProperty fragmentProperty : contigProperties) {
            newList.add(new FragmentProperty(fragmentProperty));
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

    public List<FragmentProperty> getContigProperties() {
        return contigProperties;
    }

    public List<List<Integer>> getScaffoldProperties() {
        return scaffoldProperties;
    }


    public void setCurrentState() {
        long shift = 0;
        for (List<Integer> asmGroup : scaffoldProperties) {
            for (Integer entry : asmGroup) {
                int fragmentIterator = Math.abs(entry) - 1;
                FragmentProperty currentFragmentProperty = contigProperties.get(fragmentIterator);
                currentFragmentProperty.setCurrentStart(shift);
                currentFragmentProperty.setInvertedVsInitial(false);
                if (entry < 0 && (!contigProperties.get(fragmentIterator).wasInitiallyInverted()) ||
                        entry > 0 && contigProperties.get(fragmentIterator).wasInitiallyInverted()) {
                    currentFragmentProperty.setInvertedVsInitial(true);
                }
                shift += currentFragmentProperty.getLength();
            }
        }
    }

    public void populate2DFeatures() {
        contigs = new Feature2DList();
        for (FragmentProperty fragmentProperty : contigProperties) {

            Map<String, String> attributes = new HashMap<String, String>();
            //deprecated
            attributes.put(contigName, fragmentProperty.getName());
            attributes.put(scaffoldIndexId, String.valueOf(fragmentProperty.getSignIndexId()));
            attributes.put(initiallyInvertedStatus, Boolean.toString(fragmentProperty.wasInitiallyInverted()));

            // leave only the id to lookup everything else
            attributes.put(fragmentId, String.valueOf(fragmentProperty.getIndexId()));

            // we don't need both contig and fragment. Will drop contig2D in favor of getting evertyihgn from fragment property (change to fragment state?)
            Feature2D feature2D = new Feature2D(Feature2D.FeatureType.CONTIG,
                    chromosomeName,
                    (int) Math.round(fragmentProperty.getCurrentStart() / HiCGlobals.hicMapScale),
                    (int) Math.round((fragmentProperty.getCurrentEnd()) / HiCGlobals.hicMapScale),
                    chromosomeName,
                    (int) Math.round(fragmentProperty.getCurrentStart() / HiCGlobals.hicMapScale),
                    (int) Math.round((fragmentProperty.getCurrentEnd()) / HiCGlobals.hicMapScale),
                    new Color(0, 255, 0),
                    attributes); //todo

            Contig2D contig = feature2D.toContig();
            if (fragmentProperty.isInvertedVsInitial()) {
                contig.toggleInversion(); //assuming initial contig2D inverted = false
            }
            contig.setInitialState(fragmentProperty.getInitialChr(),
                    (int) Math.round(fragmentProperty.getInitialStart() / HiCGlobals.hicMapScale),
                    (int) Math.round(fragmentProperty.getInitialEnd() / HiCGlobals.hicMapScale),
                    fragmentProperty.wasInitiallyInverted());
            contigs.add(1, 1, contig);
            fragmentProperty.setFeature2D(contig);
        }
        scaffolds = new Feature2DList();
        long groupStart = 0;
        for (int row = 0; row < scaffoldProperties.size(); row++) {
            Map<String, String> attributes = new HashMap<String, String>();
            attributes.put(scaffoldNum, String.valueOf(row));
            long totalGroupLength = 0;
            for (int fragment : scaffoldProperties.get(row)) {
                int fragmentIterator = Math.abs(fragment) - 1;
                totalGroupLength += contigProperties.get(fragmentIterator).getLength();
            }
            Feature2D scaffold = new Feature2D(Feature2D.FeatureType.SCAFFOLD,
                    chromosomeName,
                    (int) Math.round(groupStart / HiCGlobals.hicMapScale),
                    (int) Math.round((groupStart + totalGroupLength) / HiCGlobals.hicMapScale),
                    chromosomeName,
                    (int) Math.round(groupStart / HiCGlobals.hicMapScale),
                    (int) Math.round((groupStart + totalGroupLength) / HiCGlobals.hicMapScale),
                    new Color(0, 0, 255),
                    attributes);
            scaffolds.add(1, 1, scaffold);
            groupStart += totalGroupLength;
        }
    }

    //**** Split fragment ****//

    public void editFragment(Feature2D originalFeature, Feature2D debrisFeature) {
        // find the relevant fragment property
        int fragmentIterator = Integer.parseInt(originalFeature.getAttribute(fragmentId)) - 1;
        FragmentProperty toEditFragmentProperty = contigProperties.get(fragmentIterator);
        //FragmentProperty toEditFragmentProperty = feature2DtoContigProperty(originalFeature);

        // do not allow for splitting debris contigs, TODO: should probably also handle at the level of prompts
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
        List<Integer> debrisGroup = new ArrayList<>();
        for (int i = 0; i < scaffoldProperties.size(); i++) {
            int fragmentId = toEditFragmentProperty.getIndexId();
            for (int j = 0; j < scaffoldProperties.get(i).size(); j++) {
                scaffoldProperties.get(i).set(j, modifyFragmentId(scaffoldProperties.get(i).get(j), fragmentId));
            }
            if (scaffoldProperties.get(i).contains(fragmentId)) {
                scaffoldProperties.get(i).add(scaffoldProperties.get(i).indexOf(fragmentId) + 1, fragmentId + 2);
                debrisGroup.add(fragmentId + 1);
            } else if (scaffoldProperties.get(i).contains(-fragmentId)) {
                scaffoldProperties.get(i).add(scaffoldProperties.get(i).indexOf(-fragmentId), -fragmentId - 2);
                debrisGroup.add(-fragmentId - 1);
            }
        }
        scaffoldProperties.add(debrisGroup);
    }

    private int modifyFragmentId(int index, int cutElementId) {
        if (Math.abs(index) <= cutElementId)
            return index;
        else {
            if (index > 0)
                return index + 2;
            else
                return index - 2;
        }
    }

    private void editCprops(FragmentProperty originalFragmentProperty, long startCut, long endCut) {
        List<FragmentProperty> newFragmentProperties = new ArrayList<>();
        //List<FragmentProperty> addedProperties = new ArrayList<>();
        int startingFragmentNumber;
        for (FragmentProperty fragmentProperty : contigProperties) {
            if (fragmentProperty.getIndexId() < originalFragmentProperty.getIndexId()) {
                newFragmentProperties.add(fragmentProperty);
            } else if (fragmentProperty.getIndexId() == originalFragmentProperty.getIndexId()) {
                startingFragmentNumber = fragmentProperty.getFragmentNumber();
                if (startingFragmentNumber == 0) {
                    startingFragmentNumber++;
                } // first ever split
                newFragmentProperties.add(new FragmentProperty(fragmentProperty.getOriginalContigName() + ":::fragment_" + (startingFragmentNumber), fragmentProperty.getIndexId(), startCut));
                newFragmentProperties.add(new FragmentProperty(fragmentProperty.getOriginalContigName() + ":::fragment_" + (startingFragmentNumber + 1) + ":::debris", fragmentProperty.getIndexId() + 1, endCut - startCut));
                newFragmentProperties.add(new FragmentProperty(fragmentProperty.getOriginalContigName() + ":::fragment_" + (startingFragmentNumber + 2), fragmentProperty.getIndexId() + 2, fragmentProperty.getLength() - endCut));
                // set their initial properties
                if (!originalFragmentProperty.wasInitiallyInverted()) {
                    newFragmentProperties.get(newFragmentProperties.size() - 3).setInitialStart(originalFragmentProperty.getInitialStart());
                    newFragmentProperties.get(newFragmentProperties.size() - 3).setInitiallyInverted(false);
                    newFragmentProperties.get(newFragmentProperties.size() - 2).setInitialStart(originalFragmentProperty.getInitialStart() + startCut);
                    newFragmentProperties.get(newFragmentProperties.size() - 2).setInitiallyInverted(false);
                    newFragmentProperties.get(newFragmentProperties.size() - 1).setInitialStart(originalFragmentProperty.getInitialStart() + endCut);
                    newFragmentProperties.get(newFragmentProperties.size() - 1).setInitiallyInverted(false);
                } else {
                    newFragmentProperties.get(newFragmentProperties.size() - 1).setInitialStart(originalFragmentProperty.getInitialStart());
                    newFragmentProperties.get(newFragmentProperties.size() - 1).setInitiallyInverted(true);
                    newFragmentProperties.get(newFragmentProperties.size() - 2).setInitialStart(originalFragmentProperty.getInitialEnd() - endCut);
                    newFragmentProperties.get(newFragmentProperties.size() - 2).setInitiallyInverted(true);
                    newFragmentProperties.get(newFragmentProperties.size() - 3).setInitialStart(originalFragmentProperty.getInitialEnd() - startCut);
                    newFragmentProperties.get(newFragmentProperties.size() - 3).setInitiallyInverted(true);
                }
            } else {
                if (fragmentProperty.getOriginalContigName().equals(originalFragmentProperty.getOriginalContigName())) {
                    if (fragmentProperty.isDebris())
                        fragmentProperty.setName(fragmentProperty.getOriginalContigName() + ":::fragment_" + (fragmentProperty.getFragmentNumber() + 2) + ":::debris");
                    else
                        fragmentProperty.setName(fragmentProperty.getOriginalContigName() + ":::fragment_" + (fragmentProperty.getFragmentNumber() + 2));
                }
                fragmentProperty.setIndexId(fragmentProperty.getIndexId() + 2);
                newFragmentProperties.add(fragmentProperty);
            }
        }

        contigProperties.clear();
        contigProperties.addAll(newFragmentProperties);
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
        List<FragmentProperty> selectedContigProperties = contig2DListToContigPropertyList(contigs);
        for (FragmentProperty fragmentProperty : selectedContigProperties) {
            fragmentProperty.toggleInversion();
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

    public List<FragmentProperty> contig2DListToContigPropertyList(List<Feature2D> contigs) {
        List<FragmentProperty> newList = new ArrayList<FragmentProperty>();
        for (Feature2D feature2D : contigs) {
            newList.add(contig2DToContigProperty(feature2D));
        }
        return newList;
    }

    public FragmentProperty contig2DToContigProperty(Feature2D feature2D) {
        for (FragmentProperty fragmentProperty : contigProperties) {
            if (fragmentProperty.getFeature2D().equals(feature2D)) { //make sure it is okay

                return fragmentProperty;
            }
        }
        return null;
    }

    //**** Move selection ****//

    public void moveSelection(List<Feature2D> selectedFeatures, Feature2D upstreamFeature){
        int id1 = Integer.parseInt(selectedFeatures.get(0).getAttribute(scaffoldIndexId));
        int id2 = Integer.parseInt(selectedFeatures.get(selectedFeatures.size()-1).getAttribute(scaffoldIndexId));
        int id3 = Integer.parseInt(upstreamFeature.getAttribute(scaffoldIndexId));
        moveSelection(id1, id2, id3);
    }

    //**** Move selection ****//

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

        if (gr1==gr2){
            newSplitGroup(gr1, id1);
        }else {
            newMergeGroup(gr1, gr2);
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
        for (FragmentProperty fragmentProperty : contigProperties) {
            s += fragmentProperty.toString() + "\n";
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


    //deprecated
//    public void generateContigsAndScaffolds(boolean initialGeneration, boolean modifiedGeneration, AssemblyFragmentHandler originalAssemblyFragmentHandler) {
//        Map<String, Pair<List<FragmentProperty>, List<FragmentProperty>>> splitFragments = new HashMap<>();
//        contigs = new Feature2DList();
//        scaffolds = new Feature2DList();
//        long contigStartPos = 0;
//        long scaffoldStartPos = 0;
//        long scaffoldLength = 0;
//        Integer rowNum = 0;
//        for (List<Integer> row : scaffoldProperties) {
//            for (Integer contigIndex : row) {
//                FragmentProperty fragmentProperty = contigProperties.get(Math.abs(contigIndex) - 1);
//                String contigName = fragmentProperty.getName();
//                Long contigLength = new Long(fragmentProperty.getLength());
//
//                if (initialGeneration && !modifiedGeneration) {
//                    fragmentProperty.setInitialState(chromosomeName, contigStartPos, (contigStartPos + contigLength), fragmentProperty.isInvertedVsInitial());
//                }
//                if (modifiedGeneration) {
//                    List<FragmentProperty> newList = getRelatedFragmentProperties(originalAssemblyFragmentHandler, fragmentProperty);
//                    //works for non-modified, rewrite
//                    FragmentProperty originalFragmentProperty = newList.get(0);
//
//                    boolean invertedInAsm = false;
//
//                    List<Integer> list = Arrays.asList(fragmentProperty.getIndexId());
//                    if ((originalFragmentProperty.wasInitiallyInverted() && scaffoldProperties.get(getScaffoldRow(list)).contains(fragmentProperty.getIndexId())) ||
//                            ((!originalFragmentProperty.wasInitiallyInverted()) && scaffoldProperties.get(getScaffoldRow(list)).contains(-fragmentProperty.getIndexId()))){
//                    //if (fragmentProperty.getIndexId() != originalFragmentProperty.getIndexId()) {
//                        invertedInAsm = true;
//                    }
//                    fragmentProperty.setInitialState(chromosomeName, originalFragmentProperty.getInitialStart(), originalFragmentProperty.getInitialEnd(), invertedInAsm);
//                    fragmentProperty.setInitiallyInverted(originalFragmentProperty.wasInitiallyInverted());
//
//                    if (fragmentProperty.getName().contains(":::fragment_")) {
//                        addRelevantOriginalContigProperties(newList, splitFragments, fragmentProperty);
//                    }
//                }
//
//                Map<String, String> attributes = new HashMap<String, String>();
//                attributes.put(this.contigName, contigName);
//                attributes.put(scaffoldIndexId, contigIndex.toString());
//                attributes.put(initiallyInvertedStatus, Boolean.toString(fragmentProperty.wasInitiallyInverted()));
//                //put attribute here
//                Feature2D feature2D = new Feature2D(Feature2D.FeatureType.CONTIG, chromosomeName, (int) Math.round(contigStartPos / HiCGlobals.hicMapScale), (int) Math.round((contigStartPos + contigLength) / HiCGlobals.hicMapScale),
//                        chromosomeName, (int) Math.round(contigStartPos / HiCGlobals.hicMapScale), (int) Math.round((contigStartPos + contigLength) / HiCGlobals.hicMapScale),
//                        new Color(0, 255, 0), attributes); //todo
//
//                Contig2D contig = feature2D.toContig();
//                if (fragmentProperty.isInvertedVsInitial()) {
//                    contig.toggleInversion(); //assuming initial contig2D inverted = false
//                }
//                contig.setInitialState(fragmentProperty.getInitialChr(), (int) Math.round(fragmentProperty.getInitialStart() / HiCGlobals.hicMapScale), (int) Math.round(fragmentProperty.getInitialEnd() / HiCGlobals.hicMapScale), fragmentProperty.wasInitiallyInverted());
//                contigs.add(1, 1, contig);
//                fragmentProperty.setFeature2D(contig);
//
//                contigStartPos += contigLength;
//                scaffoldLength += contigLength;
//            }
//            Map<String, String> attributes = new HashMap<String, String>();
//            attributes.put(scaffoldNum, rowNum.toString());
//
//            Feature2D scaffold = new Feature2D(Feature2D.FeatureType.SCAFFOLD, chromosomeName, (int) Math.round(scaffoldStartPos / HiCGlobals.hicMapScale), (int) Math.round((scaffoldStartPos + scaffoldLength) / HiCGlobals.hicMapScale),
//                    chromosomeName, (int) Math.round(scaffoldStartPos / HiCGlobals.hicMapScale), (int) Math.round((scaffoldStartPos + scaffoldLength) / HiCGlobals.hicMapScale),
//                    new Color(0, 0, 255), attributes);
//            scaffolds.add(1, 1, scaffold);
//
//            scaffoldStartPos += scaffoldLength;
//            scaffoldLength = 0;
//            rowNum++;
//        }
//
//        if (modifiedGeneration && !splitFragments.isEmpty()) {
//            printRelevantOriginalContigProperties(splitFragments);
//            fixInitialState(splitFragments);
//        }
//    }

    //    public List<FragmentProperty> getRelatedFragmentProperties(AssemblyFragmentHandler originalAssemblyFragmentHandler, FragmentProperty lookupFragmentProperty) {
//        List<FragmentProperty> originalContigProperties = originalAssemblyFragmentHandler.getContigProperties();
//        List<FragmentProperty> newList = new ArrayList<>();
//        for (FragmentProperty fragmentProperty : originalContigProperties) {
//            if (lookupFragmentProperty.getOriginalContigName().equals(fragmentProperty.getOriginalContigName())){
//                newList.add(fragmentProperty);
//            }
//        }
//        return newList;
//    }

//    public void addRelevantOriginalContigProperties(List<FragmentProperty> originalContigProperties, Map<String, Pair<List<FragmentProperty>, List<FragmentProperty>>> splitFragments, FragmentProperty lookupFragmentProperty) {
//        String originalContigName = lookupFragmentProperty.getOriginalContigName();
//
//        for (FragmentProperty originalFragmentProperty : originalContigProperties) {
//            Pair<List<FragmentProperty>, List<FragmentProperty>> newEntry = new Pair<List<FragmentProperty>, List<FragmentProperty>>(new ArrayList<FragmentProperty>(), new ArrayList<FragmentProperty>());
//            if (!splitFragments.containsKey(originalContigName)) {
//                Pair<List<FragmentProperty>, List<FragmentProperty>> newPair = new Pair<List<FragmentProperty>, List<FragmentProperty>>(new ArrayList<FragmentProperty>(), new ArrayList<FragmentProperty>());
//                splitFragments.put(originalContigName, newPair);
//            }
//            newEntry = splitFragments.get(originalContigName);
//            if (!newEntry.getFirst().contains(originalFragmentProperty))
//                newEntry.getFirst().add(originalFragmentProperty);
//            if (!newEntry.getSecond().contains(lookupFragmentProperty))
//                newEntry.getSecond().add(lookupFragmentProperty);
//        }
//    }

//    public void printRelevantOriginalContigProperties(Map<String, Pair<List<FragmentProperty>, List<FragmentProperty>>> splitFragments) {
//        for (String key : splitFragments.keySet()) {
//            Pair<List<FragmentProperty>, List<FragmentProperty>> pair = splitFragments.get(key);
//            System.out.println("Old fragments");
//            for (FragmentProperty fragmentProperty : pair.getFirst()) {
//                System.out.println(fragmentProperty.getName());
//            }
//            System.out.println("New fragments");
//            for (FragmentProperty fragmentProperty : pair.getSecond()) {
//                System.out.println(fragmentProperty.getName());
//            }
//            System.out.println();
//        }
//    }

//    public void fixInitialState(Map<String, Pair<List<FragmentProperty>, List<FragmentProperty>>> newSplitFragments) {
//        Comparator<FragmentProperty> comparator = new Comparator<FragmentProperty>() {
//            @Override
//            public int compare(final FragmentProperty o1, final FragmentProperty o2) {
//                return o1.getFragmentNumber() - o2.getFragmentNumber();
//            }
//        };
//        for (String key : newSplitFragments.keySet()) {
//            //reconstruct from here
//            Pair<List<FragmentProperty>, List<FragmentProperty>> entry = newSplitFragments.get(key);
//
////            System.out.println(key);
////            Collections.sort(entry.getFirst(),comparator);
////            Collections.sort(entry.getSecond(),comparator);
////
////            List arrayList = new ArrayList<>();
////            for(FragmentProperty contigProperty : entry.getFirst()){
////                arrayList.add(contigProperty.getName());
////            }
////            System.out.println(arrayList);
////            arrayList.clear();
////            for(FragmentProperty contigProperty : entry.getSecond()){
////                arrayList.add(contigProperty.getName());
////            }
////            System.out.println(arrayList);
//
//
//            List<FragmentProperty> splitList = entry.getSecond();
//            List<FragmentProperty> originalContigList = entry.getFirst();
//            Collections.sort(splitList, comparator);
//            Collections.sort(originalContigList, comparator);
//
//
//            if (splitList.size() == originalContigList.size()) {
//                for (int i = 0; i < splitList.size(); i++) {
//                    setInitialStatesBasedOnOriginalContig(originalContigList.get(i), splitList.subList(i, i + 1), originalContigList.get(i).wasInitiallyInverted());
//                }
//            } else {
//                int index = 0;
//                for (FragmentProperty originalFragmentProperty : originalContigList) {
//                    List<FragmentProperty> neededFragments = new ArrayList<>();
//                    int sumLength = 0;
//                    long length = originalFragmentProperty.getLength();
//                    while (sumLength < length) {
//                        neededFragments.add(splitList.get(index));
//
//                        System.out.println(splitList.get(index).getName()+" "+splitList.get(index).getFragmentNumber()+" "+splitList.get(index).getLength());
//
//                        sumLength += splitList.get(index).getLength();
//                        index++;
//                    }
//
//                    if (originalFragmentProperty.wasInitiallyInverted()) {  //not sure if needed yet depends
//                        Collections.reverse(neededFragments);
//                    }
//                    setInitialStatesBasedOnOriginalContig(originalFragmentProperty, neededFragments, originalFragmentProperty.wasInitiallyInverted());
//                }
//            }
//        }
//    }

    //    public FragmentProperty feature2DtoContigProperty(Feature2D feature2D) {
//        for (FragmentProperty fragmentProperty : contigProperties) {
//            if (fragmentProperty.getFeature2D().getStart1() == feature2D.getStart1())
//                return fragmentProperty;
//        }
//        System.err.println("error finding corresponding contig");
//        return null;
//    }

//    public List<FragmentProperty> splitContig(boolean invertedInAsm, Feature2D originalFeature, Feature2D debrisFeature, FragmentProperty originalContig) {
//
//
//        if (originalFeature.overlapsWith(debrisFeature)) {
//            if (!invertedInAsm)  //not inverted
//                return generateNormalSplit(originalFeature, debrisFeature, originalContig);
//            else  //is inverted so flip order of contigs
//                return generateInvertedSplit(originalFeature, debrisFeature, originalContig);
//        } else {
//            System.out.println("error splitting contigs");
//            return null;
//        }
//    }

//    public List<FragmentProperty> generateNormalSplit(Feature2D originalFeature, Feature2D debrisFeature, FragmentProperty originalContig) {
//        List<FragmentProperty> splitContig = new ArrayList<>();
//        List<String> newContigNames = getNewContigNames(originalContig);
//
//        int originalIndexId = originalContig.getIndexId();
//        int originalStart = originalFeature.getStart1();
//        int debrisStart = debrisFeature.getStart1();
//        int originalEnd = originalFeature.getEnd2();
//        int debrisEnd = debrisFeature.getEnd1();
//        int length;
//        boolean initiallyInverted = originalContig.wasInitiallyInverted();
//
//        length = debrisStart - originalStart;
//        FragmentProperty firstFragment = new FragmentProperty(newContigNames.get(0), originalIndexId, length, initiallyInverted);
//
//        length = debrisEnd - debrisStart;
//        FragmentProperty secondFragment = new FragmentProperty(newContigNames.get(1), (originalIndexId + 1), length, initiallyInverted);
//
//        length = originalEnd - debrisEnd;
//        FragmentProperty thirdFragment = new FragmentProperty(newContigNames.get(2), (originalIndexId + 2), length, initiallyInverted);
//
//        splitContig.add(firstFragment);
//        splitContig.add(secondFragment);
//        splitContig.add(thirdFragment);
//
//        setInitialStatesBasedOnOriginalContig(originalContig, splitContig, false);
//        return splitContig;
//    }

//    public List<FragmentProperty> generateInvertedSplit(Feature2D originalFeature, Feature2D debrisFeature, FragmentProperty originalContig) {
//        List<FragmentProperty> splitContig = new ArrayList<>();
//        List<String> newContigNames = getNewContigNames(originalContig);
//
//        int originalIndexId = originalContig.getIndexId();
//        int originalStart = originalFeature.getStart1();
//        int debrisStart = debrisFeature.getStart1();
//        int originalEnd = originalFeature.getEnd2();
//        int debrisEnd = debrisFeature.getEnd1();
//        int length;
//        boolean initiallyInverted = originalContig.wasInitiallyInverted();
//
//        length = originalEnd - debrisEnd;
//        FragmentProperty firstFragment = new FragmentProperty(newContigNames.get(0), (originalIndexId + 2), length, initiallyInverted);
//
//        length = debrisEnd - debrisStart;
//        FragmentProperty secondFragment = new FragmentProperty(newContigNames.get(1), (originalIndexId + 1), length, initiallyInverted);
//
//        length = debrisStart - originalStart;
//        FragmentProperty thirdFragment = new FragmentProperty(newContigNames.get(2), originalIndexId, length, initiallyInverted);
//
//        splitContig.add(thirdFragment);
//        splitContig.add(secondFragment);
//        splitContig.add(firstFragment);
//
//        setInitialStatesBasedOnOriginalContig(originalContig, splitContig, true);
//        return splitContig;
//    }

//    public void setInitialStatesBasedOnOriginalContig(FragmentProperty originalContig, List<FragmentProperty> splitContig, boolean invertedInAsm) {
//        long newInitialStart;
//        long newInitialEnd;
//        boolean initiallyInverted = originalContig.wasInitiallyInverted();
//
//        if (invertedInAsm && !initiallyInverted || !invertedInAsm && initiallyInverted) { //inverted in map
//            newInitialEnd = originalContig.getInitialEnd();
//            newInitialStart = newInitialEnd;
//            for (FragmentProperty fragmentProperty : splitContig) {
//                //    50-100, 40-50, 0-40
//                newInitialStart = newInitialStart - fragmentProperty.getLength();
//                fragmentProperty.setInitialState(originalContig.getInitialChr(), newInitialStart, newInitialEnd, originalContig.isInvertedVsInitial());
//
//                newInitialEnd = newInitialStart;
//            }
//        } else { //not inverted in map
//            newInitialStart = originalContig.getInitialStart();
//            newInitialEnd = newInitialStart;
//            //    0-40, 40-50, 50-100
//            for (FragmentProperty fragmentProperty : splitContig) {
//                newInitialEnd = newInitialEnd + fragmentProperty.getLength();
//                fragmentProperty.setInitialState(originalContig.getInitialChr(), newInitialStart, newInitialEnd, originalContig.isInvertedVsInitial());
//                newInitialStart = newInitialEnd;
//            }
//        }
//    }

//    public List<String> getNewContigNames(FragmentProperty contigProperty) {
//        String fragmentString = ":::fragment_";
//        String contigName = contigProperty.getName();
//        List<String> newNames = new ArrayList<String>();
//        if (contigName.contains(fragmentString)) {
//            if (contigName.contains(":::debris")) {
//                System.err.println("cannot split a debris fragment");
//            } else {
//                String originalContigName = contigProperty.getOriginalContigName();
//                int fragmentNum = contigProperty.getFragmentNumber();
//                newNames.add(originalContigName + fragmentString + fragmentNum);
//                newNames.add(originalContigName + fragmentString + (fragmentNum + 1) + ":::debris");
//                newNames.add(originalContigName + fragmentString + (fragmentNum + 2));
//            }
//        } else {
//            newNames.add(contigName + fragmentString + "1");
//            newNames.add(contigName + fragmentString + "2:::debris");
//            newNames.add(contigName + fragmentString + "3");
//        }
//        return newNames;
//    }

//    public void addScaffoldProperties(int splitIndex, boolean invertedInAsm, List<FragmentProperty> splitContigs, int rowNum, int posNum) {
//        List<Integer> splitContigsIds = new ArrayList<>();
//        int multiplier;
//        if (invertedInAsm)
//            multiplier = -1;
//        else
//            multiplier = 1;
//
//        for (FragmentProperty contigProperty : splitContigs) {
//            splitContigsIds.add(multiplier * contigProperty.getIndexId());
//
//        }
//
//        shiftScaffoldProperties(splitIndex);
//
//        scaffoldProperties.get(rowNum).addAll(posNum, splitContigsIds);
//        scaffoldProperties.get(rowNum).remove(posNum + 3);
//    }

//    public void shiftScaffoldProperties(int splitIndex) {
//        int i;
//
//        for (List<Integer> scaffoldRow : scaffoldProperties) {
//            i = 0;
//            for (Integer indexId : scaffoldRow) {
//                if (Math.abs(indexId) > (Math.abs(splitIndex))) {
//                    if (indexId < 0)
//                        scaffoldRow.set(i, indexId - 2);
//                    else
//                        scaffoldRow.set(i, indexId + 2);
//                }
//                i++;
//            }
//        }
//    }

//    public void addContigProperties(FragmentProperty originalContig, List<FragmentProperty> splitContig) {
//        int splitContigIndex = contigProperties.indexOf(originalContig);
//        shiftContigIndices(splitContigIndex);
//
//        List<FragmentProperty> fromInitialContig = findContigsSplitFromInitial(originalContig);
//        if (fromInitialContig.indexOf(originalContig) != fromInitialContig.size() - 1) { //if there are fragments past the one you are splitting
//            List<FragmentProperty> shiftedContigs = fromInitialContig.subList(fromInitialContig.indexOf(originalContig) + 1, fromInitialContig.size());
//            for (FragmentProperty contigProperty : shiftedContigs) {
//                String newContigName = contigProperty.getName();
//                if (newContigName.contains(":::debris")) {
//                    newContigName = newContigName.replaceFirst(":::fragment_.*", ":::fragment_" + (contigProperty.getFragmentNumber() + 2) + ":::debris");
//                } else {
//                    newContigName = newContigName.replaceFirst(":::fragment_\\d+", ":::fragment_" + (contigProperty.getFragmentNumber() + 2));
//                }
//                contigProperty.setName(newContigName);
//            }
//        }
//
//        contigProperties.addAll(splitContigIndex, splitContig);
//        contigProperties.remove(originalContig);
//    }

//    public void shiftContigIndices(int splitIndexId) {
//        for (FragmentProperty contigProperty : contigProperties) {
//            if (Math.abs(contigProperty.getIndexId()) > (Math.abs(splitIndexId) + 1)) {
//                contigProperty.setIndexId(contigProperty.getIndexId() + 2);
//            }
//        }
//    }

//    public List<FragmentProperty> findContigsSplitFromInitial(FragmentProperty originalContig) {
//        List<FragmentProperty> contigPropertiesFromSameInitial = new ArrayList<>();
//        String originalContigName = originalContig.getOriginalContigName();
//
//        for (FragmentProperty contigProperty : contigProperties) {
//            if (contigProperty.getName().contains(originalContigName))
//                contigPropertiesFromSameInitial.add(contigProperty);
//        }
//        return contigPropertiesFromSameInitial;
//    }

//    public int getScaffoldRow(List<Integer> contigIds) {
//        int i = 0;
//        for (List<Integer> scaffoldRow : scaffoldProperties) {
//            List<Integer> absoluteRow = findAbsoluteValuesList(scaffoldRow);
//            if (absoluteRow.containsAll(findAbsoluteValuesList(contigIds)))
//                return i;
//            i++;
//        }
//        System.err.println("Can't Find row");
//        return -1;
//    }


}