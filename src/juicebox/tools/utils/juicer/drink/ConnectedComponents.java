/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2019 Broad Institute, Aiden Lab
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

package juicebox.tools.utils.juicer.drink;

import juicebox.data.feature.FeatureFilter;
import juicebox.data.feature.GenomeWideList;

import java.util.*;

class ConnectedComponents {

    public static int[][] generateAdjacencyMatrix(Map<SimpleInterval, Set<Integer>> intervalToClusterIDs) {
        int n = UniqueSubcompartmentClusterID.tempInitialClusterID.get();
        int[][] incidence = new int[n][n];

        for (Set<Integer> clusterIDs : intervalToClusterIDs.values()) {
            for (int i : clusterIDs) {
                for (int j : clusterIDs) {
                    incidence[i][j] += 1;
                    incidence[j][i] += 1;
                }
            }
        }
        return incidence;
    }

    public static Set<Set<Integer>> calculateConnectedComponents(int[][] adjacencyMatrix, int minNumContacts) {

        Set<Set<Integer>> setOfConnectedClusterIDs = new HashSet<>();

        int n = UniqueSubcompartmentClusterID.tempInitialClusterID.get();
        boolean[] nodeVisited = new boolean[n];

        for (int idx = 0; idx < n; idx++) {
            if (!nodeVisited[idx]) {
                Set<Integer> connectedClusterIDs = new HashSet<>();
                Queue<Integer> queue = new LinkedList<>();
                queue.add(idx);

                while (queue.size() > 0) {
                    int nextNode = queue.remove();
                    if (!nodeVisited[nextNode]) {
                        //System.out.print("."+nextNode);
                        processNextNode(nextNode, connectedClusterIDs, queue, adjacencyMatrix, nodeVisited, minNumContacts);
                    }
                }
                setOfConnectedClusterIDs.add(connectedClusterIDs);
            }
            //System.out.println();
        }

        /*
        1) Initialize all vertices as not visited.
        2) Do following for every vertex 'v'.
        (a) If 'v' is not visited before, call DFSUtil(v)
        (b) Print new line character

        DFSUtil(v)
        1) Mark 'v' as visited.
        2) Print 'v'
        3) Do following for every adjacent 'u' of 'v'.
                If 'u' is not visited, then recursively call DFSUtil(u)
                */
        System.out.println("Total num components " + setOfConnectedClusterIDs.size());

        int nSize = 2;
        int numLargeComponents = numberOfLargeComponents(setOfConnectedClusterIDs, nSize);

        System.out.println("Total num large (>" + nSize + ") components " + numLargeComponents);

        return setOfConnectedClusterIDs;
    }

    private static int numberOfLargeComponents(Set<Set<Integer>> setOfConnectedClusterIDs, int minSize) {
        int numLargeComponents = 0;
        for (Set<Integer> set : setOfConnectedClusterIDs) {
            if (set.size() > minSize) {
                numLargeComponents++;
            }
        }
        return numLargeComponents;
    }

    /**
     * Process for each node
     *
     * @param nextNode
     * @param connectedClusterIDs
     * @param queue
     * @param adjacencyMatrix
     * @param nodeVisited
     */
    private static void processNextNode(int nextNode, Set<Integer> connectedClusterIDs, Queue<Integer> queue,
                                        int[][] adjacencyMatrix, boolean[] nodeVisited, int minNumContacts) {
        // add to answer list
        connectedClusterIDs.add(nextNode);

        // has been visited
        nodeVisited[nextNode] = true;

        // add partners to processing list if not visited
        for (int k = 0; k < adjacencyMatrix[nextNode].length; k++) {
            if (adjacencyMatrix[nextNode][k] > minNumContacts && !nodeVisited[k]) {
                queue.add(k);
            }
        }
    }

    /**
     * creates adjacency contact matrix of cluster IDs which co-occur
     *
     * @param intervalToClusterIDs
     * @return
     */
    public static boolean[][] generateBooleanAdjacencyMatrix(Map<SimpleInterval, Set<Integer>> intervalToClusterIDs) {
        int n = UniqueSubcompartmentClusterID.tempInitialClusterID.get();
        boolean[][] incidence = new boolean[n][n];

        for (Set<Integer> clusterIDs : intervalToClusterIDs.values()) {
            for (int i : clusterIDs) {
                for (int j : clusterIDs) {
                    incidence[i][j] = true;
                    incidence[j][i] = true;
                }
            }
        }
        return incidence;
    }


    public static GenomeWideList<SubcompartmentInterval>
    stitchSubcompartments(Set<Set<Integer>> connectedComponents, GenomeWideList<SubcompartmentInterval> origIntraSubcompartments) {

        GenomeWideList<SubcompartmentInterval> finalList = origIntraSubcompartments.deepClone();

        final Map<Integer, Integer> tempClusterIdToUniqueClusterID = new HashMap<>();
        for (Set<Integer> connectedComponent : connectedComponents) {
            int uniqueClusterIDX = UniqueSubcompartmentClusterID.finalClusterID.getAndIncrement();
            for (Integer node : connectedComponent) {
                tempClusterIdToUniqueClusterID.put(node, uniqueClusterIDX);
            }
        }


        finalList.filterLists(new FeatureFilter<SubcompartmentInterval>() {
            @Override
            public List<SubcompartmentInterval> filter(String chr, List<SubcompartmentInterval> featureList) {

                List<SubcompartmentInterval> newList = new ArrayList<>(featureList.size());

                for (SubcompartmentInterval interval0 : featureList) {
                    newList.add(new SubcompartmentInterval(interval0.getChrIndex(), interval0.getChrName(), interval0.getX1(),
                            interval0.getX2(), tempClusterIdToUniqueClusterID.get(interval0.getClusterID())));
                }

                return newList;
            }
        });

        SubcompartmentInterval.reSort(finalList);

        SubcompartmentInterval.collapseGWList(finalList);

        return finalList;
    }
}
