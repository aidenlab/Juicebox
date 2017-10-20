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
import juicebox.track.feature.Feature2DList;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * Created by ranganmostofa on 6/29/17.
 */
public class AssemblyFileImporter {
    private String cpropsFilePath;
    private String asmFilePath;
    private List<FragmentProperty> fragmentProperties;
    private List<List<Integer>> assemblyGroups;
    private AssemblyFragmentHandler assemblyFragmentHandler;

    // Deprecated
//    public AssemblyFileImporter(String cpropsFilePath, String asmFilePath) {
//        this.cpropsFilePath = cpropsFilePath;
//        this.asmFilePath = asmFilePath;
//        fragmentProperties = new ArrayList<>();
//        assemblyGroups = new ArrayList<>();
//        readFiles();
//        assemblyFragmentHandler = new AssemblyFragmentHandler(fragmentProperties, assemblyGroups);
//        assemblyFragmentHandler.generateContigsAndScaffolds(true, false, assemblyFragmentHandler);
//    }

    public AssemblyFileImporter(String cpropsFilePath, String asmFilePath, boolean modified) {
        this.cpropsFilePath = cpropsFilePath;
        this.asmFilePath = asmFilePath;
        fragmentProperties = new ArrayList<>();
        assemblyGroups = new ArrayList<>();
        try {
            newParseCpropsFile();
            parseAsmFile();
            if (!modified)
                setInitialState();
            else
                setModifiedInitialState();
        } catch (IOException exception) {
            System.err.println("Error reading files!");
        }
        updateAssemblyScale();
        assemblyFragmentHandler = new AssemblyFragmentHandler(fragmentProperties, assemblyGroups);
    }

    public void updateAssemblyScale() {
        long totalLength = 0;
        for (FragmentProperty fragmentProperty : fragmentProperties) {
            totalLength += fragmentProperty.getLength();
        }
        HiCGlobals.hicMapScale = (int) (1 + totalLength / 2100000000);
        System.out.println(HiCGlobals.hicMapScale);
    }

    private void newParseCpropsFile() throws IOException {
        if (validateCpropsFile()) {
            List<String> rawFileData = readFile(cpropsFilePath);

            for (String row : rawFileData) {
                String[] splitRow = row.split(" ");
                // Name<\s>ID<\s>length
                FragmentProperty fragmentProperty = new FragmentProperty(splitRow[0], Integer.parseInt(splitRow[1]), Integer.parseInt(splitRow[2]));
                fragmentProperties.add(fragmentProperty);
            }
        } else System.out.println("Invalid cprops file");
    }

    private boolean validateCpropsFile() {
        //TODO: more restrictions for user-proofness
        return getCpropsFilePath().endsWith(FILE_EXTENSIONS.CPROPS.toString());
    }

    private void parseAsmFile() throws IOException {
        if (validateAsmFile()) {
            List<String> rawFileData = readFile(asmFilePath);

            for (String row : rawFileData) {
                //   System.out.println("Scaffold: "+row);
                List<Integer> currentContigIndices = new ArrayList<>();
                for (String index : row.split(" ")) {
                    currentContigIndices.add(Integer.parseInt(index));
                }

                assemblyGroups.add(currentContigIndices);
            }
        } else
            System.out.println("Invalid asm file");
    }

    private boolean validateAsmFile() {
        //TODO: more restrictions for user-proofness
        return getAsmFilePath().endsWith(FILE_EXTENSIONS.ASM.toString());
    }

    private void setInitialState() {
        long shift = 0;
        for (List<Integer> group : assemblyGroups) {
            for (Integer entry : group) {
                int fragmentIterator = Math.abs(entry) - 1;
                fragmentProperties.get(fragmentIterator).setInitiallyInverted(false);
                if (entry < 0) {
                    fragmentProperties.get(fragmentIterator).setInitiallyInverted(true);
                } else if (entry == 0) {
                    System.err.println("Something is wrong with the input."); // should not happen
                }
                fragmentProperties.get(fragmentIterator).setInitialStart(shift);
                shift += fragmentProperties.get(fragmentIterator).getLength();
            }
        }
    }

    private void setModifiedInitialState() {
        List<FragmentProperty> originalFragmentProperties = AssemblyHeatmapHandler.getSuperAdapter().getAssemblyStateTracker().getInitialAssemblyFragmentHandler().getListOfScaffoldProperties();
        long modifiedShift = 0;
        int originalFragmentIterator = 0;
        FragmentProperty originalFragmentProperty = originalFragmentProperties.get(originalFragmentIterator);
        long containingStart = originalFragmentProperty.getInitialStart();
        long containingEnd = originalFragmentProperty.getInitialEnd();
        for (FragmentProperty modifiedFragmentProperty : fragmentProperties) {

            modifiedFragmentProperty.setInitiallyInverted(originalFragmentProperty.wasInitiallyInverted());
            if (!modifiedFragmentProperty.wasInitiallyInverted()) {
                modifiedFragmentProperty.setInitialStart(containingStart);
                containingStart += modifiedFragmentProperty.getLength();
            } else {
                modifiedFragmentProperty.setInitialStart(containingEnd - modifiedFragmentProperty.getLength());
                containingEnd -= modifiedFragmentProperty.getLength();
            }
            // trace movement along the original feature
            modifiedShift += modifiedFragmentProperty.getLength();
            // check if need to switch to next original feature
            if (modifiedShift == originalFragmentProperty.getLength()) {
                if (originalFragmentIterator == originalFragmentProperties.size() - 1) {
                    if (modifiedFragmentProperty != fragmentProperties.get(fragmentProperties.size() - 1)) {
                        System.err.println("Modified assembly incompatible with the original one.");
                    }
                    break;
                }
                originalFragmentIterator++;
                originalFragmentProperty = originalFragmentProperties.get(originalFragmentIterator);
                containingStart = originalFragmentProperty.getInitialStart();
                containingEnd = originalFragmentProperty.getInitialEnd();
                modifiedShift = 0;
            }
        }
        //TODO: more safeguards e.g. by name
    }


    private List<String> readFile(String filePath) throws IOException {
        List<String> fileData = new ArrayList<>();

        File file = new File(filePath);
        Scanner scanner = new Scanner(file);

        while (scanner.hasNext()) {
            fileData.add(scanner.nextLine());
        }

        return fileData;
    }

    private String getCpropsFilePath() {
        return this.cpropsFilePath;
    }

    private String getAsmFilePath() {
        return this.asmFilePath;
    }

    public Feature2DList getContigs() {
        return this.assemblyFragmentHandler.getScaffoldFeature2DList();
    } //why do we have this here?

    public Feature2DList getScaffolds() {
        return this.assemblyFragmentHandler.getSuperscaffoldFeature2DList(); //why do we have this here?
    }

    public AssemblyFragmentHandler getAssemblyFragmentHandler() {
        return assemblyFragmentHandler;
    }

    private enum FILE_EXTENSIONS {
        CPROPS("cprops"),
        ASM("asm");

        private final String extension;

        FILE_EXTENSIONS(String extension) {
            this.extension = extension;
        }

        public boolean equals(String otherExtension) {
            return this.extension.equals(otherExtension);
        }

        public String toString() {
            return this.extension;
        }
    }
}
