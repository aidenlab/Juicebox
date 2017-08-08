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

import juicebox.track.feature.Feature2DList;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Created by ranganmostofa on 6/29/17.
 */
public class AssemblyFileImporter {
    private String cpropsFilePath;
    private String asmFilePath;
    private List<ContigProperty> contigProperties;
    private List<List<Integer>> scaffoldProperties;
    private AssemblyFragmentHandler assemblyFragmentHandler;

    public AssemblyFileImporter(String cpropsFilePath, String asmFilePath) {
        this.cpropsFilePath = cpropsFilePath;
        this.asmFilePath = asmFilePath;
        contigProperties = new ArrayList<>();
        scaffoldProperties = new ArrayList<>();
        readFiles();
        assemblyFragmentHandler = new AssemblyFragmentHandler(contigProperties, scaffoldProperties);
        assemblyFragmentHandler.generateContigsAndScaffolds(true, false, assemblyFragmentHandler);
    }

    public void readFiles() {
        try {
            parseAsmFile();
            parseCpropsFile();
        } catch (IOException exception) {
            System.err.println("Error reading files!");
        }
    }

    private void parseCpropsFile() throws IOException {
        if (validateCpropsFile()) {
            List<String> rawFileData = readFile(cpropsFilePath);

            for (String row : rawFileData) {
                String[] splitRow = row.split(" ");
                // splitRow[0] -> Name, splitRow[2] -> length

                boolean initiallyInverted = false;
                for (List<Integer> scaffoldRow : scaffoldProperties) {
                    for (int element : scaffoldRow) {
                        if (Math.abs(element) == Math.abs(Integer.parseInt(splitRow[1]))) { //can make
                            if (Math.abs(element) != element) { //if negative
                                initiallyInverted = true;
                            }
                            break;
                        }
                    }
                }
                ContigProperty currentPair = new ContigProperty(splitRow[0], Integer.parseInt(splitRow[1]), Integer.parseInt(splitRow[2]), initiallyInverted);
                contigProperties.add(currentPair);
            }
        } else System.out.println("Invalid cprops file");
    }

    private boolean validateCpropsFile() {
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

                scaffoldProperties.add(currentContigIndices);
            }
        } else
            System.out.println("Invalid asm file");
    }

    private boolean validateAsmFile() {
        return getAsmFilePath().endsWith(FILE_EXTENSIONS.ASM.toString());
    }

    private boolean getIsInverted(Integer contigIndex) {
        return contigIndex < 0 ? Boolean.TRUE : Boolean.FALSE;
    }

    public void buildContigAttributes(String contigName, Integer contigLength) {
        Map<String, String> featureAttributes = new HashMap<>();
//        featureAttributes.put("Scaffold_ID", );
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

    private void setCpropsFilePath(String cpropsFilePath) {
        this.cpropsFilePath = cpropsFilePath;
    }

    private String getAsmFilePath() {
        return this.asmFilePath;
    }

    private void setAsmFilePath(String asmFilePath) {
        this.asmFilePath = asmFilePath;
    }

    public Feature2DList getContigs() {
        return this.assemblyFragmentHandler.getContigs();
    }

    public Feature2DList getScaffolds() {
        return this.assemblyFragmentHandler.getScaffolds();
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
