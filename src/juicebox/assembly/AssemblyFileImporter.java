/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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
import juicebox.data.basics.Chromosome;
import juicebox.gui.SuperAdapter;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

/**
 * Created by ranganmostofa on 6/29/17.
 */
public class AssemblyFileImporter {

  private SuperAdapter superAdapter = null;
  // legacy format
    private String cpropsFilePath;
    private String asmFilePath;

    // single file format
    private String assemblyFilePath;

    private boolean modified = false;

    private List<Scaffold> listOfScaffolds;
    private List<List<Integer>> listOfSuperscaffolds;
    private AssemblyScaffoldHandler assemblyScaffoldHandler;

    public AssemblyFileImporter(String cpropsFilePath, String asmFilePath, boolean modified) {
        this.cpropsFilePath = cpropsFilePath;
        this.asmFilePath = asmFilePath;
        this.modified = modified;
    }

    public AssemblyFileImporter(String assemblyFilePath, boolean modified) {
        this.assemblyFilePath = assemblyFilePath;
        this.modified = modified;
    }

  public AssemblyFileImporter(SuperAdapter superAdapter) {
    this.superAdapter = superAdapter;
  }

  public void importAssembly() {
        listOfScaffolds = new ArrayList<>();
        listOfSuperscaffolds = new ArrayList<>();
        // does it update assembly? //
        try {
            if (assemblyFilePath != null) {
                parseAssemblyFile();
            } else if (cpropsFilePath != null && asmFilePath != null) {
              parseCpropsFile();
              parseAsmFile();
            } else {
              parseChromSizes();
            }
            if (!modified)
                setInitialState();
            else
                setModifiedInitialState();
        } catch (IOException exception) {
            System.err.println("Error reading files!");
        }
        updateAssemblyScale();

        // TODO: validateImport or user dialog
        assemblyScaffoldHandler = new AssemblyScaffoldHandler(listOfScaffolds, listOfSuperscaffolds);
    }

  private void parseChromSizes() {
    for (Chromosome chr : superAdapter.getHiC().getChromosomeHandler().getChromosomeArrayWithoutAllByAll()) {
      Scaffold scaffold = new Scaffold(chr.getName(), chr.getIndex(), chr.getLength());
      listOfScaffolds.add(scaffold);
      listOfSuperscaffolds.add(Arrays.asList(chr.getIndex()));
    }
  }

  private int updateAssemblyScale() {
        long totalLength = 0;
        for (Scaffold fragmentProperty : listOfScaffolds) {
            totalLength += fragmentProperty.getLength();
        }
        HiCGlobals.hicMapScale = (int) (1 + totalLength / 2100000000);
        return (int) (totalLength / HiCGlobals.hicMapScale); // in case decide to use for validation so that not to count again
    }

    private void parseAssemblyFile() throws IOException {
        List<String> rawFileData = readFile(assemblyFilePath);
        try {
            for (String row : rawFileData) {
                if (row.startsWith(">")) {
                    String[] splitRow = row.split(" ");
                    // Name<\s>ID<\s>length
                    Scaffold scaffold = new Scaffold(splitRow[0].substring(1), Integer.parseInt(splitRow[1]), Integer.parseInt(splitRow[2]));
                    listOfScaffolds.add(scaffold);
                } else {
                    List<Integer> superscaffold = new ArrayList<>();
                    for (String index : row.split(" ")) {
                        superscaffold.add(Integer.parseInt(index));
                    }
                    listOfSuperscaffolds.add(superscaffold);
                }
            }
        } catch (NumberFormatException e) {
            e.printStackTrace();
            System.err.println("Errors in format");
        }
    }

    private void parseCpropsFile() throws IOException {
        if (validateCpropsFile()) {
            List<String> rawFileData = readFile(cpropsFilePath);

            for (String row : rawFileData) {
                String[] splitRow = row.split(" ");
                // Name<\s>ID<\s>length
                Scaffold scaffold = new Scaffold(splitRow[0], Integer.parseInt(splitRow[1]), Integer.parseInt(splitRow[2]));
                listOfScaffolds.add(scaffold);
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
                List<Integer> superscaffold = new ArrayList<>();
                for (String index : row.split(" ")) {
                    superscaffold.add(Integer.parseInt(index));
                }

                listOfSuperscaffolds.add(superscaffold);
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
        for (List<Integer> row : listOfSuperscaffolds) {
            for (Integer entry : row) {
                int fragmentIterator = Math.abs(entry) - 1;
                listOfScaffolds.get(fragmentIterator).setOriginallyInverted(false);
                if (entry < 0) {
                    listOfScaffolds.get(fragmentIterator).setOriginallyInverted(true);
                } else if (entry == 0) {
                    System.err.println("Something is wrong with the input."); // should not happen
                }
                listOfScaffolds.get(fragmentIterator).setOriginalStart(shift);
                shift += listOfScaffolds.get(fragmentIterator).getLength();
            }
        }
    }

    private void setModifiedInitialState() {
        List<Scaffold> originalScaffolds = AssemblyHeatmapHandler.getSuperAdapter().getAssemblyStateTracker().getInitialAssemblyScaffoldHandler().getListOfScaffolds();
        long modifiedShift = 0;
        int originalScaffoldIterator = 0;
        Scaffold originalScaffold = originalScaffolds.get(originalScaffoldIterator);
        long containingStart = originalScaffold.getOriginalStart();
        long containingEnd = originalScaffold.getOriginalEnd();
        for (Scaffold modifiedScaffold : listOfScaffolds) {

            modifiedScaffold.setOriginallyInverted(originalScaffold.getOriginallyInverted());
            if (!modifiedScaffold.getOriginallyInverted()) {
                modifiedScaffold.setOriginalStart(containingStart);
                containingStart += modifiedScaffold.getLength();
            } else {
                modifiedScaffold.setOriginalStart(containingEnd - modifiedScaffold.getLength());
                containingEnd -= modifiedScaffold.getLength();
            }
            // trace movement along the original feature
            modifiedShift += modifiedScaffold.getLength();
            // check if need to switch to next original feature
            if (modifiedShift == originalScaffold.getLength()) {
                if (originalScaffoldIterator == originalScaffolds.size() - 1) {
                    if (modifiedScaffold != listOfScaffolds.get(listOfScaffolds.size() - 1)) {
                        System.err.println("Modified assembly incompatible with the original one.");
                    }
                    break;
                }
                originalScaffoldIterator++;
                originalScaffold = originalScaffolds.get(originalScaffoldIterator);
                containingStart = originalScaffold.getOriginalStart();
                containingEnd = originalScaffold.getOriginalEnd();
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

    public AssemblyScaffoldHandler getAssemblyScaffoldHandler() {
        return assemblyScaffoldHandler;
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
