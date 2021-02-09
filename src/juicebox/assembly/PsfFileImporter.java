/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab
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
import juicebox.gui.SuperAdapter;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * Created by dudcha on 10/19/20.
 */
public class PsfFileImporter extends AssemblyFileImporter {

    private SuperAdapter superAdapter = null;
    // single file format
    private String psfFilePath;
    private boolean modified = false;

    private List<Scaffold> listOfScaffolds;
    private List<List<Integer>> listOfSuperscaffolds;
    private AssemblyScaffoldHandler assemblyScaffoldHandler;

    private List<String> rawFileData;

    public PsfFileImporter(String psfFilePath, boolean modified) {
        this.psfFilePath = psfFilePath;
        this.modified = modified;
    }

    @Override
    public void importAssembly() {
        importPsf();
    }

    public void importPsf() {

        listOfScaffolds = new ArrayList<>();
        listOfSuperscaffolds = new ArrayList<>();
        try {
            if (psfFilePath != null) {
                rawFileData = readFile(psfFilePath);
                parsePsfFile();
            }
            if (!modified)
                setInitialState();
            else
                setModifiedInitialState();
        } catch (IOException exception) {
            System.err.println("Error reading files!");
        }
        updateAssemblyScale();

        assemblyScaffoldHandler = new AssemblyScaffoldHandler(listOfScaffolds, listOfSuperscaffolds);
    }

    private int updateAssemblyScale() {
        long totalLength = 0;
        for (Scaffold fragmentProperty : listOfScaffolds) {
            totalLength += fragmentProperty.getLength();
        }
        HiCGlobals.hicMapScale = (int) (1 + totalLength / 2100000000);
        return (int) (totalLength / HiCGlobals.hicMapScale); // in case decide to use for validation so that not to count again
    }

    private void parsePsfFile() throws IOException {
        try {
            for (String row : rawFileData) {
                if (row.startsWith(">")) {
                    String[] splitRow = row.split(" ");
                    // Chr Pos Ref_var Alt_var ID
                    int id = Integer.parseInt(splitRow[4]);
                    Scaffold scaffold = new Scaffold(splitRow[0].substring(1) + ":" + splitRow[1] + ":" + splitRow[2], 2 * id - 1, 1000);
                    listOfScaffolds.add(scaffold);
                    scaffold = new Scaffold(splitRow[0].substring(1) + ":" + splitRow[1] + ":" + splitRow[3], 2 * id, 1000);
                    listOfScaffolds.add(scaffold);
                } else {
                    List<Integer> superscaffold = new ArrayList<>();
                    List<Integer> altSuperscaffold = new ArrayList<>();

                    for (String index : row.split(" ")) {
                        int id = Integer.parseInt(index);
                        if (id > 0) {
                            superscaffold.add(2 * id - 1);
                            altSuperscaffold.add(2 * id);
                        } else {
                            altSuperscaffold.add(-2 * id - 1);
                            superscaffold.add(-2 * id);
                        }
                    }
                    listOfSuperscaffolds.add(superscaffold);
                    listOfSuperscaffolds.add(altSuperscaffold);
                }
            }
        } catch (NumberFormatException e) {
            e.printStackTrace();
            System.err.println("Errors in format");
        }
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
        for (int i = 0; i < listOfScaffolds.size(); i++) {
            listOfScaffolds.get(i).setOriginallyInverted(false);
            listOfScaffolds.get(i).setOriginalStart(originalScaffolds.get(i).getOriginalStart());
        }
    }


    List<String> readFile(String filePath) throws IOException {
        List<String> fileData = new ArrayList<>();

        File file = new File(filePath);

        Scanner scanner = new Scanner(file);

        while (scanner.hasNext()) {
            fileData.add(scanner.nextLine());
        }
        return fileData;
    }

    public AssemblyScaffoldHandler getAssemblyScaffoldHandler() {
        return assemblyScaffoldHandler;
    }

}
