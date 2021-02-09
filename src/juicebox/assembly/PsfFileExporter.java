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

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Iterator;
import java.util.List;

/**
 * Created by dudcha on 10/27/20.
 */
public class PsfFileExporter {

    private final String outputFilePath;
    private final List<Scaffold> listOfScaffolds;
    private final List<List<Integer>> listOfSuperscaffolds;

    public PsfFileExporter(AssemblyScaffoldHandler assemblyScaffoldHandler, String outputFilePath) {
        this.outputFilePath = outputFilePath;
        this.listOfScaffolds = assemblyScaffoldHandler.getListOfScaffolds();
        this.listOfSuperscaffolds = assemblyScaffoldHandler.getListOfSuperscaffolds();
    }

    public void exportPsfFile() {
        try {
            exportPsf();
        } catch (IOException exception) {
            System.out.println("Exporting failed...");
        }
    }

    private void exportPsf() throws IOException {
        PrintWriter assemblyWriter = new PrintWriter(buildPsfOutputPath(), "UTF-8");
        String name = "";
        for (Scaffold scaffold : listOfScaffolds) {
            int id = scaffold.getIndexId();

            if (id % 2 == 0) {
                String[] splitName = scaffold.getName().split(":");
                assemblyWriter.print(">" + name + " " + splitName[2] + " " + (id / 2) + "\n");
                name = "";
            } else {
                name = scaffold.getName().replace(":", " ");
            }
        }
        for (int i = 0; i < listOfSuperscaffolds.size(); i++) {
            if (i % 2 != 0) {
                continue;
            }
            List<Integer> row = listOfSuperscaffolds.get(i);
            assemblyWriter.print(superscaffoldToString(row) + "\n");
        }
        assemblyWriter.close();
    }

    private String superscaffoldToString(List<Integer> scaffoldRow) {
        StringBuilder stringBuilder = new StringBuilder();
        Iterator<Integer> iterator = scaffoldRow.iterator();
        while (iterator.hasNext()) {
            int i = iterator.next();
            if (i % 2 == 0) {
                i = -i / 2;
            } else {
                i = (i + 1) / 2;
            }
            stringBuilder.append(i);
            if (iterator.hasNext()) {
                stringBuilder.append(" ");
            }
        }
        return stringBuilder.toString();
    }

    private String buildPsfOutputPath() {
        return this.outputFilePath + ".psf";
    }

}
