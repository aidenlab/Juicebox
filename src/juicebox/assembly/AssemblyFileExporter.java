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

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Iterator;
import java.util.List;

/**
 * Created by nathanielmusial on 6/29/17.
 */
public class AssemblyFileExporter {

    private final String outputFilePath;
    private final List<Scaffold> listOfScaffolds;
    private final List<List<Integer>> listOfSuperscaffolds;
    private final List<String> listOfBundledScaffolds;


    public AssemblyFileExporter(AssemblyScaffoldHandler assemblyScaffoldHandler, String outputFilePath) {
        this.outputFilePath = outputFilePath;
        this.listOfScaffolds = assemblyScaffoldHandler.getListOfScaffolds();
        this.listOfSuperscaffolds = assemblyScaffoldHandler.getListOfSuperscaffolds();
        this.listOfBundledScaffolds = assemblyScaffoldHandler.getListOfBundledScaffolds();
    }

    public void exportAssemblyFile() {
        try {
            exportAssembly();
        } catch (IOException exception) {
            System.out.println("Exporting failed...");
        }
    }

    private void exportAssembly() throws IOException {
        PrintWriter assemblyWriter = new PrintWriter(buildAssemblyOutputPath(), "UTF-8");
        int last = 0;
        for (Scaffold scaffold : listOfScaffolds) {
            if (scaffold.getName().equals("unattempted:::debris")) {
                continue;
            }
            assemblyWriter.print(">" + scaffold.toString() + "\n"); // Use print to account for OS difference in control characters
            last = scaffold.getIndexId();
        }

        if (listOfBundledScaffolds.size() > 0) {
            for (String row : listOfBundledScaffolds) {
                String[] splitRow = row.split(" ");
                last += 1;
                assemblyWriter.print(splitRow[0] + " " + last + " " + splitRow[2] + "\n");
            }
        }
        for (List<Integer> row : listOfSuperscaffolds) {
            if (listOfBundledScaffolds.size() > 0 && row.get(0) == listOfScaffolds.size()) {
                continue;
            }
            assemblyWriter.print(superscaffoldToString(row) + "\n");
        }
        if (listOfBundledScaffolds.size() > 0) {
            for (int i = listOfScaffolds.size(); i <= last; i++)
                assemblyWriter.print(i + "\n");
        }
        assemblyWriter.close();
    }

    private String superscaffoldToString(List<Integer> scaffoldRow) {
        StringBuilder stringBuilder = new StringBuilder();
        Iterator<Integer> iterator = scaffoldRow.iterator();
        while (iterator.hasNext()) {
            stringBuilder.append(iterator.next());
            if (iterator.hasNext()) {
                stringBuilder.append(" ");
            }
        }
        return stringBuilder.toString();
    }

    private String buildAssemblyOutputPath() {
        return this.outputFilePath + ".assembly";
    }

}
