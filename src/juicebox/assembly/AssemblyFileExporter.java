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

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Iterator;
import java.util.List;

/**
 * Created by nathanielmusial on 6/29/17.
 */
public class AssemblyFileExporter {

    private AssemblyFragmentHandler assemblyFragmentHandler;
    private String outputFilePath;
    private List<FragmentProperty> contigProperties;
    private List<List<Integer>> scaffoldProperties;

    public AssemblyFileExporter(AssemblyFragmentHandler assemblyFragmentHandler, String outputFilePath) {
        this.assemblyFragmentHandler = assemblyFragmentHandler;
        this.outputFilePath = outputFilePath;
        this.contigProperties = assemblyFragmentHandler.getListOfScaffoldProperties();
        this.scaffoldProperties = assemblyFragmentHandler.getListOfSuperscaffolds();
    }

    public void exportContigsAndScaffolds() {
        try {
            exportContigs();
            exportScaffolds();
        } catch (IOException exception) {
            System.out.println("Exporting failed...");
        }
    }

    private void exportContigs() throws IOException {
        PrintWriter contigPrintWriter = new PrintWriter(buildCpropsOutputPath(), "UTF-8");
        for (FragmentProperty fragmentProperty : contigProperties) {
            contigPrintWriter.println(fragmentProperty.toString());
        }
        contigPrintWriter.close();
    }

    private void exportScaffolds() throws IOException {
        PrintWriter scaffoldPrintWriter = new PrintWriter(buildAsmOutputPath(), "UTF-8");
        for (List<Integer> row : scaffoldProperties) {
            scaffoldPrintWriter.println(convertScaffoldRowToString(row));
        }
        scaffoldPrintWriter.close();
    }

    private String convertScaffoldRowToString(List<Integer> scaffoldRow) {
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

    private String buildCpropsOutputPath() {
        return this.outputFilePath + "." + FILE_EXTENSIONS.CPROPS.toString();
    }

    private String buildAsmOutputPath() {
        return this.outputFilePath + "." + FILE_EXTENSIONS.ASM.toString();
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
