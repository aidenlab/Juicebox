/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2022 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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

package juicebox.tools.utils.original.merge.merger;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class GraphsMerger extends Merger {

    private final long[] A = new long[2000];
    private final long[][] B = new long[3][200];
    private final long[][] D = new long[4][100];
    private final long[] x = new long[100];

    @Override
    public void parse(String s) {

        Scanner scanner = new Scanner(s);
        try {
            skipUntilNextArray(scanner);
            addTo1DArray(A, scanner);

            skipUntilNextArray(scanner);
            addTo2DArray(B, scanner);

            skipUntilNextArray(scanner);
            addTo2DArray(D, scanner);

            skipUntilNextArray(scanner);
            for (int idx = 0; idx < x.length; idx++) {
                long newX = scanner.nextLong();
                if (x[idx] > 0L) {
                    if (x[idx] != newX) {
                        System.err.println("X mismatch error? " + x[idx] + " - " + newX);
                    }
                }
                x[idx] = newX;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void printToMergedFile(String filename) {
        try {
            BufferedWriter histWriter = new BufferedWriter(new FileWriter(filename));
            histWriter.write("A = [\n");
            write1DArray(A, histWriter);
            histWriter.write("\n];\n");

            histWriter.write("B = [\n");
            write2DArray(B, histWriter);
            histWriter.write("\n];\n");

            histWriter.write("D = [\n");
            write2DArray(D, histWriter);
            histWriter.write("\n];");

            histWriter.write("x = [\n");
            write1DArray(x, histWriter);
            histWriter.write("\n];\n");
            histWriter.close();
        } catch (IOException error) {
            error.printStackTrace();
        }
    }

    private void addTo1DArray(long[] arr, Scanner scanner) {
        for (int idx = 0; idx < arr.length; idx++) {
            arr[idx] += scanner.nextLong();
        }
    }

    private void write1DArray(long[] arr, BufferedWriter histWriter) throws IOException {
        for (long tmp : arr) {
            histWriter.write(tmp + " ");
        }
    }

    private void addTo2DArray(long[][] arr, Scanner scanner) {
        int numRows = arr.length;
        int len = arr[0].length;
        for (int idx = 0; idx < len; idx++) {
            for (int r = 0; r < numRows; r++) {
                try {
                    arr[r][idx] += scanner.nextLong();
                } catch (Exception e) {
                    System.err.println(scanner.next());
                    e.printStackTrace();
                }
            }
        }
    }

    private void write2DArray(long[][] arr, BufferedWriter histWriter) throws IOException {
        int numRows = arr.length;
        int len = arr[0].length;
        for (int idx = 0; idx < len; idx++) {
            StringBuilder s = new StringBuilder("" + arr[0][idx]);
            for (int r = 1; r < numRows; r++) {
                s.append(" ").append(arr[r][idx]);
            }
            histWriter.write(s + "\n");
        }
    }

    private void skipUntilNextArray(Scanner scanner) {
        while (!scanner.next().equals("[")) ;
    }
}
