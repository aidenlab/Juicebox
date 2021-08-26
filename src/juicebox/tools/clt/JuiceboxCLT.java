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

package juicebox.tools.clt;

import juicebox.data.Dataset;
import juicebox.data.HiCFileTools;
import juicebox.data.iterator.IteratorContainer;
import juicebox.windowui.NormalizationType;

import java.util.Arrays;

/**
 * All command line tools should extend from this class
 */
public abstract class JuiceboxCLT {

    private static String usage;
    protected Dataset dataset = null;
    protected NormalizationType norm = null;
    protected static int numCPUThreads = 1;
    protected static int numCPUThreadsForSecondTask = 1;
    protected boolean usingMultiThreadedVersion = false;

    protected JuiceboxCLT(String usage) {
        setUsage(usage);
    }

    public static String[] splitToList(String nextLine) {
        return nextLine.trim().split("\\s+");
    }

    public abstract void readArguments(String[] args, CommandLineParser parser);

    public abstract void run();

    private void setUsage(String newUsage) {
        usage = newUsage;
    }

    public void printUsageAndExit() {
        System.out.println("Usage:   juicer_tools " + usage);
        System.exit(0);
    }

    public void printUsageAndExit(int exitcode) {
        System.out.println("Usage:   juicer_tools " + usage);
        System.exit(exitcode);
    }

    protected void setDatasetAndNorm(String files, String normType, boolean allowPrinting) {
        dataset = HiCFileTools.extractDatasetForCLT(Arrays.asList(files.split("\\+")), allowPrinting);

        norm = dataset.getNormalizationHandler().getNormTypeFromString(normType);
        if (norm == null) {
            System.err.println("Normalization type " + norm + " unrecognized.  Normalization type must be one of \n" +
                    "\"NONE\", \"VC\", \"VC_SQRT\", \"KR\", \"GW_KR\"," +
                    " \"GW_VC\", \"INTER_KR\", \"INTER_VC\", or a custom added normalization.");
            System.exit(16);
        }
    }

    public static int getAppropriateNumberOfThreads(int numThreads, int defaultNum) {
        if (numThreads > 0) {
            return numThreads;
        } else if (numThreads < 0) {
            return Math.abs(numThreads) * Runtime.getRuntime().availableProcessors();
        } else {
            return defaultNum;
        }
    }

    protected void updateNumberOfCPUThreads(CommandLineParser parser, int numDefaultThreads) {
        int numThreads = parser.getNumThreads();
        numCPUThreads = getAppropriateNumberOfThreads(numThreads, numDefaultThreads);
        System.out.println("Using " + numCPUThreads + " CPU thread(s) for primary task");
    }

    protected void updateSecondaryNumberOfCPUThreads(CommandLineParser parser, int numDefaultThreads) {
        int numMThreads = parser.getNumMatrixOperationThreads();
        numCPUThreadsForSecondTask = getAppropriateNumberOfThreads(numMThreads, numDefaultThreads);
        System.out.println("Using " + IteratorContainer.numCPUMatrixThreads + " CPU thread(s) for secondary task");
    }
}

