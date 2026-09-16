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

package juicebox.tools.clt.old;


import juicebox.tools.clt.CommandLineParser;
import juicebox.tools.clt.JuiceboxCLT;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.text.NumberFormat;
import java.text.ParseException;
import java.util.Locale;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.io.File;

public class LibraryComplexity extends JuiceboxCLT {

    private String localWorkingDirectory;
    private long dupReadPairs = 0;
    private long uniqueReadPairs = 0;
    private long opticalDups = 0;
    private long totalReadPairs = 0;
    private String fileName = "inter.txt";
    private final boolean filesAlreadyCounted;

    public LibraryComplexity(String command) {
        super(getUsage());
        filesAlreadyCounted = command.contains("fast");
    }

    private static String getUsage() {
        return "LibraryComplexity <directory> <output file>\n" +
                "FastLibraryComplexity <file> <output file>\n" +
                "FastLibraryComplexity <unique> <pcr> <opt>";
    }

    /**
     * Estimates the size of a library based on the number of paired end molecules
     * observed and the number of unique pairs observed.
     * <br>
     * Based on the Lander-Waterman equation that states:<br>
     * C/X = 1 - exp( -N/X )<br>
     * where<br>
     * X = number of distinct molecules in library<br>
     * N = number of read pairs<br>
     * C = number of distinct fragments observed in read pairs<br>
     */
    public static long estimateLibrarySize(final long duplicateReadPairs,
                                           final long uniqueReadPairs) {

        long totalReadPairs = duplicateReadPairs + uniqueReadPairs;
        if (totalReadPairs > 0 && duplicateReadPairs > 0) {

            double m = 1.0, M = 100.0;

            if (uniqueReadPairs >= totalReadPairs || f(m * uniqueReadPairs, uniqueReadPairs, totalReadPairs) < 0) {
                throw new IllegalStateException("Invalid values for pairs and unique pairs: " + totalReadPairs + ", " + uniqueReadPairs);
            }

            while (f(M * uniqueReadPairs, uniqueReadPairs, totalReadPairs) >= 0) {
                m = M;
                M *= 10.0;
            }

            double r = (m + M) / 2.0;
            double u = f(r * uniqueReadPairs, uniqueReadPairs, totalReadPairs);
            int i = 0;
            while (u != 0 && i < 1000) {
                if (u > 0) m = r;
                else M = r;
                r = (m + M) / 2.0;
                u = f(r * uniqueReadPairs, uniqueReadPairs, totalReadPairs);
                i++;
            }
            if (i == 1000) {
                System.err.println("Iterated 1000 times, returning estimate");
            }

            return (long) (uniqueReadPairs * (m + M) / 2.0);
        } else {
            return 0;
        }
    }

    /**
     * Method that is used in the computation of estimated library size.
     */
    private static double f(double x, double c, double n) {
        return c / x - 1 + Math.exp(-n / x);
    }

    @Override
    public void readArguments(String[] args, CommandLineParser parser) {
        if (args.length != 2 && args.length != 3 && args.length != 4) {
            System.out.println(getUsage());
            System.exit(0);
        }

        if (filesAlreadyCounted) {
            if (args.length == 4) {
                try {
                    determineCountsFromInput(args);
                } catch (NumberFormatException error) {
                    System.err.println("When called with three arguments, must be integers");
                    System.exit(1);
                }
            } else {
                try {
                    determineCountsFromFile(args[1]);
                    if (args.length == 3) fileName = args[2];
                } catch (Exception error) {
                    System.err.println(error.getLocalizedMessage());
                    System.exit(1);
                }
            }
        } else if (args.length == 3 || args.length == 2) {
            localWorkingDirectory = args[1];
            if (args.length == 3) fileName = args[2];
        }
    }

    private void determineCountsFromFile(String file) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(file));
        dupReadPairs = Long.parseLong(reader.readLine().trim());
        uniqueReadPairs = Long.parseLong(reader.readLine().trim());
        try {
            opticalDups = Long.parseLong(reader.readLine().trim());
        } catch (Exception e) {
            opticalDups = 0L;
        }
        reader.close();
    }

    private void determineCountsFromInput(String[] args) throws NumberFormatException {
        uniqueReadPairs = Long.parseLong(args[1]);
        dupReadPairs = Long.parseLong(args[2]);
        opticalDups = Long.parseLong(args[3]);
    }

    @Override
    public void run() {

        NumberFormat nf = NumberFormat.getInstance(Locale.US);

        if (!filesAlreadyCounted) {
            final AtomicBoolean somethingFailed = new AtomicBoolean(false);

            try {
                ExecutorService executor = Executors.newFixedThreadPool(3);

                Callable<Long> taskOptDups = () -> {
                    File f = new File(localWorkingDirectory + "/opt_dups.txt");
                    if (f.exists()) {
                        try {
                            long opticalDupsT = 0L;
                            BufferedReader reader = new BufferedReader(new FileReader(f));
                            while (reader.readLine() != null) opticalDupsT++;
                            reader.close();
                            return opticalDupsT;
                        } catch (Exception e) {
                            somethingFailed.set(true);
                            return 0L;
                        }
                    } else {
                        return 0L;
                    }
                };

                Callable<Long> taskUniqueReads = () -> {
                    File f = new File(localWorkingDirectory + "/merged_nodups.txt");
                    if (f.exists()) {
                        try {
                            long uniqueReadPairsT = 0L;
                            BufferedReader reader = new BufferedReader(new FileReader(f));
                            while (reader.readLine() != null) uniqueReadPairsT++;
                            reader.close();
                            return uniqueReadPairsT;
                        } catch (Exception e) {
                            somethingFailed.set(true);
                            return 0L;
                        }
                    } else {
                        return 0L;
                    }
                };

                Callable<Long> taskDupReadPairs = () -> {
                    File f = new File(localWorkingDirectory + "/dups.txt");
                    if (f.exists()) {
                        try {
                            long readPairsT = 0;
                            BufferedReader reader = new BufferedReader(new FileReader(localWorkingDirectory + "/dups.txt"));
                            while (reader.readLine() != null) readPairsT++;
                            reader.close();
                            return readPairsT;
                        } catch (Exception e) {
                            somethingFailed.set(true);
                            return 0L;
                        }
                    } else {
                        return 0L;
                    }
                };

                Future<Long> futureOptDups = executor.submit(taskOptDups);
                Future<Long> futureUniqueReads = executor.submit(taskUniqueReads);
                Future<Long> futureDupReadPairs = executor.submit(taskDupReadPairs);

                File f = new File(localWorkingDirectory + File.separator + fileName);
                if (f.exists()) {
                    BufferedReader reader = new BufferedReader(new FileReader(localWorkingDirectory + File.separator + fileName));
                    String line = reader.readLine();
                    boolean done = false;
                    while (line != null && !done) {
                        if (line.contains("Sequenced Read")) {
                            String[] parts = line.split(":");
                            try {
                                totalReadPairs = nf.parse(parts[1].trim()).longValue();
                            } catch (ParseException ignored) {
                            }
                            done = true;
                        }
                        line = reader.readLine();
                    }
                    reader.close();
                }

                opticalDups = futureOptDups.get();
                uniqueReadPairs = futureUniqueReads.get();
                dupReadPairs = futureDupReadPairs.get();
                executor.shutdown();

                if (somethingFailed.get()) {
                    System.err.println("Something failed in a thread");
                    System.exit(1);
                }

            } catch (IOException error) {
                System.err.println("Problem counting lines in merged_nodups and dups");
                System.exit(1);
            } catch (InterruptedException e) {
                System.err.println("Threads interrupted exception");
                System.exit(1);
            } catch (ExecutionException e) {
                System.err.println("Threads execution exception");
                System.exit(1);
            }
        }

        NumberFormat decimalFormat = NumberFormat.getPercentInstance();
        decimalFormat.setMinimumFractionDigits(2);
        decimalFormat.setMaximumFractionDigits(2);

        System.out.print("Unique Reads: " + NumberFormat.getInstance().format(uniqueReadPairs) + " ");
        if (totalReadPairs > 0) {
            System.out.println("(" + decimalFormat.format(uniqueReadPairs / (double) totalReadPairs) + ")");
        } else {
            System.out.println();
        }

        System.out.print("PCR Duplicates: " + nf.format(dupReadPairs) + " ");
        if (totalReadPairs > 0) {
            System.out.println("(" + decimalFormat.format(dupReadPairs / (double) totalReadPairs) + ")");
        } else {
            System.out.println();
        }
        System.out.print("Optical Duplicates: " + nf.format(opticalDups) + " ");
        if (totalReadPairs > 0) {
            System.out.println("(" + decimalFormat.format(opticalDups / (double) totalReadPairs) + ")");
        } else {
            System.out.println();
        }
        long result;
        try {
            result = estimateLibrarySize(dupReadPairs, uniqueReadPairs);
        } catch (NullPointerException e) {
            result = 0;
        }
        System.out.println("Library Complexity Estimate: " + nf.format(result));
    }
}
