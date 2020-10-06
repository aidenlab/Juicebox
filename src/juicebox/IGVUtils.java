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

package juicebox;

import juicebox.data.ChromosomeHandler;
import org.broad.igv.ui.IGV;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

//import org.broad.igv.lists.GeneList;
//import org.broad.igv.ui.WaitCursorManager;
//import org.broad.igv.util.LongRunningTask;
//import org.broad.igv.util.NamedRunnable;
//import java.net.UnknownHostException;

/**
 * @author Jim Robinson
 * @since 1/5/12
 */
class IGVUtils {

    private static final ExecutorService threadExecutor = Executors.newFixedThreadPool(1);

    private static SocketHelper helper = null;

    private static void createSocketHelper() {
        if (helper == null) {
            Socket socket;
            PrintWriter out;
            BufferedReader in;
            try {
                socket = new Socket("127.0.0.1", 60151);
                out = new PrintWriter(socket.getOutputStream(), true);
                in = new BufferedReader(new InputStreamReader(socket.getInputStream()), HiCGlobals.bufferSize);
                helper = new SocketHelper(in, out, socket);
            } catch (IOException e) {
                System.err.println("IOException" + e.getLocalizedMessage());
                helper = null;
            }

        }
    }

    /**
     * Send instructions to IGV to open or adjust views on the 2 loci.
     *
     * @param locus1
     * @param locus2
     */
    public static void sendToIGV(final String locus1, final String locus2) {

        Runnable runnable = new Runnable() {
            public void run() {

                if (ChromosomeHandler.isAllByAll(locus1)) {
                    return;
                }

                // Same JVM?
                if (IGV.hasInstance()) {
                    IGV.getInstance().goToLociList(Arrays.asList(locus1, locus2));

                } else {

                    if (helper == null) createSocketHelper();

                    if (helper != null) {
                        String cmd = "gotoimmediate " + locus1 + " " + locus2;
                        helper.out.println(cmd);
                        String response = null;
                        try {
                            response = helper.in.readLine();
                        } catch (IOException e) {
                            e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
                        }
                        System.out.println(cmd + " " + response);
                    }
                }
            }
        };

        threadExecutor.submit(runnable);

    }

    static class SocketHelper {
        Socket socket = null;
        PrintWriter out = null;
        BufferedReader in = null;

        SocketHelper(BufferedReader in, PrintWriter out, Socket socket) {
            this.in = in;
            this.out = out;
            this.socket = socket;
        }
    }

}
