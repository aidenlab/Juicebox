/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2015 Broad Institute, Aiden Lab
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

import java.io.IOException;
import java.io.PrintWriter;
import java.net.Socket;

//import java.io.InputStreamReader;
//import java.net.UnknownHostException;
//import java.util.ArrayList;
//import java.util.List;

/**
 * @author jrobinso
 *         Date: 10/21/13
 *         Time: 2:59 PM
 */
class CommandBroadcaster {

    public static int selfPort;

    public static void broadcast(String command) {
        // Broadcast self port to other running instances
        for (int p = 30000; p <= 30009; p++) {
            if (p == selfPort) continue;  // don't broadcast to self
            try {
                CommandBroadcaster.broadcastCommand(command, p);
            } catch (java.net.ConnectException e) {
                // Expected
            } catch (IOException e) {
                e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
            }
        }
    }

    private static void broadcastCommand(String command, int port) throws IOException {
        Socket socket = null;
        PrintWriter out = null;
        try {
            socket = new Socket("127.0.0.1", port);
            out = new PrintWriter(socket.getOutputStream(), true);
            out.println(command);

        } finally {
            try {
                if (out != null) out.close();
                if (socket != null) socket.close();
            } catch (IOException e) {
                e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
            }
        }
    }

}
