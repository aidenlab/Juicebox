/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2016 Broad Institute, Aiden Lab
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

import org.apache.log4j.Logger;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.net.Socket;

class CommandListener implements Runnable {

    private static final Logger log = Logger.getLogger(CommandListener.class);

    private static CommandListener listener;
    private final Thread listenerThread;
    private final HiC hic;
    private int port = -1;
    private ServerSocket serverSocket = null;
    private Socket clientSocket = null;
    private boolean halt = false;

    private CommandListener(int port, HiC hic) throws IOException {
        this.port = port;
        this.hic = hic;
        listenerThread = new Thread(this);
        serverSocket = new ServerSocket(port);
    }

    public static synchronized void start(HiC hic) {

        // Grab the first available port
        int port = 0;
        for (int p = 30000; p <= 30009; p++) {
            try {
                listener = new CommandListener(p, hic);
                listener.listenerThread.start();
                port = p;
                break;
            } catch (IOException e) {
                // Expected condition -- port in use
            }
        }
        CommandBroadcaster.selfPort = port;
    }

    public static synchronized void halt() {
        if (listener != null) {
            listener.halt = true;
            listener.listenerThread.interrupt();
            listener.closeSockets();
            listener = null;
        }
    }

    /**
     * Loop forever, processing client requests synchronously.  The server is single threaded.
     */
    public void run() {

        CommandExecutor cmdExe = new CommandExecutor(hic);
        log.info("Listening on port " + port);

        try {
            while (!halt) {
                clientSocket = serverSocket.accept();
                processClientRequest(cmdExe);
                if (clientSocket != null) {
                    try {
                        clientSocket.close();
                        clientSocket = null;
                    } catch (IOException e) {
                        log.error("Error in client socket loop", e);
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
        }

    }

    /**
     * Process a client request
     *
     * @param cmdExe
     * @throws IOException
     */
    private void processClientRequest(CommandExecutor cmdExe) throws IOException {

        BufferedReader in = null;

        try {
            in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()), HiCGlobals.bufferSize);
            String cmd;
            while (!halt && (cmd = in.readLine()) != null) {
                if (cmd.toLowerCase().equals("halt")) {
                    halt = true;
                    return;
                }
                cmdExe.execute(cmd);
            }
        } catch (IOException e) {
            log.error("Error processing client session", e);
        } finally {
            if (in != null) in.close();
        }
    }

    private void closeSockets() {
        if (clientSocket != null) {
            try {
                clientSocket.close();
                clientSocket = null;
            } catch (IOException e) {
                log.error("Error closing clientSocket", e);
            }
        }

        if (serverSocket != null) {
            try {
                serverSocket.close();
                serverSocket = null;
            } catch (IOException e) {
                log.error("Error closing server socket", e);
            }
        }
    }

}
