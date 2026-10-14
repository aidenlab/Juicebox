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

package juicebox;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.net.Socket;

class CommandListener implements Runnable {

    private static CommandListener listener;
    private final Thread listenerThread;
    private final HiC hic;
    private int port = -1;
    private final ServerSocket serverSocket;
    private volatile Socket clientSocket = null;
    private volatile boolean halt = false;

    private CommandListener(int port, HiC hic) throws IOException {
        this.port = port;
        this.hic = hic;
        listenerThread = new Thread(this);
        serverSocket = new ServerSocket(port);
    }

    public static synchronized void start(HiC hic) {
        if (listener != null && !listener.halt) return;

        // Grab the first available port
        int port = 0;
        for (int p = 30000; p <= 30000 + CommandBroadcaster.numPorts - 1; p++) {
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
        System.out.println("Listening on port " + port);

        try {
            while (!halt) {
                Socket socket = serverSocket.accept();
                clientSocket = socket;
                if (halt) break;
                socket.setSoTimeout(5000);
                processClientRequest(cmdExe);
                closeClientSocket();
            }
        } catch (IOException e) {
            if (!halt) e.printStackTrace();
        } finally {
            closeSockets();
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
        Socket socket = clientSocket;
        if (socket == null) return;

        try {
            in = new BufferedReader(new InputStreamReader(socket.getInputStream()), HiCGlobals.bufferSize);
            String cmd;
            while (!halt && (cmd = in.readLine()) != null) {
                if (cmd.equalsIgnoreCase("halt")) {
                    halt = true;
                    return;
                }
                cmdExe.execute(cmd);
            }
        } catch (IOException e) {
            if (!halt) System.err.println("Error processing client session" + e.getLocalizedMessage());
        } finally {
            if (in != null) in.close();
        }
    }

    private void closeSockets() {
        closeClientSocket();
        try {
            serverSocket.close();
        } catch (IOException e) {
            System.err.println("Error closing server socket" + e.getLocalizedMessage());
        }
    }

    private void closeClientSocket() {
        Socket socket = clientSocket;
        if (socket != null) {
            try {
                socket.close();
            } catch (IOException e) {
                System.err.println("Error closing clientSocket" + e.getLocalizedMessage());
            } finally {
                if (clientSocket == socket) clientSocket = null;
            }
        }
    }

}
