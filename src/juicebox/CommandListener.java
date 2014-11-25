/*
 * Copyright (c) 2007-2011 by The Broad Institute of MIT and Harvard.  All Rights Reserved.
 *
 * This software is licensed under the terms of the GNU Lesser General Public License (LGPL),
 * Version 2.1 which is available at http://www.opensource.org/licenses/lgpl-2.1.php.
 *
 * THE SOFTWARE IS PROVIDED "AS IS." THE BROAD AND MIT MAKE NO REPRESENTATIONS OR
 * WARRANTES OF ANY KIND CONCERNING THE SOFTWARE, EXPRESS OR IMPLIED, INCLUDING,
 * WITHOUT LIMITATION, WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE, NONINFRINGEMENT, OR THE ABSENCE OF LATENT OR OTHER DEFECTS, WHETHER
 * OR NOT DISCOVERABLE.  IN NO EVENT SHALL THE BROAD OR MIT, OR THEIR RESPECTIVE
 * TRUSTEES, DIRECTORS, OFFICERS, EMPLOYEES, AND AFFILIATES BE LIABLE FOR ANY DAMAGES
 * OF ANY KIND, INCLUDING, WITHOUT LIMITATION, INCIDENTAL OR CONSEQUENTIAL DAMAGES,
 * ECONOMIC DAMAGES OR INJURY TO PROPERTY AND LOST PROFITS, REGARDLESS OF WHETHER
 * THE BROAD OR MIT SHALL BE ADVISED, SHALL HAVE OTHER REASON TO KNOW, OR IN FACT
 * SHALL KNOW OF THE POSSIBILITY OF THE FOREGOING.
 */
package juicebox;

import org.apache.log4j.Logger;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.net.Socket;

public class CommandListener implements Runnable {

    private static final Logger log = Logger.getLogger(CommandListener.class);

    private static CommandListener listener;

    private int port = -1;
    private ServerSocket serverSocket = null;
    private Socket clientSocket = null;
    private final Thread listenerThread;
    private boolean halt = false;
    private final HiC hic;

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

    private CommandListener(int port, HiC hic) throws IOException {
        this.port = port;
        this.hic = hic;
        listenerThread = new Thread(this);
        serverSocket = new ServerSocket(port);
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
            in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
            String cmd;
            while (!halt && (cmd = in.readLine()) != null) {
                if(cmd.toLowerCase().equals("halt")) {
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
