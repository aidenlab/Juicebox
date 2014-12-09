/*
 * Copyright (C) 2011-2014 Aiden Lab - All Rights Reserved
 * SOFTWARE COPYRIGHT NOTICE
 * This software and its documentation are the copyright of the Aiden Lab All rights are reserved.
 *
 * This software is supplied without any warranty or guaranteed support whatsoever. Aiden Lab is not responsible for its use, misuse, or functionality.
 *
 * This software is licensed under the terms of the GNU Lesser General Public License (LGPL),
 * Version 2.1 which is available at http://www.opensource.org/licenses/lgpl-2.1.php.
 */
package juicebox;

//import java.io.BufferedReader;

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
public class CommandBroadcaster {

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
