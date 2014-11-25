package juicebox;

//import java.io.BufferedReader;
import java.io.IOException;
//import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
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
            if(p == selfPort) continue;  // don't broadcast to self
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

        }  finally {
            try {
                if(out != null) out.close();
                if(socket != null) socket.close();
            } catch (IOException e) {
                e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
            }
        }
    }

}
