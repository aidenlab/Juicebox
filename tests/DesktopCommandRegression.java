package juicebox;

import java.io.PrintWriter;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.net.ServerSocket;
import java.net.Socket;

public class DesktopCommandRegression {
    private static void check(boolean value, String message) {
        if (!value) throw new AssertionError(message);
    }

    public static void main(String[] args) throws Exception {
        org.broad.igv.DirectoryManager.setIgvDirectory(java.nio.file.Files.createTempDirectory("desktop-command-").toFile());
        int[] calls = {0};
        HiC hic = new HiC(null) {
            @Override public void setLocation(String x, String y, Unit unit, int bin, double originX,
                    double originY, double scale, ZoomCallType type, String message, boolean broadcast) {
                calls[0]++;
            }
        };
        CommandExecutor executor = new CommandExecutor(hic);
        check(executor.execute("setlocation chr1 chr1 BP 1000 0 0 1").equals("OK") && calls[0] == 1,
                "valid sync command");
        for (String command : new String[]{"setlocation chr1 chr1 bogus 1000 0 0 1",
                "setlocation chr1 chr1 BP 1000 NaN 0 1", "setlocation chr1 chr1 BP 0 0 0 1",
                "setlocation chr1 chr1 BP 1000 0 0 Infinity"}) {
            check(executor.execute(command).startsWith("Error:"), "invalid command rejected");
        }
        check(calls[0] == 1, "invalid sync commands cannot change model");
        check(executor.execute("bogus").equals("Unknown command string"), "unknown command response");

        Constructor<CommandListener> constructor = CommandListener.class.getDeclaredConstructor(int.class, HiC.class);
        constructor.setAccessible(true);
        Field server = CommandListener.class.getDeclaredField("serverSocket");
        Field threadField = CommandListener.class.getDeclaredField("listenerThread");
        Field singleton = CommandListener.class.getDeclaredField("listener");
        server.setAccessible(true);
        threadField.setAccessible(true);
        singleton.setAccessible(true);
        for (boolean remote : new boolean[]{true, false}) {
            CommandListener listener = constructor.newInstance(0, hic);
            ServerSocket socket = (ServerSocket) server.get(listener);
            Thread thread = (Thread) threadField.get(listener);
            singleton.set(null, listener);
            thread.start();
            try {
                if (remote) {
                    try (Socket client = new Socket("127.0.0.1", socket.getLocalPort());
                            PrintWriter out = new PrintWriter(client.getOutputStream(), true)) {
                        out.println("halt");
                    }
                } else {
                    CommandListener.halt();
                }
                thread.join(3000);
                check(!thread.isAlive() && socket.isClosed(), "listener halt releases socket and thread");
            } finally {
                CommandListener.halt();
                thread.join(3000);
            }
        }
        System.out.println("Desktop commands: validation and local/remote listener shutdown passed");
    }
}
