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

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Created by nchernia on 12/23/15.
 */
public class ProcessHelper {


    public ProcessHelper() {


    }

    public Process startNewJavaProcess()
            throws IOException {

        ProcessBuilder processBuilder = createProcess();
        processBuilder.redirectErrorStream(true); // redirect error stream to output stream
        processBuilder.redirectOutput(ProcessBuilder.Redirect.INHERIT);
        return processBuilder.start();
    }


    private ProcessBuilder createProcess() {
        String jvm = System.getProperty("java.home") + File.separator + "bin" + File.separator + "java";
        String classpath = System.getProperty("java.class.path");
        //System.err.println("classpath: " + classpath);
        // String workingDirectory = System.getProperty("user.dir");

        // String[] options = optionsAsString.split(" ");
        List<String> command = new ArrayList<>();
        command.add(jvm);

        String url = System.getProperty("jnlp.loadMenu");
        if (url != null) {
            System.err.println(url);
            command.add("-Djnlp.loadMenu=" + url);
            //     String[] options = {"-Xmx2000m", "-Djnlp.loadMenu=" + url};
            //     command.addAll(Arrays.asList(options));
        }
        command.add("-Xmx5000m");

        // command.addAll(Arrays.asList(options));
        command.add(MainWindow.class.getCanonicalName());
        //  command.addAll(Arrays.asList(arguments));

        ProcessBuilder processBuilder = new ProcessBuilder(command);
        Map<String, String> environment = processBuilder.environment();
        environment.put("CLASSPATH", classpath);
        return processBuilder;
    }

}
