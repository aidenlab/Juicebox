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

package juicebox.tools.clt;

import juicebox.tools.HiCDBUtils;
import juicebox.tools.HiCTools;
import java.io.IOException;
import java.sql.SQLException;

/**
 * Created by muhammadsaadshamim on 1/21/15.
 */
public class SQLDatabase extends JuiceboxCLT {

    String[] dbArgs;

    @Override
    public void readArguments(String[] args, HiCTools.CommandLineParser parser) throws IOException {
        setUsage("juicebox db <frag|annot|update> [items]");
        dbArgs = new String[args.length - 1];
        System.arraycopy(args, 1, dbArgs, 0, args.length - 1);
    }

    @Override
    public void run() throws IOException{

        try {
            HiCDBUtils.main(dbArgs);
        } catch (SQLException e) {
            System.err.println("Sql exception: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
        catch (IOException e){
            e.printStackTrace();
            throw new IOException("-1");
        }
    }
}
