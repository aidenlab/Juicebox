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

import juicebox.tools.AsciiToBinConverter;
import juicebox.tools.HiCTools;
import org.broad.igv.feature.Chromosome;

import java.io.IOException;
import java.util.List;

public class PairsToBin extends JuiceboxCLT {

    private String ifile, ofile, genomeId;

    @Override
    public void readArguments(String[] args, HiCTools.CommandLineParser parser) throws IOException {
        //setUsage("juicebox pairsToBin <infile> <outfile> <genomeID>");
        if (args.length != 4) {
            throw new IOException("1");
        }
        ifile = args[1];
        ofile = args[2];
        genomeId = args[3];
    }

    @Override
    public void run() throws IOException {
        List<Chromosome> chromosomes = CommonTools.loadChromosomes(genomeId);
        AsciiToBinConverter.convert(ifile, ofile, chromosomes);
    }
}
