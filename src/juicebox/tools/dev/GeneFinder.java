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

package juicebox.tools.dev;

import juicebox.data.HiCFileTools;
import juicebox.tools.clt.CommandLineParserForJuicer;
import juicebox.tools.clt.JuicerCLT;
import org.broad.igv.feature.Chromosome;

import java.util.List;

/**
 * Created by muhammadsaadshamim on 8/3/16.
 */
public class GeneFinder extends JuicerCLT {

    private String genomeID, bedFilePath, loopList;

    protected GeneFinder() {
        super("genes <genomeID> <bed_file> <looplist>");
    }

    @Override
    protected void readJuicerArguments(String[] args, CommandLineParserForJuicer juicerParser) {
        int i = 1;
        genomeID = args[i++];
        bedFilePath = args[i++];
        loopList = args[i++];
    }

    @Override
    public void run() {
        List<Chromosome> chromosomes = HiCFileTools.loadChromosomes(genomeID);


        //GenomeWideList<Locus> bedFilePositions = MotifAnchorParser.loadFromBEDFile(chromosomes, bedFilePath);
        //GenomeWideList<GeneLocation> genes = GeneTools.parseGenome(genomeID, handler);

    }
}
