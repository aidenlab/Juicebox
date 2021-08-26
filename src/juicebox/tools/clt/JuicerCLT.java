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

package juicebox.tools.clt;

import juicebox.data.ChromosomeHandler;
import juicebox.data.Dataset;
import juicebox.data.Matrix;
import juicebox.data.basics.Chromosome;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationHandler;
import juicebox.windowui.NormalizationType;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by muhammadsaadshamim on 9/21/15.
 */
public abstract class JuicerCLT extends JuiceboxCLT {

    protected NormalizationType norm = NormalizationHandler.SCALE;
    protected List<String> givenChromosomes = null; //TODO set to protected

    protected JuicerCLT(String usage) {
        super(usage);
    }

    protected int determineHowManyChromosomesWillActuallyRun(Dataset ds, ChromosomeHandler chromosomeHandler, HiCZoom zoom) {
        int maxProgressStatus = 0;
        for (Chromosome chr : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {
            Matrix matrix = ds.getMatrix(chr, chr);
            if (matrix == null) continue;
            if (matrix.getZoomData(zoom) == null) continue;
            maxProgressStatus++;
        }
        return maxProgressStatus;
    }

    @Override
    public void readArguments(String[] args, CommandLineParser parser) {
        CommandLineParserForJuicer juicerParser = (CommandLineParserForJuicer)parser;
        assessIfChromosomesHaveBeenSpecified(juicerParser);
        readJuicerArguments(args, juicerParser);
    }

    protected abstract void readJuicerArguments(String[] args, CommandLineParserForJuicer juicerParser);

    private void assessIfChromosomesHaveBeenSpecified(CommandLineParserForJuicer juicerParser) {
        List<String> possibleChromosomes = juicerParser.getChromosomeListOption();
        if (possibleChromosomes != null && possibleChromosomes.size() > 0) {
            givenChromosomes = new ArrayList<>(possibleChromosomes);
        }
    }
}
