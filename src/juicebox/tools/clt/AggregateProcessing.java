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

import jargs.gnu.CmdLineParser;
import juicebox.tools.HiCTools;

import java.io.IOException;

/**
 * Created for testing multiple CLTs at once
 * Basically scratch space
 */
class AggregateProcessing {


    public static void main(String[] argv) throws IOException, CmdLineParser.UnknownOptionException, CmdLineParser.IllegalOptionValueException {
        String[] l4 = {"arrowhead", "-c", "22", "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
                "/Users/muhammadsaadshamim/Desktop/j3/out8/", "25000"};

        String[] l42 = {"arrowhead", "-m", "5000", "-c", "assembly",
                "http://adam.bcma.bcm.edu/temp/assembly/human_stringent_OnOByhg19_curated.hic",
                "/Users/muhammadsaadshamim/Desktop/j3/out8/", "100000"};


        String[] l5 = {"dump", "observed", "KR", "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
                "22", "22", "BP", "25000", "/Users/muhammadsaadshamim/Desktop/BioScripts/22_blocks"};

        String[] l6 = {"apa", "-r", "5000", "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
                "/Users/muhammadsaadshamim/Downloads/APA_Ivan/gmloops.txt.gz",
                "/Users/muhammadsaadshamim/Downloads/APA_Ivan/temp3"};

        String[] l51 = {"dump", "observed", "KR", "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
                "22", "22", "BP", "5000", "/Users/muhammadsaadshamim/Desktop/j3/1212chr22_5000_aa.bin"};

        String[] l52 = {"dump", "observed", "KR", "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
                "22", "22", "BP", "10000", "/Users/muhammadsaadshamim/Desktop/j3/out9/chr22_10000.bin"};

        String[] l53 = {"dump", "observed", "KR", "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
                "22", "22", "BP", "25000", "/Users/muhammadsaadshamim/Desktop/j3/out9/chr22_25000.bin"};

        String[] l61 = {"dump", "expected", "KR", "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
                "22", "22", "BP", "5000", "/Users/muhammadsaadshamim/Desktop/j3/out9/exp22_5000"};

        String[] l62 = {"dump", "expected", "KR", "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
                "22", "22", "BP", "10000", "/Users/muhammadsaadshamim/Desktop/j3/out9/exp22_10000"};

        String[] l63 = {"dump", "expected", "KR", "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
                "22", "22", "BP", "25000", "/Users/muhammadsaadshamim/Desktop/j3/out9/exp22_25000"};

        String[] l71 = {"dump", "norm", "KR", "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
                "22", "22", "BP", "5000", "/Users/muhammadsaadshamim/Desktop/j3/out9/norm22_5000"};

        String[] l72 = {"dump", "norm", "KR", "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
                "22", "22", "BP", "10000", "/Users/muhammadsaadshamim/Desktop/j3/out9/norm22_10000"};

        //
        String[] l73 = {"afa", "-r", "25000", "-w", "60", "-a", ",Common,Rare,Aphidicolin,FolicAcid", "-l", "BottomRight", "KR",
                "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
                "/Users/muhammadsaadshamim/Desktop/RandomScripts3/Muhammad/GlassDragon/cfss_cancer_associated_rearranged.px2",
                "/Users/muhammadsaadshamim/Desktop/RandomScripts3/Muhammad/GlassDragon/G9/"};

        //afa [-w window]  [-r resolution(s)] [-c chromosomes] [-a attribute(s)] [-l TopLeft/BottomRight/Center]
        // <NONE/VC/VC_SQRT/KR> <hic file(s)> <FeatureFile> <SaveFolder> [SavePrefix]

        String[] l7 = {"hiccups", "-c", "22", "-m", "90", "-r", "10000,5000",
                "-f", ".1,.1", "-p", "4,2", "-w", "7,5", "-d", "20000,20000",
                "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
                "/Users/muhammadsaadshamim/Desktop/j3/out10/loops0"};

        String[] l111 = new String[]{"motifs", "-g", "/Applications/bedtools2",
                "/Users/muhammadsaadshamim/Desktop/suhas_motif/bed/CTCF_ENCFF001SUB.bed.gz", "hg19",
                "/Users/muhammadsaadshamim/Desktop/suhas_motif/loops",
                "/Users/muhammadsaadshamim/Desktop/suhas_motif/temp"};

        String[] l112 = new String[]{"motifs",
                "/Users/muhammadsaadshamim/Desktop/cleanRunMotifFinderCopy/bed/CTCF_ENCFF001SUB.bed", "hg19",
                "/Users/muhammadsaadshamim/Desktop/cleanRunMotifFinderCopy/looplist.txt",
                "/Users/muhammadsaadshamim/Desktop/cleanRunMotifFinderCopy/temp2"};

        // https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic
        //

        String[] l1132 = new String[]{"arrowhead", "-m", "2000",
                "/Users/muhammadsaadshamim/Desktop/juicer/inter.hic",
                "/Users/muhammadsaadshamim/Desktop/juicer/arrowhead_contact_domains", "10000"};

        String[] l11323 = new String[]{"hiccups", "-m", "90", "-c", "15,16", "-r", "10000",
                "/Users/muhammadsaadshamim/Desktop/juicer/inter.hic",
                "/Users/muhammadsaadshamim/Desktop/juicer/hiccups_loops_list"};

        // hiccups -m 500 -r 10000 ./aligned/inter.hic ./aligned/hiccups_loops_list

        // https://hicfiles.s3.amazonaws.com/hiseq/ch12-lx-b-lymphoblasts/in-situ/combined.hic

        String[] ll1132 = new String[]{"arrowhead", "-m", "2000",
                "https://hicfiles.s3.amazonaws.com/hiseq/ch12-lx-b-lymphoblasts/in-situ/combined.hic",
                "/Users/muhammadsaadshamim/Desktop/juicer/ch12lx_arrowhead_contact_domains", "10000"};

        String[] ll11323 = new String[]{"hiccups", "-m", "90", "-c", "15,16", "-r", "10000",
                "https://hicfiles.s3.amazonaws.com/hiseq/ch12-lx-b-lymphoblasts/in-situ/combined.hic",
                "/Users/muhammadsaadshamim/Desktop/juicer/ch12lx_hiccups_loops_list"};

        HiCTools.main(ll1132);


        //MotifFinder.redirectOutput("ls","/Users/muhammadsaadshamim/Desktop/suhas_motif/templs");


        //MotifFinder.executeCommand("/Applications/bedtools2/bin/bedtools > /Users/muhammadsaadshamim/Desktop/j5/temp");

        //MotifFinder.executeCommand("cd /Users/muhammadsaadshamim/Desktop/j5/");
        //System.out.println(MotifFinder.executeCommand("ll /Users/muhammadsaadshamim/Desktop/j5/temp"));

        /*InputStream is = ChromosomeSizes.class.getResourceAsStream("assembly" + ".chrom.sizes");
        BufferedReader reader = new BufferedReader(new InputStreamReader(is));
        String nextLine;
        int idx = 1;

        try {
            while ((nextLine = reader.readLine()) != null) {
                System.out.println();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
*/


        /*

        HiCGlobals.useCache = false;




        String file = "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic";
        Dataset ds = HiCFileTools.extractDatasetForCLT(Arrays.asList(file.split("\\+")), true);

        Set<String> chrs = new HashSet<String>();
        chrs.add("22");
        // select zoom level closest to the requested one

        List<Chromosome> commonChromosomes = ds.getChromosomes();
        commonChromosomes = new ArrayList<Chromosome>(HiCFileTools.stringToChromosomes(chrs, commonChromosomes));

        Feature2DList loopList5000 = Feature2DParser.parseLoopFile("/Users/muhammadsaadshamim/Desktop/j3/out8/loops_5000_pre", commonChromosomes, true, null);
        Feature2DList loopList10000 = Feature2DParser.parseLoopFile("/Users/muhammadsaadshamim/Desktop/j3/out8/loops_10000_pre", commonChromosomes, true, null);
        Feature2DList loopList25000 = Feature2DParser.parseLoopFile("/Users/muhammadsaadshamim/Desktop/j3/out8/loops_25000_pre", commonChromosomes, true, null);

        loopList5000.setColor(Color.black);
        loopList10000.setColor(Color.black);
        loopList25000.setColor(Color.black);

        Map<Integer, Feature2DList> looplists = new HashMap<Integer, Feature2DList>();
        looplists.put(5000, loopList5000);
        looplists.put(10000, loopList10000);
        looplists.put(25000, loopList25000);


        HiCCUPS.postProcess(looplists, ds, commonChromosomes, "/Users/muhammadsaadshamim/Desktop/j3/out8/L00PS");

        */


    }
}