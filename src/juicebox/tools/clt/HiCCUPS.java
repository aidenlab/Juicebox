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

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.JCudaDriver;
import jcuda.utils.KernelLauncher;
import juicebox.HiCGlobals;
import juicebox.data.*;
import juicebox.tools.HiCTools;
import juicebox.tools.utils.Common.CommonTools;
import juicebox.tools.utils.Common.MatrixTools;
import juicebox.tools.utils.Juicer.*;
import juicebox.track.Feature2D;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.RealMatrix;
import org.broad.igv.Globals;
import org.broad.igv.feature.Chromosome;

import java.awt.*;
import java.io.*;
import java.util.*;
import java.util.List;

import static jcuda.driver.JCudaDriver.*;


/**
 * Created by muhammadsaadshamim on 1/20/15.
 */
public class HiCCUPS extends JuiceboxCLT {

    private int[] resolutions = new int[]{25000, 10000, 5000};

    private boolean chrSpecified = false;
    Set<String> chromosomesSpecified = new HashSet<String>();

    private String inputHiCFileName, saveFolderName, savePrefixName;

    public HiCCUPS() {
        super("hiccups [-r resolution] [-c chromosome] <hic file> <SaveFolder> [SavePrefix]");
        // -i input file custom
    }





    @Override
    public void readArguments(String[] args, HiCTools.CommandLineParser parser) throws IOException {
        // read


        System.out.println(args);

        if (!(args.length > 1 && args.length < 5)) {
            throw new IOException("1");
        }

        inputHiCFileName = args[1];
        saveFolderName = args[2];

        if (args.length == 4)
            savePrefixName = args[3];

        Set<String> specifiedChromosomes = parser.getChromosomeOption();
        int newResolutionVal =  parser.getResolutionOption();

        if (newResolutionVal > 0)
            resolutions = new int[]{newResolutionVal};

        if(specifiedChromosomes != null) {
            chromosomesSpecified = new HashSet<String>(specifiedChromosomes);
            chrSpecified = true;
        }
    }

    @Override
    public void run() {

        //Calculate parameters that will need later

        try {
            System.out.println("Accessing " + inputHiCFileName);
            DatasetReaderV2 reader = new DatasetReaderV2(inputHiCFileName);
            Dataset ds = reader.read();
            // select zoom level closest to the requested one

            List<Chromosome> commonChromosomes = ds.getChromosomes();
            if(chrSpecified)
                commonChromosomes = new ArrayList<Chromosome>(CommonTools.stringToChromosomes(chromosomesSpecified,
                        commonChromosomes));

            for (int resolution : resolutions) {
                HiCZoom zoom = CommonTools.getZoomLevel(ds, resolution);
                resolution = zoom.getBinSize();

                HiCGlobals.verifySupportedHiCFileVersion(reader.getVersion());

                // Loop through chromosomes
                for (Chromosome chr : commonChromosomes) {

                    if (chr.getName().equals(Globals.CHR_ALL)) continue;
                    Matrix matrix = ds.getMatrix(chr, chr);
                    if (matrix == null) continue;
                    MatrixZoomData zd = matrix.getZoomData(zoom);

                    NormalizationVector normVector = ds.getNormalizationVector(chr.getIndex(), zoom, NormalizationType.KR);

                    //RealMatrix newData = APAUtils.extractLocalizedData(zd, loop, L, resolution, window);

                }
            }
        } catch (IOException e) {
            System.out.println("Unable to extract APA data");
            e.printStackTrace();
            System.exit(-3);
        }
    }



    public static RealMatrix extractLocalizedData(MatrixZoomData zd, Feature2D loop,
                                                  int L, int resolution, int window) {
        int loopX = loop.getMidPt1() / resolution;
        int loopY = loop.getMidPt2() / resolution;
        int binXStart = loopX - (window + 1);
        int binXEnd = loopX + (window + 1);
        int binYStart = loopY - (window + 1);
        int binYEnd = loopY + (window + 1);

        Set<Block> blocks = new HashSet<Block>(zd.getNormalizedBlocksOverlapping(binXStart, binYStart, binXEnd, binYEnd,
                NormalizationType.KR));

        RealMatrix data = MatrixTools.cleanArray2DMatrix(L, L);

        for (Block b : blocks) {
            for (ContactRecord rec : b.getContactRecords()) {

                // [0..window-1  window  window+1..2*window+1]
                int relativeX = window + (rec.getBinX() - loopX);
                int relativeY = window + (rec.getBinY() - loopY);

                if (relativeX >= 0 && relativeX < L) {
                    if (relativeY >= 0 && relativeY < L) {
                        data.addToEntry(relativeX, relativeY, rec.getCounts());
                        System.out.println(relativeX+" "+relativeY+" "+rec.getCounts());
                    }
                }
            }
        }

        //System.out.println((System.nanoTime()-time)/1000000000.);
        return data;
    }


    /**
     *
     *
     *
     */
    private String kernelCode(int window, int matrixSize, int peakWidth, int divisor) {
        return "extern \"C\"\n" +
                "__global__ void BasicPeakCallingKernel(float *c, float *expectedbl, float *expecteddonut," +
                "float *expectedh, float *expectedv, float *observed, float *b_bl, float *b_donut," +
                "float *b_h, float *b_v, float *p, float *tbl, float *td, float *th, float *tv," +
                "float *d, float *kr1, float *kr2, float *bound1, float *bound3)\n" +
                "{\n" +
                "       // 2D Thread ID \n" +
                "       int t_col = threadIdx.x + blockIdx.x * blockDim.x;\n" +
                "       int t_row = threadIdx.y + blockIdx.y * blockDim.y;\n" +
                "" +
                "       // Evalue is used to store the element of the matrix\n" +
                "       // that is computed by the thread\n" +
                "       float Evalue_bl =  0;\n" +
                "" +
                "       float Edistvalue_bl = 0;\n" +
                "       float Evalue_donut =  0;\n" +
                "       float Edistvalue_donut = 0;\n" +
                "       float Evalue_h =  0;\n" +
                "       float Edistvalue_h = 0;\n" +
                "       float Evalue_v =  0;\n" +
                "       float Edistvalue_v = 0;\n" +
                "       float e_bl = 0;\n" +
                "       float e_donut = 0;\n" +
                "       float e_h = 0;\n" +
                "       float e_v = 0;\n" +
                "       float o = 0;\n" +
                "       float sbtrkt = 0;\n" +
                "       float bvalue_bl = 0;\n" +
                "       float bvalue_donut = 0;\n" +
                "       float bvalue_h = 0;\n" +
                "       float bvalue_v = 0;\n" +
                "       int wsize = " + window + ";\n" +
                "       int msize = " + matrixSize + ";\n" +
                "       int pwidth = " + peakWidth + ";\n" +
                "       //int dvisor = " + divisor + ";\n" +  // TODO remove?
                "       int diff = bound1[0] - bound3[0];\n" +
                "" +
                "       while (abs(t_row+diff-t_col)<=(2*wsize)) {\n" +
                "               wsize = wsize - 1;\n" +
                "       }\n" +
                "" +
                "       if (wsize<=pwidth) {\n" +
                "               wsize = pwidth + 1;\n" +
                "       }\n" +
                "" +
                "       if (t_row>=20&&t_row<=(msize-20)&&t_col>=20&&t_col<=(msize-20)) {\n" +
                "               // calculate initial bottom left box\n" +
                "               for (int i = max(0,t_row+1); i < min(t_row+wsize+1, msize); ++i) {\n" +
                "                       int test=max(0,t_col-wsize);\n" +
                "                       for (int j = test; j < min(t_col, msize); ++j) {\n" +
                "                               if (!isnan(c[i * msize + j])) {  \n" +
                "                                       if (i+diff-j<0) {\n" +
                "                                               Evalue_bl += c[i * msize +j];\n" +
                "                                               Edistvalue_bl += d[abs(i+diff-j)];\n" +
                "                                       }\n" +
                "                               }\n" +
                "                       }\n" +
                "               }\n" +
                "" +
                "               //Subtract off the middle peak\n" +
                "               for (int i = max(0,t_row+1); i < min(t_row+pwidth+1, msize); ++i) {\n" +
                "                       int test=max(0,t_col-pwidth);\n" +
                "                       for (int j = test; j < min(t_col, msize); ++j) {\n" +
                "                               if (!isnan(c[i * msize + j])) {  \n" +
                "                                       if (i+diff-j<0) {\n" +
                "                                               Evalue_bl -= c[i * msize +j];\n" +
                "                                               Edistvalue_bl -= d[abs(i+diff-j)];\n" +
                "                                       }\n" +
                "                               }\n" +
                "                       }\n" +
                "               }\n" +
                "" +
                "               //fix box dimensions\n" +
                "               while (Evalue_bl<16) {\n" +
                "                       Evalue_bl=0;\n" +
                "                       Edistvalue_bl=0;\n" +
                "                       wsize+=1;\n" +
                "                       //dvisor = powf(wsize,2.0) - powf(pwidth,2.0);\n" + //TODO remove?
                "                       for (int i = max(0,t_row+1); i < min(t_row+wsize+1, msize); ++i) {\n" +
                "                               int test=max(0,t_col-wsize);\n" +
                "                               for (int j = test; j < min(t_col, msize); ++j) {\n" +
                "                                       if (!isnan(c[i * msize + j])) {  \n" +
                "                                               if (i+diff-j<0) {\n" +
                "                                                       Evalue_bl += c[i * msize +j];\n" +
                "                                                       Edistvalue_bl += d[abs(i+diff-j)];\n" +
                "                                                       if (i>=t_row+1) {\n" +
                "                                                               if (i<t_row+pwidth+1) {\n" +
                "                                                                       if (j>=t_col-pwidth) {\n" +
                "                                                                               if (j<t_col) {\n" +
                "                                                                                       Evalue_bl -= c[i * msize +j];\n" +
                "                                                                                       Edistvalue_bl -= d[abs(i+diff-j)];\n" +
                "                                                                               }\n" +
                "                                                                       }\n" +
                "                                                                }\n" +
                "                                                       }\n" +
                "                                               }\n" +
                "                                       }\n" +
                "                               }\n" +
                "                       }\n" +
                "" +
                "                       //Subtact off the middle peak\n" +
                "                       //for (int i = max(0,t_row+1); i < min(t_row+pwidth+1, msize); ++i) {\n" +
                "                       //int test=max(0,t_col-pwidth);\n" +
                "                       //for (int j = test; j < min(t_col, msize); ++j) {\n" +
                "                       //if (!isnan(c[i * msize + j])) {  \n" +
                "                       //if (i+diff-j<0) {\n" +
                "                       //Evalue_bl -= c[i * msize +j];\n" +
                "                       //Edistvalue_bl -= d[abs(i+diff-j)];\n" +
                "                       //}\n" +
                "                       //}\n" +
                "                       //}\n" +
                "                       //}\n" +
                "" +
                "                       if (wsize == 20) {\n" +
                "                               break;\n" +
                "                       }\n" +
                "                       if (2*wsize>=abs(t_row+diff-t_col)) {\n" +
                "                               break;\n" +
                "                       }\n" +
                "               }\n" +
                "" +
                "               // calculate donut\n" +
                "               for (int i = max(0,t_row-wsize); i < min(t_row+wsize+1, msize); ++i) {\n" +
                "                       int test=max(0,t_col-wsize);\n" +
                "                       for (int j = test; j < min(t_col+wsize+1, msize); ++j) {\n" +
                "                               if (!isnan(c[i * msize + j])) {  \n" +
                "                                       if (i+diff-j<0) {\n" +
                "                                               Evalue_donut += c[i * msize +j];\n" +
                "                                               Edistvalue_donut += d[abs(i+diff-j)];\n" +
                "                                       }\n" +
                "                               }\n" +
                "                       }\n" +
                "               }\n" +
                "" +
                "               //Subtract off the middle peak\n" +
                "               for (int i = max(0,t_row-pwidth); i < min(t_row+pwidth+1, msize); ++i) {\n" +
                "                       int test=max(0,t_col-pwidth);\n" +
                "                       for (int j = test; j < min(t_col+pwidth+1, msize); ++j) {\n" +
                "                               if (!isnan(c[i * msize + j])) {  \n" +
                "                                       if (i+diff-j<0) {\n" +
                "                                               Evalue_donut -= c[i * msize +j];\n" +
                "                                               Edistvalue_donut -= d[abs(i+diff-j)];\n" +
                "                                       }\n" +
                "                               }\n" +
                "                       }\n" +
                "               }\n" +
                "               //Subtract off the cross hairs\n" +
                "               if ((t_row-pwidth)>0) {\n" +
                "                       for (int i = max(0,t_row-wsize); i < (t_row-pwidth); ++i) {\n" +
                "                               if (!isnan(c[i * msize + t_col])) {  \n" +
                "                                       Evalue_donut -= c[i * msize + t_col];\n" +
                "                                       Edistvalue_donut -= d[abs(i+diff-t_col)];\n" +
                "                               }\n" +
                "                               for (int j = -1; j <=1 ; ++j) {\n" +
                "                                       Evalue_v += c[i * msize + t_col + j];\n" +
                "                                       Edistvalue_v += d[abs(i+diff-t_col-j)];\n" +
                "                               }\n" +
                "                       }\n" +
                "               }\n" +
                "               if ((t_row+pwidth)<msize) {\n" +
                "                       for (int i = (t_row+pwidth+1); i < min(t_row+wsize+1,msize); ++i) {\n" +
                "                               if (!isnan(c[i * msize + t_col])) {  \n" +
                "                                       Evalue_donut -= c[i * msize + t_col];\n" +
                "                                       Edistvalue_donut -= d[abs(i+diff-t_col)];\n" +
                "                               }\n" +
                "                               for (int j = -1; j <=1 ; ++j) {\n" +
                "                                       Evalue_v += c[i * msize + t_col + j];\n" +
                "                                       Edistvalue_v += d[abs(i+diff-t_col-j)];\n" +
                "                               }\n" +
                "                       }\n" +
                "               }\n" +
                "               if ((t_col-pwidth)>0) {\n" +
                "                       for (int j = max(0,t_col-wsize); j < (t_col-pwidth); ++j) {\n" +
                "                               if (!isnan(c[t_row * msize + j])) {  \n" +
                "                                       Evalue_donut -= c[t_row * msize + j];\n" +
                "                                       Edistvalue_donut -= d[abs(t_row+diff-j)];\n" +
                "                               }\n" +
                "                               for (int i = -1; i <=1 ; ++i) {\n" +
                "                                       Evalue_h += c[(t_row+i) * msize + j];\n" +
                "                                       Edistvalue_h += d[abs(t_row+i+diff-j)];\n" +
                "                               }\n" +
                "                       }\n" +
                "               }\n" +
                "               if ((t_col+pwidth)<msize) {\n" +
                "                       for (int j = (t_col+pwidth+1); j < min(t_col+wsize+1,msize); ++j) {\n" +
                "                               if (!isnan(c[t_row * msize + j])) {  \n" +
                "                                       Evalue_donut -= c[t_row * msize + j];\n" +
                "                                       Edistvalue_donut -= d[abs(t_row+diff-j)];\n" +
                "                               }\n" +
                "                               for (int i = -1; i <=1 ; ++i) {\n" +
                "                                       Evalue_h += c[(t_row+i) * msize + j];\n" +
                "                                       Edistvalue_h += d[abs(t_row+i+diff-j)];\n" +
                "                               }\n" +
                "                       }\n" +
                "               }\n" +
                "       }\n" +
                "" +
                "       //if (t_row+diff-t_col<(-1*pwidth)-2) {\n" +
                "       e_bl = ((Evalue_bl*d[abs(t_row+diff-t_col)])/Edistvalue_bl)*kr1[t_row]*kr2[t_col];\n" +
                "       e_donut = ((Evalue_donut*d[abs(t_row+diff-t_col)])/Edistvalue_donut)*kr1[t_row]*kr2[t_col];\n" +
                "       e_h = ((Evalue_h*d[abs(t_row+diff-t_col)])/Edistvalue_h)*kr1[t_row]*kr2[t_col];\n" +
                "       e_v = ((Evalue_v*d[abs(t_row+diff-t_col)])/Edistvalue_v)*kr1[t_row]*kr2[t_col];\n" +
                "" +
                "       if (!isnan(e_bl)) {\n" +
                "               if (e_bl<=1) {\n" +
                "                       bvalue_bl = 0;\n" +
                "               }\n" +
                "               else {\n" +
                "                       bvalue_bl = floorf(logf(e_bl)/logf(powf(2.0,.33)));\n" +
                "               }\n" +
                "       }\n" +
                "       if (!isnan(e_donut)) {\n" +
                "               if (e_donut<=1) {\n" +
                "                       bvalue_donut = 0;\n" +
                "               }\n" +
                "               else {\n" +
                "                       bvalue_donut = floorf(logf(e_donut)/logf(powf(2.0,.33)));\n" +
                "               }\n" +
                "       }\n" +
                "       if (!isnan(e_h)) {\n" +
                "               if (e_h<=1) {\n" +
                "                       bvalue_h = 0;\n" +
                "               }\n" +
                "               else {\n" +
                "                       bvalue_h = floorf(logf(e_h)/logf(powf(2.0,.33)));\n" +
                "               }\n" +
                "       }\n" +
                "       if (!isnan(e_v)) {\n" +
                "               if (e_v<=1) {\n" +
                "                       bvalue_v = 0;\n" +
                "               }\n" +
                "               else {\n" +
                "                       bvalue_v = floorf(logf(e_v)/logf(powf(2.0,.33)));\n" +
                "               }\n" +
                "       }\n" +
                "" +
                "       // Write the matrix to device memory;\n" +
                "       // each thread writes one element\n" +
                "       expectedbl[t_row * msize + t_col] = e_bl;\n" +
                "       expecteddonut[t_row * msize + t_col] = e_donut;\n" +
                "       expectedh[t_row * msize + t_col] = e_h;\n" +
                "       expectedv[t_row * msize + t_col] = e_v;\n" +
                "       o = roundf(c[t_row * msize + t_col]*kr1[t_row]*kr2[t_col]);\n" +
                "       observed[t_row * msize + t_col] = o; //roundf(c[t_row * msize + t_col]*kr1[t_row]*kr2[t_col]);\n" +
                "       b_bl[t_row * msize + t_col] = bvalue_bl;\n" +
                "       b_donut[t_row * msize + t_col] = bvalue_donut;\n" +
                "       b_h[t_row * msize + t_col] = bvalue_h;\n" +
                "       b_v[t_row * msize + t_col] = bvalue_v;\n" +
                "       sbtrkt = fmaxf(tbl[(int) bvalue_bl],td[(int) bvalue_donut]);\n" +
                "       sbtrkt = fmaxf(sbtrkt, th[(int) bvalue_h]);\n" +
                "       sbtrkt = fmaxf(sbtrkt, tv[(int) bvalue_v]);\n" +
                "       p[t_row * msize + t_col] = o-sbtrkt;\n" +
                "}";
    }




    private void runKernel(String[] args){
        long begin_time= System.currentTimeMillis();

        // take in input sheet
        PrintWriter outfile1 = CommonTools.openWriter(args[2]);
        PrintWriter outfile2 = CommonTools.openWriter(args[3]);

        int fdr = Integer.parseInt(args[4]);
        int MATRIX_SIZE = 540;
        int PEAKWIDTH = Integer.parseInt(args[5]);
        int WINDOW = Integer.parseInt(args[6]);
        int DIVISOR = (WINDOW)^2-(PEAKWIDTH)^2;

        int w1 = 40, w2 = 10000;

        int[][] hist_bl = new int[w1][w2];
        int[][] hist_donut = new int[w1][w2];
        int[][] hist_h = new int[w1][w2];
        int[][] hist_v = new int[w1][w2];

        float[][] fdrlog_bl = new float[w1][w2];
        float[][] fdrlog_donut = new float[w1][w2];
        float[][] fdrlog_h = new float[w1][w2];
        float[][] fdrlog_v = new float[w1][w2];

        int[] threshold_bl = new int[40];
        int[] threshold_donut = new int[40];
        int[] threshold_h = new int[40];
        int[] threshold_v = new int[40];

        float[] bound1array = new float[1];
        float[] bound3array = new float[1];



        //GPU stuff
        //number of threads in block
        int block_size = 16;

        //threads per block = block_size*block_size
        int[] block =  new int[]{block_size, block_size, 1};

        // for grid of blocks
        int[] grid = new int[]{(MATRIX_SIZE/block_size)+1, (MATRIX_SIZE/block_size)+1};

        //print "Using block", block, " and grid", grid

        // get the kernel code from the template
        // by specifying the constants
        String kernel_code = kernelCode(WINDOW,MATRIX_SIZE,PEAKWIDTH,DIVISOR);

        // compile the kernel code
        KernelLauncher kernelLauncher =
                KernelLauncher.compile(kernel_code, "BasicPeakCallingKernel");

        /*

        // get the kernel function from the compiled module
        matrixmul = mod.get_function("BasicPeakCallingKernel");

        for (int runNum : new int[]{0,1}) {
            input_file = open(sys.argv[1], 'r');
            input_line = input_file.readline();
            while (input_line != "") {
                input_splits = input_line.split();
                long start_time = System.currentTimeMillis();

                // load inputs
                f = open(input_splits[1], 'rb');
                index1 = array.array('i');
                index2 = array.array('i');
                val = array.array('f');
                while (true) {
                    try {
                        index1.fromfile(f, 1);
                        index2.fromfile(f, 1);
                        val.fromfile(f, 1);
                    }
                    catch (Exception e) {
                        //except EOFError:break
                        val=map(np.float32, val);
                    }
                }
                int[] index1 = np.asarray(index1);
                int[] index2 = np.asarray(index2);
                val = np.asarray(val);
                d_cpu = np.loadtxt(input_splits[2]).astype(np.float32);
                kr_total_cpu = np.loadtxt(input_splits[3]).astype(np.float32);
                res =int(input_splits[4]);
                //threshold_bl_cpu=copy.deepcopy(threshold_bl)
                //threshold_donut_cpu=copy.deepcopy(threshold_donut)
                //print threshold_bl
                res =Integer.parseInt(input_splits[4]);

                long load_time = System.currentTimeMillis();
                System.out.println("Time to load chr" + input_splits[0] + " matrix: " + (load_time - start_time));

                // make dense matrix on CPU memory
                //a_dense_cpu=np.asarray(sparse.coo_matrix((val,(index1,index2))).todense())
                //long convert_time=System.currentTimeMillis();
                //print "Time to convert chr"+input_splits[0]+" matrix to dense: "+str(convert_time-load_time)+"s"

                int dim = arrayMax(index1);
                int dim2 = arrayMax(index2);
                for (int i = 0; i < (dim / 500) + 1; i++) {
                    int bound1_r = Math.max((i * 500), 0);
                    int bound2_r = Math.min((i + 1) * 500, dim);
                    int bound1 = Math.max((i * 500) - 20, 0);
                    int bound2 = Math.min(((i + 1) * 500) + 20, dim);
                    if (bound1 == 0) {
                        bound2 = 540;
                    }
                    if (bound2 == dim) {
                        bound1 = dim - 540;
                    }
                    int diff1 = bound1_r - bound1;
                    int diff2 = bound2 - bound2_r;
                    int cut1 = index1 >= bound1;
                    int cut1_index1 = index1[cut1];
                    int cut1_index2 = index2[cut1];
                    int cut1_val = val[cut1];
                    int cut2 = cut1_index1 < bound2;
                    int cut2_index1 = cut1_index1[cut2];
                    int cut2_index2 = cut1_index2[cut2];
                    int cut2_val = cut1_val[cut2];
                    for (int j = i, j <  (dim / 500) + 1; j++) {
                        int bound3_r = max((j * 500), 0);
                        int bound4_r = min((j + 1) * 500, dim);
                        int bound3 = max((j * 500) - 20, 0);
                        int bound4 = min(((j + 1) * 500) + 20, dim);
                        if (bound3 == 0) {
                            bound4 = 540;
                        }
                        if (bound4 == dim) {
                            bound3 = dim - 540;
                        }
                        int diff3 = bound3_r - bound3;
                        int diff4 = bound4 - bound4_r;
                        int cut3 = cut2_index2 >= bound3;
                        int cut3_index1 = cut2_index1[cut3];
                        int cut3_index2 = cut2_index2[cut3];
                        int cut3_val = cut2_val[cut3];
                        int cut4 = cut3_index2 < bound4;
                        int cut4_index1 = cut3_index1[cut4] - bound1;
                        int cut4_index2 = cut3_index2[cut4] - bound3;
                        int cut4_val = cut3_val[cut4];
                        int size_test1 = cut4_index1 == 539;
                        int size_test2 = cut4_index2[size_test1] == 539;
                        if len(cut4_val[size_test2]) == 0 {
                            cut4_index1 = np.concatenate([cut4_index1, np.asarray([539])]);
                            cut4_index2 = np.concatenate([cut4_index2, np.asarray([539])]);
                            cut4_val = np.concatenate([cut4_val, np.asarray([0])]);
                        }
                        a_cpu = np.asarray(sparse.coo_matrix((cut4_val, (cut4_index1, cut4_index2))).todense(), np.float32);
                        kr1_cpu = copy.deepcopy(kr_total_cpu[bound1:bound2]);
                        kr2_cpu = copy.deepcopy(kr_total_cpu[bound3:bound4]);
                        bound1array[0] = bound1;
                        bound3array[0] = bound3;

                        long gpu_time1 = System.currentTimeMillis();
                        // transfer host (CPU) memory to device (GPU) memory
                        a_gpu = gpuarray.to_gpu(a_cpu);
                        d_gpu = gpuarray.to_gpu(d_cpu);
                        kr1_gpu = gpuarray.to_gpu(kr1_cpu);
                        kr2_gpu = gpuarray.to_gpu(kr2_cpu);
                        threshold_bl_gpu = gpuarray.to_gpu(threshold_bl);
                        threshold_donut_gpu = gpuarray.to_gpu(threshold_donut);
                        threshold_h_gpu = gpuarray.to_gpu(threshold_h);
                        threshold_v_gpu = gpuarray.to_gpu(threshold_v);
                        bound1array_gpu = gpuarray.to_gpu(bound1array);

                        CUdeviceptr bound3array_gpu = GPUHelper.allocateInput(Pointer.to(bound3array),
                                bound3array.length, Sizeof.FLOAT);


                        // TODO size of gpu matrix
                        Dimension areaSize = new Dimension(5,6);
                        int flattenedSize = areaSize.height*areaSize.width;

                        // create empty gpu array for the result
                        CUdeviceptr expected_bl_gpu = GPUHelper.allocateOutput(flattenedSize, Sizeof.FLOAT);
                        CUdeviceptr expected_donut_gpu = GPUHelper.allocateOutput(flattenedSize, Sizeof.FLOAT);
                        CUdeviceptr expected_h_gpu = GPUHelper.allocateOutput(flattenedSize, Sizeof.FLOAT);
                        CUdeviceptr expected_v_gpu = GPUHelper.allocateOutput(flattenedSize, Sizeof.FLOAT);
                        CUdeviceptr bin_bl_gpu = GPUHelper.allocateOutput(flattenedSize, Sizeof.FLOAT);
                        CUdeviceptr bin_donut_gpu = GPUHelper.allocateOutput(flattenedSize, Sizeof.FLOAT);
                        CUdeviceptr bin_h_gpu = GPUHelper.allocateOutput(flattenedSize, Sizeof.FLOAT);
                        CUdeviceptr bin_v_gpu = GPUHelper.allocateOutput(flattenedSize, Sizeof.FLOAT);
                        CUdeviceptr observed_gpu = GPUHelper.allocateOutput(flattenedSize, Sizeof.FLOAT);
                        CUdeviceptr peak_gpu = GPUHelper.allocateOutput(flattenedSize, Sizeof.FLOAT);

                        // call the kernel on the card
                        matrixmul(
                                // inputs
                                a_gpu,
                                // output
                                expected_bl_gpu,
                                expected_donut_gpu,
                                expected_h_gpu,
                                expected_v_gpu,
                                observed_gpu,
                                bin_bl_gpu,
                                bin_donut_gpu,
                                bin_h_gpu,
                                bin_v_gpu,
                                peak_gpu,
                                // thresholds
                                threshold_bl_gpu,
                                threshold_donut_gpu,
                                threshold_h_gpu,
                                threshold_v_gpu,
                                // distance expected
                                d_gpu,
                                // kr
                                kr1_gpu,
                                kr2_gpu,
                                // bounds
                                bound1array_gpu,
                                bound3array_gpu,
                                //  grid of blocks
                                grid = grid,
                                //  block of threads
                                block = block
                        );

                        float[] expected_bl_result = new float[flattenedSize];
                        float[] expected_donut_result = new float[flattenedSize];
                        float[] expected_h_result = new float[flattenedSize];
                        float[] expected_v_result = new float[flattenedSize];
                        float[] bin_bl_result = new float[flattenedSize];
                        float[] bin_donut_result = new float[flattenedSize];
                        float[] bin_h_result = new float[flattenedSize];
                        float[] bin_v_result = new float[flattenedSize];
                        float[] observed_result = new float[flattenedSize];
                        float[] peak_result = new float[flattenedSize];

                        cuMemcpyDtoH(Pointer.to(expected_bl_result), expected_bl_gpu, flattenedSize * Sizeof.FLOAT);
                        cuMemcpyDtoH(Pointer.to(expected_donut_result), expected_bl_gpu, flattenedSize * Sizeof.FLOAT);
                        cuMemcpyDtoH(Pointer.to(expected_h_result), expected_bl_gpu, flattenedSize * Sizeof.FLOAT);
                        cuMemcpyDtoH(Pointer.to(expected_v_result), expected_bl_gpu, flattenedSize * Sizeof.FLOAT);
                        cuMemcpyDtoH(Pointer.to(bin_bl_result), expected_bl_gpu, flattenedSize * Sizeof.FLOAT);
                        cuMemcpyDtoH(Pointer.to(bin_donut_result), expected_bl_gpu, flattenedSize * Sizeof.FLOAT);
                        cuMemcpyDtoH(Pointer.to(bin_h_result), expected_bl_gpu, flattenedSize * Sizeof.FLOAT);
                        cuMemcpyDtoH(Pointer.to(bin_v_result), expected_bl_gpu, flattenedSize * Sizeof.FLOAT);
                        cuMemcpyDtoH(Pointer.to(observed_result), expected_bl_gpu, flattenedSize * Sizeof.FLOAT);
                        cuMemcpyDtoH(Pointer.to(peak_result), expected_bl_gpu, flattenedSize * Sizeof.FLOAT);


                        long gpu_time2 = System.currentTimeMillis();

                        //print gpu_time2-gpu_time1, bound1_r, bound3_r
                        observed_dense_cpu = np.zeros((540 - diff2 - diff1, 540 - diff4 - diff3), dtype =int);
                        observed_dense_cpu[:,:]=copy.deepcopy(observed_result[diff1:(540 - diff2), diff3:(540 - diff4)]);
                        //print observed_dense_cpu[observed_dense_cpu[:,:]>0]
                        expected_bl_dense_cpu = copy.deepcopy(expected_bl_result[diff1:(540 - diff2), diff3:(540 - diff4)]);
                        expected_donut_dense_cpu = copy.deepcopy(expected_donut_result[diff1:(540 - diff2), diff3:(540 - diff4)]);
                        expected_h_dense_cpu = copy.deepcopy(expected_h_result[diff1:(540 - diff2), diff3:(540 - diff4)]);
                        expected_v_dense_cpu = copy.deepcopy(expected_v_result[diff1:(540 - diff2), diff3:(540 - diff4)]);
                        bin_bl_dense_cpu = copy.deepcopy(bin_bl_result[diff1:(540 - diff2), diff3:(540 - diff4)]);
                        bin_donut_dense_cpu = copy.deepcopy(bin_donut_result[diff1:(540 - diff2), diff3:(540 - diff4)]);
                        bin_h_dense_cpu = copy.deepcopy(bin_h_result[diff1:(540 - diff2), diff3:(540 - diff4)]);
                        bin_v_dense_cpu = copy.deepcopy(bin_v_result[diff1:(540 - diff2), diff3:(540 - diff4)]);
                        peak_dense_cpu = copy.deepcopy(peak_result[diff1:(540 - diff2), diff3:(540 - diff4)]);
                        if run == 0 {

                            cleanUpNaNs(expected_bl_dense_cpu,expected_donut_dense_cpu,expected_h_dense_cpu,expected_v_dense_cpu,
                                    bin_bl_dense_cpu, bin_donut_dense_cpu, bin_h_dense_cpu, bin_v_dense_cpu);

                            nanvals_bl = np.isnan(expected_bl_dense_cpu);
                            nanvals_donut = np.isnan(expected_donut_dense_cpu);
                            nanvals_h = np.isnan(expected_h_dense_cpu);
                            nanvals_v = np.isnan(expected_v_dense_cpu);

                            bin_bl_dense_cpu[nanvals_bl] =  Float.NaN;
                            bin_bl_dense_cpu[nanvals_donut] =Float.NaN;
                            bin_bl_dense_cpu[nanvals_h] =Float.NaN;
                            bin_bl_dense_cpu[nanvals_v] =Float.NaN;

                            bin_donut_dense_cpu[nanvals_bl] =Float.NaN;
                            bin_donut_dense_cpu[nanvals_donut] =Float.NaN;
                            bin_donut_dense_cpu[nanvals_h] =Float.NaN;
                            bin_donut_dense_cpu[nanvals_v] =Float.NaN;

                            bin_h_dense_cpu[nanvals_bl] =Float.NaN;
                            bin_h_dense_cpu[nanvals_donut] =Float.NaN;
                            bin_h_dense_cpu[nanvals_h] =Float.NaN;
                            bin_h_dense_cpu[nanvals_v] =Float.NaN;

                            bin_v_dense_cpu[nanvals_bl] =Float.NaN;
                            bin_v_dense_cpu[nanvals_donut] =Float.NaN;
                            bin_v_dense_cpu[nanvals_h] =Float.NaN;
                            bin_v_dense_cpu[nanvals_v] =Float.NaN;

                            d_correct = (bound1_r - bound3_r) +int(sys.argv[5]) + 2;
                            dim_box = np.shape(observed_dense_cpu)[0];

                            if (d_correct >= (-1 * dim_box)) {
                                temp_cpu = np.ones(np.shape(observed_dense_cpu));
                                nan_cpu = np.tril(temp_cpu, d_correct);
                                nanvals_temp = nan_cpu == 1;
                                nan_cpu[nanvals_temp] =Float.NaN;
                                del temp_cpu;
                                bin_bl_dense_cpu = bin_bl_dense_cpu + nan_cpu;
                                bin_donut_dense_cpu = bin_donut_dense_cpu + nan_cpu;
                                bin_h_dense_cpu = bin_h_dense_cpu + nan_cpu;
                                bin_v_dense_cpu = bin_v_dense_cpu + nan_cpu;
                            }
                            for (int k = 0; k < w1; k++) {
                                blvals = bin_bl_dense_cpu == k;
                                donutvals = bin_donut_dense_cpu == k;
                                hvals = bin_h_dense_cpu == k;
                                vvals = bin_v_dense_cpu == k;
                                //hist, bin_edges = np.histogram(observed_dense_cpu[blvals],bins=range(10001))
                                //hist_bl[i,:]=hist_bl[i,:]+hist
                                hist_bl[k,:]=hist_bl[k,:]+np.bincount(observed_dense_cpu[blvals], minlength = 10000)[0:10000];
                                //hist, bin_edges = np.histogram(observed_dense_cpu[donutvals],bins=range(10001))
                                hist_donut[k,:]=hist_donut[k,:];
                                +np.bincount(observed_dense_cpu[donutvals], minlength = 10000)[0:10000];
                                //hist_donut[i,:]=hist_donut[i,:]+hist
                                hist_h[k,:]=hist_h[k,:]+np.bincount(observed_dense_cpu[hvals], minlength = 10000)[0:10000];
                                hist_v[k,:]=hist_v[k,:]+np.bincount(observed_dense_cpu[vvals], minlength = 10000)[0:10000];
                            }
                        }
                        if (runNum == 1) {
                            index_dense_cpu = np.asarray(range(np.shape(observed_dense_cpu)[0] * np.shape(observed_dense_cpu)[1]));
                            index_dense_cpu = np.reshape(index_dense_cpu, np.shape(observed_dense_cpu));
                            nanvals_bl = np.isnan(expected_bl_dense_cpu);
                            nanvals_donut = np.isnan(expected_donut_dense_cpu);
                            nanvals_h = np.isnan(expected_h_dense_cpu);
                            nanvals_v = np.isnan(expected_v_dense_cpu);
                            peak_dense_cpu[nanvals_bl] =Float.NaN;
                            peak_dense_cpu[nanvals_donut] =Float.NaN;
                            peak_dense_cpu[nanvals_h] =Float.NaN;
                            peak_dense_cpu[nanvals_v] =Float.NaN;
                            d_correct = (bound1_r - bound3_r) +int(sys.argv[5]) + 2;
                            dim_box = np.shape(observed_dense_cpu)[0];
                            dim_box2 = np.shape(observed_dense_cpu)[1];
                            if (d_correct >= (-1 * dim_box)) {
                                temp_cpu = np.ones(np.shape(observed_dense_cpu));
                                nan_cpu = np.tril(temp_cpu, d_correct);
                                nanvals_temp = nan_cpu == 1;
                                nan_cpu[nanvals_temp] =Float.NaN;
                                del temp_cpu;
                                peak_dense_cpu = peak_dense_cpu + nan_cpu;
                            }
                            peak_index = peak_dense_cpu > 0;
                            int numPeaks = len(index_dense_cpu[peak_index]);
                            indexp = index_dense_cpu[peak_index];
                            op = observed_dense_cpu[peak_index];
                            epbl = expected_bl_dense_cpu[peak_index];
                            epdonut = expected_donut_dense_cpu[peak_index];
                            eph = expected_h_dense_cpu[peak_index];
                            epv = expected_v_dense_cpu[peak_index];
                            bpbl = bin_bl_dense_cpu[peak_index];
                            bpdonut = bin_donut_dense_cpu[peak_index];
                            bph = bin_h_dense_cpu[peak_index];
                            bpv = bin_v_dense_cpu[peak_index];
                            for (int k = 0; k < numPeaks; k++) {
                                row = indexp[k] / dim_box2 + bound1_r;
                                col = indexp[k] % dim_box2 + bound3_r;
                                if (int(bpbl[k]) < 40 and op[ k]<10000 and int(bpdonut[k]) < 40 and int(bph[k]) < 40 and int(bpv[k]) < 40)
                                {
                                    fdr_bl = fdrlog_bl[int(bpbl[k]), op[k]];
                                    fdr_donut = fdrlog_donut[int(bpdonut[k]), op[k]];
                                    fdr_h = fdrlog_h[int(bph[k]), op[k]];
                                    fdr_v = fdrlog_v[int(bpv[k]), op[k]];
                                    outfile2.println(input_splits[0] + "\t" + str(row * res) + "\t" + input_splits[0] + "\t" +
                                            str(col * res) + "\t" + str(op[k]) + "\t" + str(epbl[k]) + "\t" + str(epdonut[k]) + "\t" +
                                            str(eph[k]) + "\t" + str(epv[k]) + "\t" + str(bpbl[k]) + "\t" + str(bpdonut[k]) + "\t" +
                                            str(bph[k]) + "\t" + str(bpv[k]) + "\t" + str(fdr_bl) + "\t" + str(fdr_donut) + "\t" +
                                            str(fdr_h) + "\t" + str(fdr_v));
                                }
                            }
                        }
                    }
                }
                if (runNum == 0) {
                    long hist_time = System.currentTimeMillis();
                    System.out.println("Time to calculate chr" + input_splits[0] + " expecteds and add to hist: " + (hist_time - load_time) + "s");
                }
                if (runNum == 1) {
                    long peak_time = System.currentTimeMillis();
                    System.out.println("Time to print chr" + input_splits[0] + " peaks: " + (peak_time - load_time) + "s");
                }
                input_line = input_file.readline();
            }
            input_file.close();
            if (runNum == 0) {

                long thresh_time0 = System.currentTimeMillis();

                run0ProcessHistogram(hist_bl, w1, w2, fdr, threshold_bl, fdrlog_bl);
                run0ProcessHistogram(hist_donut, w1, w2, fdr, threshold_donut, fdrlog_donut);
                run0ProcessHistogram(hist_h, w1, w2, fdr, threshold_h, fdrlog_h);
                run0ProcessHistogram(hist_v, w1, w2, fdr, threshold_v, fdrlog_v);

                for (int i = 0; i < w1; i++){
                    outfile1.println(i + "\t" + threshold_bl[i] + "\t" + threshold_donut[i] + "\t" +
                            threshold_h[i] + "\t" + threshold_v[i]);
                }
                long thresh_time1 = System.currentTimeMillis();
                System.out.println( "Time to calculate thresholds: " + (thresh_time1 - thresh_time0));
            }
        }
        long final_time = System.currentTimeMillis();
        System.out.println( "Total time: "+ (final_time-begin_time));
        */

        outfile1.close();
        outfile2.close();
    }

    private void run0ProcessHistogram(int[][] hist, int w, int h, int fdr, int[] threshold, float[][] fdrLog) {

        int[][] rcsHist = new int[w][h];
        for(int i = 0; i < w; i++) {
            rcsHist[i] = makeReverseCumulativeArray(hist[i]);
        }

        for (int i = 0; i < w; i++) {
            calculateThresholdAndFDR(i, h, fdr, rcsHist, threshold, fdrLog);
        }

    }

    private void calculateThresholdAndFDR(int index, int width, int fdr, int[][] rcsHist,
                                          int[] threshold, float[][] fdrLog) {
        if (rcsHist[index][0]>0){
            float[] expected = generatePoissonPMF(index,rcsHist[index][0]);
            float[] rcsExpected = makeReverseCumulativeArray(expected);
            for (int j = 0; j < width; j++) {
                if( fdr * rcsExpected[j] <= rcsHist[index][j]){
                    threshold[index] = (j - 1);
                    break;
                }
            }
            for (int j = threshold[index]; j < width; j++){
                float sum1 = rcsExpected[j];
                float sum2 = rcsHist[index][j];
                if (sum2 > 0) {
                    fdrLog[index][j]= (sum1 / (sum2 * 1f));
                }
                else {
                    break;
                }
            }
        }
        else{
            threshold[index] = width;
        }
    }

    private int arrayMax(int[] array){
        int max = array[0];
        for(int val : array)
            if(val > max)
                max = val;
        return max;
    }

    private int arrayMin(int[] array){
        int min = array[0];
        for(int val : array)
            if(val < min)
                min = val;
        return min;
    }

    private float[] generatePoissonPMF(int index, int total) {
        // TODO optimize because poisson calculation repeated multiple times
        //total * stats.poisson.pmf(range(10000), 2 * * ((i + 1) / 3.0))
        return null;
    }

    private float[] makeReverseCumulativeArray(float[] inputArray) {
        float[] outputArray = new float[inputArray.length];
        float total = 0f;
        for (int i = inputArray.length-1; i > -1 ; i--) {
            total += inputArray[i];
            outputArray[i] = total;
        }
        return outputArray;
    }

    private int[] makeReverseCumulativeArray(int[] inputArray) {
        int[] outputArray = new int[inputArray.length];
        int total = 0;
        for (int i = inputArray.length-1; i > -1 ; i--) {
            total += inputArray[i];
            outputArray[i] = total;
        }
        return outputArray;
    }
}

