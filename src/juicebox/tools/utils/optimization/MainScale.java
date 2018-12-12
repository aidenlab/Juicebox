/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2018 Broad Institute, Aiden Lab
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

package juicebox.tools.utils.optimization;

public class MainScale {

    // int scale(int c, int *m,int **i,int **j,double **x, double *z,double *b, double *report, int verb=0,
    // double tol=1.0e-3,double perc=1e-2,double perc1=0.25e-2, int maxiter=200, double del=1.0e-2,int trials=5);

    /*
    static void usage(const char *argv0)
    {
        fprintf(stderr, "Usage: %s [-m memarray][-p percent][-v verbose][-C max_array_count][-t tol] <infile> <vector_file> <outfile>\n", argv0);
        fprintf(stderr, "  <infile>: matrix file in sparse upper triangular notation\n");
        fprintf(stderr, "  <vector_file>: vector to scale to, all 1s for balanced\n");
        fprintf(stderr, "  <outfile>: normalization vector output  file\n");
    }
    */


    int main(int argc, char *argv[]) {

        // parameters
        int maxC = 100;
        double tol = 5.0e-4;
        int m1 = (int) 7e8;
        double perc = 1.0e-2;
        double perc1 = 0.25e-2;
        int verb = 0;
        int maxiter = 300;

        //////////////////time?ftime(&start);

        int[] m = new int[maxC];
        double[] z = new double[m1];
        double[] z0 = new double[m1];

        int c = 0;
        int[][] i = new int[maxC][m1];
        int[][] j = new int[maxC][m1];
        double[][] x = new double[maxC][m1];

        while (c < maxC) {
            long t0 = System.currentTimeMillis();
            int k = 0;
            while (fscanf(fin, "%d %d %lf", & i[c][k],&j[c][k],&x[c][k]) ==3){
                k++;
                if (k == m1) break;
            }
            if (k == 0) break;
            for (int p = 0; p < k; p++) {
                i[c][p]--;
                j[c][p]--;
            }
            m[c++] = k;
            //////////////////time?ftime(&t1);
            System.out.printf("took %ld seconds to read %d records\n", (long) (t1.time - t0.time), k);
            if (k < m1) break;
        }
        fclose(fin);
        System.out.printf("finished reading\n");
        n = 0;
        while (fscanf(finV, "%lf", & z0[n]) ==1)n++;
        for (int p = 0; p < n; p++) {
            z[p] = z0[p];
        }
        double[] b = new double[n + 1];
        for (int p = 0; p < n; p++) {
            b[p] = Double.NaN;
        }

        double[] report = new double[maxiter];

        //////////////////time?ftime(&t1);
        //	iter = scale(c, m,i,j,x, z,b, tol,perc,perc1,maxiter, report,verb);
        FinalScale.scale(c, m, i, j, x, z, b, report, verb);

        //////////////////time?ftime(&end);
        int iter = 0;


        System.out.printf("took %ld seconds\n", (long) (end.time - start.time));
        printf("iterations took %15.10lf seconds\n", ((double) (end.time - t1.time)) + 0.001 * (end.millitm - t1.millitm));
        if (verb) for (p = 0; p < abs(iter); p++) printf("%d: %30.15lg\n", p + 1, report[p]);
        int count = 0;
        while (iter < 0 && count++ < 3) {
            System.out.print("Did not converge!!!\n");
            perc = 1.5 * perc;
            System.out.printf("new perc = %lg\n", perc);
            for (p = 0; p < n; p++) z[p] = z0[p];
            //////////////////time?ftime(&t1);
            //		iter = scale(c, m,i,j,x, z,b, tol,perc,perc1,maxiter, report,verb);
            iter = scale(c, m, i, j, x, z, b, report, verb, -1, perc, 0.005);
            //////////////////time?ftime(&end);
            System.out.printf("iterations took %15.10lf seconds\n", ((double) (end.time - t1.time)) + 0.001 * (end.millitm - t1.millitm));
            if (verb == 0) {
                for (int p = 0; p < Math.abs(iter); p++) {
                    System.out.printf("%d: %30.15lg\n", p + 1, report[p]);
                }
            }
        }

        for (int p = 0; p < n; p++) {
            System.out.fprintf(fout, "%30.15lf\n", b[p]);
        }
        if (count >= 3) {
            System.out.printf("still did not converge!!!\n");
        }
        return iter;
    }


}
