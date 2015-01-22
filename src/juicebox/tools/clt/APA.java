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

import juicebox.tools.HiCTools;
import org.apache.commons.math.linear.Array2DRowRealMatrix;

import java.io.IOException;

public class APA extends JuiceboxCLT {
    private Array2DRowRealMatrix xMatrix;

    private String[] files;
    private double[] bounds;
    private int window = -100, resolution = -100;

    @Override
    public void readArguments(String[] args, HiCTools.CommandLineParser parser) throws IOException {
        setUsage("juicebox apa <minval maxval window  resolution> CountsFolder PeaksFile/PeaksFolder SaveFolder SavePrefix");

        if (!(args.length == 7 || args.length == 8)) {
            throw new IOException("1");
        }
        files = new String[args.length - 4];

     System.arraycopy(args, 4, files, 0, files.length);

        bounds = new double[2];

        try {
            bounds[0] = Double.valueOf(args[0]);
            bounds[1] = Double.valueOf(args[1]);
            window = Integer.valueOf(args[2]);
            resolution = Integer.valueOf(args[3]);
        } catch (NumberFormatException error) {
            throw new IOException("2");
        }

    }

    @Override
    public void run() {
        System.err.println("This method is not currently implemented.");
        System.exit(7);
    }




/*

 #enrichment measures

 def peak2mean(x,mdpt,width=None):
 return x[mdpt,mdpt]/((x.sum()-x[mdpt,mdpt])/(size(x)-1))
 def peak2UL(x,mdpt,width):
 return x[mdpt,mdpt]/mean(x[0:width,0:width])
 def peak2UR(x,mdpt,width):
 return x[mdpt,mdpt]/mean(x[0:width,-width:])
 def peak2LL(x,mdpt,width):
 return x[mdpt,mdpt]/mean(x[-width:,0:width])
 def peak2LR(x,mdpt,width):
 return x[mdpt,mdpt]/mean(x[-width:,-width:])
 def ZscoreLL(x,mdpt,width):
 y=x[-width:,0:width]
 return (x[mdpt,mdpt]-y.mean())/std(y)


 def get_chromosome_list(restriction_site_filename):
 import csv
 csv_sites=csv.reader(open(restriction_site_filename),delimiter=' ')
 chrom_list=[]
 for row in csv_sites:
 chrom_list.append(row[0])
 return chrom_list

 def get_peak_list(loop_filename, chrom):
 #The preferred method for getting loops is to give filename of full loop list. Backward compatible with old method of giving a folder, where each chr has indvidual loops named bp_chr_peaks.txt
 if os.path.isfile(loop_filename):
 chr=chrom.strip('chr')
 #file must be in standard loop notation and the first 6 columns must be chr1 x1 x2 chr2 y1 y2
 output = subprocess.check_output("awk '{ if ( (($1=="+chr+") || ($1==\"chr"+chr+"\")|| ($1==\""+chr+"\")) && (($4=="+chr+") || ($4==\"chr"+chr+"\") || ($4==\""+chr+"\")) && ($2 ~ /^[0-9]+$/)) print $2, $3, $5, $6}' "+ loop_filename, shell=True) #awk the relevent loops in the chromosome
 try:
 x= reshape(asarray(output.split(),dtype=int), (-1,4))
 #take the midpoint of the the loops
 x[:,0]+=(x[:,1]-x[:,0])/2
 x[:,2]+=(x[:,3]-x[:,2])/2
 x=sort(x,axis=1)
 return x[:,0::2] #every other
 except:
 return None
 elif os.path.isdir(loop_filename):
 peaks_filename=os.path.join(loop_filename,"bp_"+chrom +"_peaks.txt")
 if not os.path.isfile(peaks_filename):
 return None
 x=loadtxt(peaks_filename)
 if len(x)==0:
 return None
 else:
 x=array(x,dtype=int)
 x=sort(x,axis=1)
 return x

 #from http://stackoverflow.com/questions/8560440/removing-duplicate-columns-and-rows-from-a-numpy-2d-array
 def unique_rows(a):
 unique_a = unique(a.view([('', a.dtype)]*a.shape[1]))
 return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

 def SaveMeasures(x,mdpt,fname):
 f=open(fname,'w')
 width=6 #width of boxes
 tot_width=len(x)
 widthp5=width+.5
 half_width=width/2
 f.write('P2M'+'\t'+str(peak2mean(x,mdpt))+'\n')
 f.write('P2UL'+'\t'+str(peak2UL(x,mdpt,width))+'\n')
 f.write('P2UR'+'\t'+str(peak2UR(x,mdpt,width))+'\n')
 f.write('P2LL'+'\t'+ str(peak2LL(x,mdpt,width))+'\n')
 f.write('P2LR'+'\t'+ str(peak2LR(x,mdpt,width))+'\n')
 f.write('ZscoreLL'+'\t'+str(ZscoreLL(x,mdpt,width)))
 f.close()

 #defaults
 min_peak_dist=30 #distance between two bins, can be changed in opts
 max_peak_dist=Inf
 window=10
 width=6 #size of boxes
 peakwidth=2 #for enrichment calculation of crosshair norm
 res=1e4
 save_all=0


 #Parameter and Option Inputs
 try:
 opts, args =  getopt.getopt(sys.argv[1:],"hw:m:x:r:s")
 except:
 print "Correct usage:"
 print "PSEA_norm_GW.py <-m minval> <-x max> <-w window>  <-r res> CountsFolder PeaksFile/PeaksFolder SaveFolder SavePrefix "
 print   getopt.getopt(sys.argv[1:],"hw:m:r:")
 sys.exit(2)

 #Options
 for opt, val in opts:
 if opt == '-h':
 print "Correct usage:"
 print "PSEA_norm_GW.py <-m minval> <-x max> <-w window>  <-r res> CountsFolder PeaksFile/PeaksFolder SaveFolder SavePrefix "
 sys.exit()
 elif opt == '-w':
 window=int(val)
 elif opt == '-m':
 min_peak_dist=float(val)
 elif opt == '-x':
 max_peak_dist=float(val)
 elif opt == '-r':
 res=float(val)
 elif opt == '-s':
 save_all=1
 #Parameters (non optional)
 try:
 counts_folder = args[0]
 peaks_filename = args[1]
 save_folder=args[2]
 save_prefix=args[3]
 if len(args) > 4:
 restriction_site_filename =args[4]
 else:
 restriction_site_filename='/aidenlab/restriction_sites/hg19_HindIII.txt'
 except:
 print "Help on usage:"
 print "PSEA_norm_GW.py <-m minval> <-x max> <-w window>  <-r res> CountsFolder PeaksFile/PeaksFolder SaveFolder SavePrefix "
 sys.exit(2)

 #Create folders fo saving data
 if not os.path.exists(save_folder):
 os.makedirs(save_folder)
 data_folder=os.path.join(save_folder,"Data")
 if not os.path.exists(data_folder):
 os.makedirs(data_folder)

 #Calculate parameters that will need later
 L=2*window+1
 midpoint=window*(2*window+1)+window #midpoint of flattened matrix
 shift=range(-window,window+1) #window on which to do psea
 mdpt=len(shift)/2
 alpha=[str(s) for s in shift] #need this for tick marks on plots

 #define gw data structures
 GW_npeaks=0
 GW_npeaks_used=0
 GW_npeaks_used_nonunique=0
 GW_psea=zeros((L,L))
 GW_normed_psea=zeros((L,L))
 GW_center_normed_psea=zeros((L,L))
 GW_rank_psea=zeros((L,L))
 GW_coverage=zeros((L,L))
 GW_enhancement=[]
 shift=range(-window,window+1) #window on which to do psea
 chr_list=get_chromosome_list(restriction_site_filename)
 if save_all==1:
 f_allsave=open(os.path.join(data_folder,'GW_alldata.txt'), 'w')
 f_peaksused=open(os.path.join(data_folder,'GW_peaks_used.txt'), 'w')

 #Main loop
 for chr in chr_list:
 #define chromosome data structures
 psea=zeros((L,L))
 normed_psea=zeros((L,L))
 center_normed_psea=zeros((L,L))
 rank_psea=zeros((L,L))
 coverage=zeros((L,L))
 enhancement=[]

 #load peaks, take out invalid ones
 print "loading ", peaks_filename, "chr:",  chr
 peaks=get_peak_list(peaks_filename, chr)
 if (peaks is None) or (len(peaks)==0):
 print "No peaks found"
 continue
 npeaks=len(peaks)
 GW_npeaks+=len(peaks)
 peaks=array(peaks/float(res),dtype=int) #bin number
 peaks=peaks[where(peaks[:,0]>=window)]
 if len(peaks)>0:
 peaks=peaks[abs(peaks[:,1]-peaks[:,0])>=min_peak_dist,:] #minimum distance between two fragments in peak call
 if len(peaks)>0:
 peaks=peaks[abs(peaks[:,1]-peaks[:,0])<=max_peak_dist,:] #maximum distance between two fragments in peak call
 if len(peaks)==0:
 print "No peaks found after window, min, and max filtering"
 continue
 npeaks_used_nonunique=len(peaks)
 GW_npeaks_used_nonunique+=len(peaks)
 peaks=unique_rows(peaks)
 npeaks_used=len(peaks)
 GW_npeaks_used+=len(peaks)

 #load counts data
 counts_filename=os.path.join(counts_folder ,'counts_'+chr+'.txt')
 try:
 data = loadtxt(counts_filename) #newest hictools has no header
 except:
 data = loadtxt(counts_filename,skiprows=1) #first line of old hictools counts file is header
 if len(data)==0:
 continue
 I=array(data[:,0]/float(res),dtype=int)
 J=array(data[:,1]/float(res),dtype=int)
 V=data[:,2]
 V[isnan(V)]=0
 I, J = hstack((I,J)), hstack((J,I)) #fill out the lower triangle of matrix
 V=hstack((V,V)) #lower triangle
 chr_len=max(max(I),max(peaks.flatten()))+window
 contact_mat_coo = sparse.coo_matrix((V,(I,J)),shape=(chr_len+1,chr_len+1),dtype=float)#dtype float to include KR
 coverage=contact_mat_coo.sum(axis=1)
 contact_mat=sparse.dok_matrix(contact_mat_coo)  #dok_matrix supports slicing
 del contact_mat_coo, data, I, J,V

 #go through every peak, slice contact matrix and extract counts
 for p, peak in enumerate(peaks):
 x_index_list=arange(peak[0]-window,peak[0]+window+1,dtype=int)
 y_index_list=arange(peak[1]-window,peak[1]+window+1,dtype=int)
 small_mat=zeros((len(x_index_list),len(y_index_list)))
 for ii, i in enumerate(x_index_list):
 for jj, j in enumerate(y_index_list):
 small_mat[ii,jj]+=contact_mat[i,j]
 psea+=small_mat #no normalization
 normed_psea+=small_mat/max(1,mean(small_mat)) #normalization
 center_norm_val=small_mat[mdpt,mdpt]#normalize by midpoint
 if center_norm_val==0:
 tmp=small_mat.flatten()
 try:
 center_norm_val=min(tmp[tmp>0])
 except:
 center_norm_val=1 # if the whole matrix is 0
 center_normed_psea+=small_mat/center_norm_val
 x =small_mat.flatten()
 enhancement.append(x[midpoint]*(len(x)-1)/(sum(x[:midpoint])+sum(x[midpoint+1:])) ) #peak to mean value
 tmp_psea = array([stats.percentileofscore(x, i) for i in x])
 rank_psea+=reshape(tmp_psea,(2*window+1,2*window+1))
 if save_all==1:
 f_allsave.write(' '.join(map(str, small_mat.flatten()))+'\n')
 f_peaksused.write(chr+'\t'+str(peak[0])+'\t'+str(peak[1])+'\n')
 del peaks, contact_mat
 GW_psea+=psea
 GW_normed_psea+=normed_psea
 GW_center_normed_psea+=center_normed_psea
 GW_rank_psea+=rank_psea
 GW_enhancement+=enhancement
 normed_psea/=npeaks_used
 center_normed_psea/=npeaks_used
 rank_psea/=npeaks_used



 #save chromosome data
 savetxt(os.path.join(data_folder,os.path.split(counts_filename)[1].split('.',1)[0]+'_psea.txt'),psea)
 savetxt(os.path.join(data_folder,os.path.split(counts_filename)[1].split('.',1)[0]+'_normed_psea.txt'),normed_psea)
 savetxt(os.path.join(data_folder,os.path.split(counts_filename)[1].split('.',1)[0]+'_center_normed_psea.txt'),center_normed_psea)
 savetxt(os.path.join(data_folder,os.path.split(counts_filename)[1].split('.',1)[0]+'_rank_psea.txt'),rank_psea)
 savetxt(os.path.join(data_folder,os.path.split(counts_filename)[1].split('.',1)[0]+'_enhancement.txt'),enhancement)



 GW_normed_psea/=GW_npeaks_used
 GW_center_normed_psea/=GW_npeaks_used
 GW_rank_psea/=GW_npeaks_used

 #save GW data
 savetxt(os.path.join(data_folder,'GW_psea.txt'),GW_psea)
 savetxt(os.path.join(data_folder,'GW_normed_psea.txt'),GW_normed_psea)
 savetxt(os.path.join(data_folder,'GW_rank_psea.txt'),GW_rank_psea)
 savetxt(os.path.join(data_folder,'GW_enhancement.txt'),GW_enhancement)
 SaveMeasures(GW_psea,mdpt,os.path.join(data_folder,'GW_measures.txt'))
 if save_all==1:
 f_allsave.close()
 f_peaksused.close()

 */
}