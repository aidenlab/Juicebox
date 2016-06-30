# Juicer
Juicer is a platform for analyzing kilobase resolution Hi-C data. In this distribution,
we include the pipeline for generating Hi-C maps from fastq raw data files and command
line tools for feature annotation on the Hi-C maps.

----------------------
Example for Reviewers
----------------------
We've provided a step-by-step guide to showcase some of the features of
Juicer. If you run into problems, see below for more detailed documentation.
This example runs on Amazon Web Services, but you can install the pipeline
on any LSF, Univa Grid Engine, or SLURM cluster.

1. Make sure you're in the top-level directory, with this README and the
   Juicer_AWS.pem file. (NOTE: please do not share this .pem file with anyone)
2. You were given an anonymous IP address. At a command line prompt, type:
      ssh -i Juicer_AWS.pem ubuntu@<given IP address>
3. This will log you into an AWS instance that contains all the software
   needed to run the pipeline. Type
      cd /opt/juicer/work/
4. We will run the pipeline on a test dataset of a single chromosome of the primary+
   replicate map from (Rao+Huntley et al., 2014). Type:
      cd MBR19
5. Run the Juicer pipeline on the raw data, which is stored in the fastq
   directory:
      /opt/juicer/scripts/juicer.sh -g hg19 -s MboI
6. You will see a series of messages sending jobs to the cluster. Do not
   kill the script or close the server connection until you see:
      “(-: Finished adding all jobs... please wait while processing.”
7. At this point you can close the connection and come back later.
   To see the progress of the pipeline as it works, type:
      bjobs -w
7. Eventually the bjobs command will report “No unfinished job found”. Type:
      tail lsf.out
   You should see “(-: Pipeline successfully completed (-:”
8. Results are available in the aligned directory. The Hi-C maps are in
   inter.hic (for MAPQ > 0) and inter_30.hic (for MAPQ >= 30). The Hi-C maps
   can be loaded in Juicebox and explored. They can also be used for
   automatic feature annotation and to extract matrices at specific
   resolutions.
   These results also include automatic feature annotation. The output files include
   a genome-wide annotation of loops and, whenever possible, the CTCF motifs that anchor
   them (identified using the HiCCUPS algorithm). The files also include a genome-wide
   annotation of contact domains (identified using the Arrowhead algorithm). The formats
   of these files are described in the Juicebox tutorial online; both files can be loaded
   into Juicebox as a 2D annotation.
9. To download a file (e.g. inter.hic) from AWS to load into Juicebox, type:
      sftp -i Juicer_AWS.pem ubuntu@<given IP address>
      cd /opt/juicer/work/MBR19/aligned
      get inter.hic
      get inter_30.hic
      get ... (each of hiccups, apa, motifs, and arrowhead output files)
10. You can also run the pipeline on genome-wide dataset that is lower resolution. Type
      cd /opt/juicer/work/HIC003
   Then
      /opt/juicer/scripts/juicer.sh -g hg19 -s MboI
   Again the pipeline will run. The results will be available in the aligned directory.
   Because this is not a deeply sequenced map, loop lists and domain lists will not be
   produced.

See below for more documentation.

------------
Distribution
------------
The files included on the AWS distribution are in /opt/juicer and are:
*references        - Genome references
*restriction_sites - Restriction positions for combinations of reference genome and restriction enzymes
*scripts           - Juicer main scripts
*work              - Data samples
*hic_files         - Internal property file

In this zip file, we include the scripts for running Juicer on LSF,
Univa Grid Engine, and SLURM

/AWS - scripts for running pipeline and postprocessing on AWS
/UGER - scripts for running pipeline and postprocessing on UGER
/SLURM - scripts for running pipeline and postprocessing on SLURM
/juicebox_tools - source files for postprocessing algorithms
/Juicer_AWS.pem - ssh key to access anonymous AWS server

----------------------------------
Hardware and Software Requirements
----------------------------------
Juicer is a pipeline optimized for parallel computation on a cluster. Juicer
consists of two parts: the pipeline that creates Hi-C files from raw data,
and the post-processing command line tools.

*
* Cluster requirements:
*

Juicer requires the use of a cluster, with ideally >= 4 cores (min 1 core)
and >= 64 GB RAM (min 16 GB RAM)

Juicer currently works with the following resource management software:
- OpenLava (http://www.openlava.org/)
- LSF (http://www-03.ibm.com/systems/services/platformcomputing/lsf.html)
- SLURM (http://slurm.schedmd.com/download.html)
- GridEngine (Univa, etc. any flavor)

*
* Command line tool requirements:
*

The minimum software requirement to run Juicer is a working Java installation
(version >= 1.7) on Windows, Linux, and Mac OSX.  We recommend using the
latest Java version available, but please do not use the Java Beta Version.
Minimum system requirements for running Java can be found at
http://java.com/en/download/help/sysreq.xml

To download and install the latest Java Runtime Environment (JRE), please go
to http://www.java.com/download


*
* GNU CoreUtils
*

The latest version of GNU coreutils can be downloaded from
https://www.gnu.org/software/coreutils/manual/

*
* Burrows-Wheeler Aligner (BWA)
*

The latest version of BWA should be installed from
http://bio-bwa.sourceforge.net/


*
* CUDA (for HiCCUPS peak calling)
*

You must have an NVIDIA GPU to install CUDA
Instructions for installing the latest version of CUDA can be found
on the NVIDIA Developer site:
   https://developer.nvidia.com/cuda-downloads

The native libraries included with Juicer are compiled for CUDA 7.
Other versions of CUDA can be used, but you will need to download the
respective native libraries from
   http://www.jcuda.org/downloads/downloads.html

For best performance, use a dedicated GPU. You may also be able to obtain
access to GPU clusters through Amazon Web Services or a local research
institution.

*
* Java 1.7 or 1.8 JDK (for compiling from source files)
*

The instructions here are for the Java 1.8 JDK.
For Windows/Mac/Linux, the Java 1.8 JDK can be installed from here:
   http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html
(Alternative) For Ubuntu/LinuxMint
   http://tecadmin.net/install-oracle-java-8-jdk-8-ubuntu-via-ppa/

*
* Apache Ant (for compiling from source files)
*

Mac
   Ant should be installed on most Macs. To verify installation via the
command prompt, type
       ant -version
   If Ant is not on your Mac, install it via homebrew. At the command prompt,
type
       brew update
       brew install ant
   You may need to install Homebrew (http://brew.sh/) on your machine
   See the following Stackoverflow post for more details:
       http://stackoverflow.com/questions/3222804/how-can-i-install-apache-ant-on-mac-os-x

Windows
   Installing Ant requires some minor changes to your system environment. Follow the instructions in this article:
       http://www.nczonline.net/blog/2012/04/12/how-to-install-apache-ant-on-windows/

Linux
   In the command prompt, type
       sudo apt-get install ant
or
       sudo yum install ant
depending on your package installer


--------------------------------
Compiling Jars from Source Files
--------------------------------
1. You should have Java 1.7 (or 1.8) JDK and Apache Ant installed on your system. See
   below for more information.
2. Go to the folder containing the Juicebox source files and edit the
   juicebox.properties file with the proper Java JDK Address.
3. Open the command line, navigate to the folder containing the build.xml file
   and type
       ant
   The process should take no more than a minute to build on most machines.
4. The jars are written to the directory out/.  You can change this by editing
   the build.xml file.

-------------
Documentation
-------------
We have extensive documentation below for how to use Juicer.

------------------------
Command Line Tools Usage
------------------------
To launch the command line tools, use the shell script “juicebox.sh” on Unix/MacOS
or type
   java -jar juicebox_tools.jar (command...) [flags...] <parameters...>

For HiCCUPS loop calling without the shell or bat script, you will need to
call:
   java -Xms512m -Xmx2048m -Djava.library.path=path/to/natives/
    -jar Juicebox_CLT.jar hiccups [flags...] <parameters...>
   where path/to/natives is the path to the native libraries used for Jcuda
   By default, these are located in the lib/jcuda folder.

In the command line tools, there are 4 functions:
> "apa" for conducting aggregate peak analysis
> "hiccups" for annotating loops
> "motifs" for finding CTCF motifs
> "arrowhead" for annotating contact domains

The "juicebox.sh” (Unix/MacOS) script can be used in place of the unwieldy
	"java -Djava.library.path=path/to/natives/ -jar juicebox_tools.jar"

---
APA
---
The "apa" command takes three required arguments and a number of optional
arguments.

apa [-n minval] [-x maxval] [-w window]  [-r resolution(s)] [-c chromosome(s)]
   [-k NONE/VC/VC_SQRT/KR] <hicFile(s)> <PeaksFile> <SaveFolder>

The required arguments are:

<hicFile(s)>: Address of hic file(s) which should end with ".hic". This is the file you will
   load into Juicebox. URLs or local addresses may be used. To sum multiple hic Files together,
   use the '+' symbol between the addresses (no whitespace between addresses)
<PeaksFile>: List of peaks in standard 2D feature format (chr1 x1 x2 chr2 y1 y2 color ...)
<SaveFolder>: Working directory where outputs will be saved

The optional arguments are:
   -n <int> minimum distance away from the diagonal. Used to filter peaks too close to the diagonal.
       Units are in terms of the provided resolution. (e.g. -n 30 @ resolution 5kB will filter loops
       within 30*(5000/sqrt(2)) units of the diagonal)
   -x <int> maximum distance away from the diagonal. Used to filter peaks too far from the diagonal.
       Units are in terms of the provided resolution. (e.g. -n 30 @ resolution 5kB will filter loops
       further than 30*(5000/sqrt(2)) units of the diagonal)
   -w <int> width of region to be aggregated around the specified loops (units of resolution)
   -r <int(s)> resolution for APA; multiple resolutions can be specified using commas (e.g. 5000,10000)
   -c <String(s)> Chromosome(s) on which APA will be run. The number/letter for the chromosome can be
       used with or without appending the "chr" string. Multiple chromosomes can be specified using
       commas (e.g. 1,chr2,X,chrY)
   -k <NONE/VC/VC_SQRT/KR> Normalizations (case sensitive) that can be selected. Generally, KR (Knight-Ruiz)
       balancing should be used when available.

Default settings of optional arguments:
   -n 30
   -x (infinity)
   -w 10
   -r 25000,10000
   -c (all chromosomes)
   -k KR

------------
APA Examples
------------

apa HIC006.hic all_loops.txt results1
> This command will run APA on HIC006 using loops from the all_loops files
> and save them under the results1 folder.

apa https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic
   all_loops.txt results1
> This command will run APA on the GM12878 mega map using loops from the all_loops
> files and save them under the results1 folder.

apa -r 10000,5000 -c 17,18 HIC006.hic+HIC007.hic all_loops.txt results
> This command will run APA at 50 kB resolution on chromosomes 17 and 18 for the
> summed HiC maps (HIC006 and HIC007) using loops from the all_loops files
> and save them under the results folder

-------
HiCCUPS
-------

hiccups [-m matrixSize] [-c chromosome(s)] [-r resolution(s)] [-k normalization (NONE/VC/VC_SQRT/KR)] [-f fdr] [-p peak width] [-i window]
   [-t thresholds] [-d centroid distances] <hicFile> <outputLoopsList>

The required arguments are:

<hicFile>: Address of hic file which should end with ".hic". This is the file you will
   load into Juicebox. URLs or local addresses may be used.

<outputLoopsList>: Final list of all loops found by HiCCUPS. Can be visualized directly in Juicebox as a 2D annotation.
   By default, various values critical to the HICCUPS algorithm are saved as attributes for each loop found. These can be
   disabled using the suppress flag below.

The optional arguments are:
   -m <int> Maximum size of the submatrix within the chromosome passed on to GPU (Must be an even number greater than 40
       to prevent issues from running the CUDA kernel). The upper limit will depend on your GPU. Dedicated GPUs
       should be able to use values such as 500, 1000, or 2048 without trouble. Integrated GPUs are unlikely to run
       sizes larger than 90 or 100. Matrix size will not effect the result, merely the time it takes for hiccups.
       Larger values (with a dedicated GPU) will run fastest.
   -c <String(s)> Chromosome(s) on which HiCCUPS will be run. The number/letter for the chromosome can be used with or
       without appending the "chr" string. Multiple chromosomes can be specified using commas (e.g. 1,chr2,X,chrY)
   -r <int(s)> Resolution(s) for which HiCCUPS will be run. Multiple resolutions can be specified using commas
       (e.g. 25000,10000,5000). Due to the nature of DNA looping, it is unlikely that loops will be found at
       lower resolutions (i.e. 50kB or 100kB)
       IMPORTANT: if multiple resolutions are used, the flags below can be configured so that different parameters are
       used for the different resolutions.
   -k <NONE/VC/VC_SQRT/KR> Normalizations (case sensitive) that can be selected. Generally, KR (Knight-Ruiz)
       balancing should be used when available.
   -f <int(s)> FDR values actually corresponding to max_q_val (i.e. for 1% FDR use 0.01, for 10%FDR use 0.1). Different
       FDR values can be used for each resolution using commas. (e.g "-r 5000,10000 -f 0.1,0.15" would run HiCCUPS at
       10% FDR for resolution 5000 and 15% FDR for resolution 10000)
   -p <int(s)> Peak width used for finding enriched pixels in HiCCUPS. Different peak widths can be used for each
       resolution using commas. (e.g "-r 5000,10000 -p 4,2" would run at peak width 4 for resolution 5000 and
       peak width 2 for resolution 10000)
   -i <int(s)> Window width used for finding enriched pixels in HiCCUPS. Different window widths can be used for each
       resolution using commas. (e.g "-r 5000,10000 -p 10,6" would run at window width 10 for resolution 5000 and
       window width 6 for resolution 10000)
   -t <floats> Thresholds for merging loop lists of different resolutions. Four values must be given, separated by
       commas (e.g. 0.02,1.5,1.75,2). These thresholds (in order) represent:
       > threshold allowed for sum of FDR values of the horizontal, vertical, donut, and bottom left filters
           (an accepted loop must stay below this threshold)
       > threshold ratio that both the horizontal and vertical filters must exceed
       > threshold ratio that both the donut and bottom left filters must exceed
       > threshold ratio that at least one of the donut and bottom left filters must exceed
   -d <ints> Distances used for merging nearby pixels to a centroid. Different distances can be used for each
       resolution using commas. (e.g "-r 5000,10000 -d 20000,21000” would merge pixels within 20kB of each
       other at 5kB resolution and within 21kB at 10kB resolution.

Defaults:

  Medium resolution maps:
   -m 512
   -c (all chromosomes)
   -r 10000
   -k KR
   -f .1
   -p 2
   -i 5
   -t 0.02,1.5,1.75,2
   -d 20000,20000,50000

  High resolution maps:
   -m 512
   -c (all chromosomes)
   -r 5000,10000
   -k KR
   -f .1,.1
   -p 4,2
   -i 7,5
   -t 0.02,1.5,1.75,2
   -d 20000,20000,50000

----------------
HiCCUPS Examples
----------------

hiccups HIC006.hic all_hiccups_loops
> This command will run HiCCUPS on HIC006 and save all found loops to the all_hiccups_loops files

hiccups -m 500 -r 5000,10000 -f 0.1,0.1 -p 4,2 -i 7,5 -d 20000,20000,0  -c 22  HIC006.hic all_hiccups_loops
> This command will run HiCCUPS on chromosome 22 of HIC006 at 5kB and 10kB resolution using the following values:
>> 5kB: fdr 10%, peak width 4, window width 7, and centroid distance 20kB
>> 10kB: fdr 10%, peak width 2, window width 5, and centroid distance 20kB
> The resulting loop list will be merged and saved as all_hiccups_loops
> Note that these are values used for generating the GM12878 loop list


-------
Arrowhead
-------

arrowhead [-c chromosome(s)] [-m matrix size] [-r resolution] [-k normalization (NONE/VC/VC_SQRT/KR)] " +
                "<hicFile(s)> <output_file> [feature_list] [control_list]

The required arguments are:

<hicFile(s)>: Address of hic file(s) which should end with ".hic". This is the file you will
   load into Juicebox. URLs or local addresses may be used. To sum multiple hic Files together,
   use the '+' symbol between the addresses (no whitespace between addresses)

<output_file>: Final list of all contact domains found by Arrowhead. Can be visualized directly in Juicebox
  as a 2D annotation.

-- NOTE -- If you want to find scores for a feature and control list, both must be provided:

[feature_list]: Feature list of loops/domains for which block scores are to be calculated
[control_list]: Control list of loops/domains for which block scores are to be calculated


The optional arguments are:

-c <String(s)> Chromosome(s) on which Arrowhead will be run. The number/letter for the chromosome can be used with or
  without appending the "chr" string. Multiple chromosomes can be specified using commas (e.g. 1,chr2,X,chrY)
-m <int> Size of the sliding window along the diagonal in which contact domains will be found. Must be an even
  number as (m/2) is used as the increment for the sliding window. (Default 2000)
-r <int> resolution for which Arrowhead will be run. Generally, 5kB (5000) or 10kB (10000)
  resolution is used depending on the depth of sequencing in the hic file(s).
-k <NONE/VC/VC_SQRT/KR> Normalizations (case sensitive) that can be selected. Generally, KR (Knight-Ruiz)
       balancing should be used when available.


Default settings of optional arguments:

  Medium resolution maps:
   -c (all chromosomes)
   -m 2000
   -r 10000
   -k KR

  High resolution maps:
   -c (all chromosomes)
   -m 2000
   -r 5000
   -k KR

----------------
Arrowhead Examples
----------------

NOTE: Arrowhead will choose appropriate defaults for hic files if no specifications are given

arrowhead https://hicfiles.s3.amazonaws.com/hiseq/ch12-lx-b-lymphoblasts/in-situ/combined_30.hic contact_domains_list
  This command will run Arrowhead on a mouse cell line HiC map (medium resolution) at resolution 10 kB and save all
  contact domains to the contact_domains_list file. These are the settings used to generate the official contact
  domain list on the ch12-lx-b-lymphoblast cell line.

arrowhead https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined_30.hic contact_domains_list
  This command will run Arrowhead at resolution 5kB on the GM12878 HiC map (high resolution) and save all contact
  domains to the contact_domains_list file. These are the settings used to generate the official GM12878
  contact domain list.

-----------------------
Motif Finder
-----------------------

motifs <genomeID> <bed_file_dir> <looplist> [custom_global_motif_list]

The required arguments are:

<genomeID>: hg19 supported by default. For other genome assemblies, provide a 
  custom_global_motif_list in FIMO format.

<bed_file_dir> File path to a directory (e.g. ) which contains two folders: "unique" and
  "inferred". These folders should contain a combination of RAD21, SMC3, and CTCF BED files.
  By intersecting these 1D tracks, the strongest peaks will be identified. Unique motifs
  generally use a more stringent combination of BED files than inferred motifs.

<looplist>: List of peaks in standard 2D feature format (chr1 x1 x2 chr2 y1 y2 color ...)

-- NOTE -- If you want to use a custom list of potential motifs:

[custom_global_motif_list]: Motif list output using FIMO format can be used as an alternative
  to the internal motif list

-------------------
Motif Finder Examples
-------------------

Assuming the following file structure is present:
/path/to/local/bed/files/unique/CTCF.bed
/path/to/local/bed/files/unique/RAD21.bed
/path/to/local/bed/files/unique/SMC3.bed
/path/to/local/bed/files/inferred/CTCF.bed

motifs hg19 /path/to/local/bed/files /gm12878_hiccups_loops.txt
  This command will find motifs from the internal hg19 motif list for the loops in gm12878_hiccups_loops.txt
  and save them to gm12878_hiccups_loops_with_motifs.txt. The CTCF, RAD21, and SMC3 BED files will be used
  together (i.e. intersected) to find unique motifs. Just the CTCF track will be used to infer best motifs.

motifs hg19 /path/to/local/bed/files gm12878_hiccups_loops.txt hg_19_custom_motif_list.txt
  This command will find motifs from hg_19_custom_motif_list.txt for the loops in gm12878_hiccups_loops.txt
  and save them to gm12878_hiccups_loops_with_motifs.txt. The CTCF, RAD21, and SMC3 BED files will be used
  together (i.e. intersected) to find unique motifs. Just the CTCF track will be used to infer best motifs.
