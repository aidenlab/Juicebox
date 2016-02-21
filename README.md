--------------
About Juicebox
--------------
Juicebox is visualization software for Hi-C data.  In this distribution, we
include both the visualization software itself and command line tools for
creating files that can be loaded into Juicebox.


Check out the <a href="http://aidenlab.org/juicebox">Juicebox website</a> for more details on how to
use Juicebox, as well as following a detailed tutorial. Instructions below pertain primarily to usage of
command line tools and the Juicebox jar files.

Juicebox was created by <a href="https://github.com/jrobinso">Jim Robinson</a>,
<a href="https://github.com/nchernia">Neva C. Durand</a>, and <a href="http://www.erez.com/">Erez Lieberman Aiden</a>.

Ongoing development work is carried out by <a href="https://github.com/nchernia">Neva C. Durand</a>,
<a href="https://github.com/sa501428">Muhammad Saad Shamim</a>, <a href="https://github.com/imachol">Ido Machol</a>,
<a href="https://github.com/zgire">Zulkifl Gire</a>, and <a href="https://github.com/mhoeger">Marie Hoeger</a>.

--------------
IntelliJ Setup
--------------

Use IntelliJ IDEA (Community edition - free)

To set up in IDEA, have the Java SDK installed
then you'll point to it (IntelliJ has lots of documentation on this sort of thing).

* Then go to `VCS` -> `checkout from version control`.
* You'll need to do is be sure `*.sizes` is included as a file to be copied over to the class files.
Set this up via IntelliJ `Preferences` -> `Compiler`. Add `?*.sizes` to the list of `Resource Patterns`.
* While there, also go to `Java Compiler` and put this into additional command line options: `-Xlint:all -target 1.7`
The former turns on all warnings, the latter gives some flexibility since some people haven't updated Java to 1.8 yet.
* Then go to `Run` -> `Edit Configurations`.
* With the `+` sign, add `Application`.
* You'll create two of these, one for the GUI (call it Juicebox GUI or whatever you want, really) and one for the CLT.
* Set the main class by clicking the little `...` button next to the text box for main class

        MainWindow.java is the main method class for the visualization/GUI portion of the software.
        HiCTools.java is the main method class for the analysis/CLT portion.

* For the GUI under VM Options:

        -Xmx2000m
        -Djnlp.loadMenu="http://hicfiles.tc4ga.com/juicebox.properties"

* For the CLT use

        -Xmx2000m

* Note that the `Xmx2000m` flag sets the maximum memory heap size to 2GB.
Depending on your computer you might want more or less.
Some tools will break if there's not enough memory and the file is too large,
but don't worry about that for development; 2GB should be fine.
* One last note: be sure to `Commit and Push` when you commit files, it's hidden in the dropdown menu button in the
commit window.

-------
Example
-------
We've provided a step-by-step guide to showcase some of the features of
Juicebox.

1.  At a command line prompt, type:
      java -Xms512m -Xmx2048m -jar Juicebox.jar
2.  Click File->Open to load a new Hi-C map. Click GM12878, in situ MboI,
    primary+replicate (4.9B). The map will load and show all chromosomes.
3.  Click on chromosome 17. Using the selector below Normalization, change the
    normalization to Balanced.
4.  Slide the resolution slider to 5 KB.
5.  In the Goto panel, type 17:64500000-69000000 and hit the refresh button.
6.  Click Annotations in the menu bar and go to Load Basic Annotations. Select
    the Dataset Specific 2-D Features, Peaks and Contact Domains for this map.
    You will see a series of yellow boxes and cyan points loaded. The cyan
    points denote the exact peaks and so are small. These are the contact
    domains and peaks that we found when analyzing this map.
7.  Go to the Annotations menu and click Load Basic Annotations again. Click on
    GM12878 and select the H3K36me3 track and the CTCF track.
8.  You can line up what you see in the heat map with the tracks. Right click
    on the heat map and select Enable straight edge. Then move around on the
    map to see the features lined up with the tracks.  Press F2 to turn on and
    off the 2D annotations.
9.  Now, to create your own map from your own data, go to the command line
    prompt.
10. On Unix or Mac, type:
      juicebox pre data/test.txt.gz data/test.hic hg19
    On Windows, type:
      juicebox.bat pre data/test.txt.gz data/test.hic hg19
11. A .hic file should be produced. In Juicebox, click File -> Open.. and then
    click the Local button at the bottom of the dialog.  Navigate to the data/
    directory and click on test.hic
12. The file should load. Click on chromosome 1. Go to 1Mb resolution.
13. Move your mouse until it hovers over the bin at 100mb x 100mb.
14. The hover text on the right should read:
      1:100,000,001-101,000,000
      1:100,000,001-101,000,000
      observed value = 410
      expected value = 339.504
      O/E = 1.208
15. To extract raw counts from the .hic file, go to the command prompt.
16. On a Unix or Mac system, type:
      juicebox dump observed KR data/test.hic 1 1 BP 2500000 > chr1.txt
    On Windows, type:
      juicebox.bat dump observed KR data/test.hic 1 1 BP 2500000 > chr1.txt
17. The file chr1.txt contains the observed matrix of chromosome 1 with
    KR (Balanced) normalization at 2.5Mb resolution.

See below for more documentation.

------------
Distribution
------------
The files included in this distribution are as follows:
* README
* Juicebox.jar, Juicebox_CLT.jar, juicebox, and juicebox.bat (executables)
* build.xml and juicebox.properties (for compilation from source)
* src (directory containing source code)
* lib (directory containing libraries for compilation)
* data (directory containing test data)

----------------------------------
Hardware and Software Requirements
----------------------------------
The minimum software requirement to run Juicebox is a working Java installation
(version > 1.6) on Windows, Linux, and Mac OSX.  We recommend using the latest
Java version available, but please do not use the Java Beta Version. Minimum
system requirements for running Java can be found at
http://java.com/en/download/help/sysreq.xml. To download and install the latest
Java Runtime Environment (JRE), please go to http://www.java.com/download.

We recommend having at least 2GB free RAM for the best user experience with
Juicebox.

To launch the Juicebox application from command line, type
  java -Xms512m -Xmx2048m -jar Juicebox.jar

To launch the command line tools, run the shell script “juicebox” on Unix or
MacOS, run the batch script "juicebox.bat" on Windows, or type
  java -Xms512m -Xmx2048m -jar Juicebox_CLT.jar

Note: the -Xms512m flag sets the minimum memory heap size at 512 megabytes, and
the -Xmx2048m flag sets the maximum size at 2048 megabytes (2 gigabytes). These
values may be adjusted as appropriate for your machine.

-------------
Documentation
-------------
We have extensive documentation for how to use Juicebox at
http://www.aidenlab.org/juicebox/ including a video, a Quick Start Guide, and a
detailed tutorial.

The version of Juicebox included in this distribution has an additional
capability: you can load your own .hic files from your local machine.  To load
your own file, go to File -> Open.. and click the Local button at the bottom of
the dialog. From there you can navigate to the .hic file.

------------------------
Command Line Tools Usage
------------------------
In the command line tools, there are two functions: “pre”, for preprocessing,
which creates .hic files from text files that can be loaded into Juicebox, and
“dump”, which enables you to dump matrices from the .hic files to a text file.

The "juicebox" script can be used in place of the unwieldy
"java -Xms512m -Xmx2048m -jar Juicebox_CLT.jar" on Unix and Mac systems.

Running the command line tools without any arguments produces the following
usage message:
Juicebox Command Line Tools Usage:
  juicebox dump <observed/oe/pearson/norm> <NONE/VC/VC_SQRT/KR> <hicFile(s)> <chr1> <chr2> <BP/FRAG> <binsize>
  juicebox pre <options> <infile> <outfile> <genomeID>
<options>: -d only calculate intra chromosome (diagonal) [false]
  : -f <restriction site file> calculate fragment map
  : -m <int> only write cells with count above threshold m [0]
  : -q <int> filter by MAPQ score greater than or equal to q
  : -c <chromosome ID> only calculate map on specific chromosome
  : -h print help

---
Pre
---
The “pre” command takes three required arguments and a number of optional
arguments.  The optional arguments should go before the required ones.

The required arguments are:

&lt;infile>: Text file with paired contacts. The text file may be gzipped, and
  should be in the following 11 column format:<br><br>
                &lt;readname> &lt;str1> &lt;chr1> &lt;pos1> &lt;frag1> &lt;str2> &lt;chr2> &lt;pos2> &lt;frag2> &lt;mapq1> &lt;mapq2><br>
                str = strand (0 for forward, anything else for reverse)<br>
                chr = chromosome (must be a chromosome in the genome)<>br
                pos = position<br>
                frag = restriction site fragment<br>
                mapq = mapping quality score<br><br>
  If not using the restriction site file option, frag will be ignored.  If not using mapping quality filter, mapq will be ignored.<br><br>
&lt;outfile>: Name of outfile, should end with .hic.  This is the file you will
  load into Juicebox.
&lt;genomeID>: Must be one of hg18, hg19, hg38, dMel, mm9, mm10, anasPlat1, bTaurus3,
  canFam3, equCab2, galGal4, Pf3D7, sacCer3, sCerS288c, susScr3, or TAIR10

The optional arguments are:<br>
  -d Only calculate intra chromosome (diagonal) [false]<br>
  -f <restriction site file> Calculate fragment map.  Requires restriction site
    file; each line should start with the chromosome name followed by the position
    of each restriction site on that chromosome, in numeric order, and ending with
    the size of the chromosome<br>
  -m <int> Only write cells with count above threshold m [0]<br>
  -q <int> Filter by MAPQ score greater than or equal to q [not set]<br>
  -c <chromosome ID> Only calculate map on specific chromosome<br>

------------
Pre Examples
------------

  juicebox pre data/test.txt.gz data/test.hic hg19
This will produce a file in data/test.hic that can be loaded into Juicebox. It
will not include a fragment map and will not filter any of the reads.

  juicebox pre -q 10 data/test.txt.gz data/test.hic hg19
This will produce a file in data/test.hic that can be loaded into Juicebox.
Reads with one or more end with MAPQ < 10 will not be included. It will not
include a fragment map.

  juicebox pre -q 30 -f data/hg19_MboI.txt data/test.txt.gz data/test.hic hg19
This will produce a file in data/test.hic that can be loaded into Juicebox.
Reads with one or more end with MAPQ < 30 will not be included. The restriction
site file data/hg19_MboI.txt should list all of the restriction sites on the
hg19 genome, where each line should start with the chromosome name followed by
the position of each restriction site on that chromosome, in numeric order, and
ending with the size of the chromosome. When loaded into Juicebox, users will
be able to see the map at fragment resolution.

----
Dump
----
The “dump” command takes 7 arguments and produces text in sparse matrix format
(row, column, value).  Hi-C matrices are symmetric.  Here are the arguments:
* The kind of matrix you would like to dump.  Must be one of
  observed/oe/norm/expected.  The latter 2 will produce vectors.
* The kind of normalization you want to apply. Must be one of
  NONE/VC/VC_SQRT/KR. VC is vanilla coverage, VC_SQRT is square root of vanilla
  coverage, and KR is Knight-Ruiz or Balanced normalization.
* The .hic file or files you want to dump.  If you list multiple files, the
  matrices will be summed together then dumped
* chr1: The first chromosome.
* chr2: The second chromosome.
* The unit of resolution, BP/FRAG.  BP is base-pair delimited resolution and
  FRAG is fragment delimited.
* The bin size.  For BP, this must be one of <2500000, 1000000, 500000, 250000,
  100000, 50000, 25000, 10000, 5000> and for FRAG this must be one of <500, 200,
  100, 50, 20, 5, 2, 1>

-------------
Dump Examples
-------------
  juicebox dump observed KR filename.hic 1 1 BP 2500000 > chr1_2.5MB.txt
This will dump the observed matrix of chromosome 1 with KR normalization at
2.5Mb resolution to the file chr1_2.5MB.txt

  juicebox dump norm VC filename.hic X X BP 5000 > chrX_norm.txt
This will dump the VC normalization vector of chromosome X at 5Kb resolution to
the file chrX_norm.txt

  juicebox dump oe NONE filename.hic 2 2 FRAG 10 > chr2_oe.txt
This will dump the O/E matrix with no normalization of chromosome 2 at 10f
resolution.

  juicebox dump observed VC_SQRT filename.hic 1 5 BP 100000 > chr1_chr5.txt
This will dump the observed interchromosomal matrix of chromosome 1 vs
chromosome 5 with square root VC normalization at 100Kb resolution to the file
chr1_chr5.txt.

--------------------------------
Compiling Jars from Source Files
--------------------------------
1. You should have Java 1.8 JDK and Apache Ant installed on your system. See
   below for more information.
2. Go to the folder containing the Juicebox source files and edit the
   juicebox.properties file with the proper Java JDK Address.
3. Open the command line, navigate to the folder containing the build.xml file
   and type
     ant
   The process should take no more than a minute to build on most machines.
4. The jars are written to the directory out/.  You can change this by editing
   the build.xml file.

* Installing Java 1.8 JDK

For Windows/Mac/Linux, the Java 1.8 JDK can be installed from here:
http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html
(Alternative) For Ubuntu/LinuxMint
http://tecadmin.net/install-oracle-java-8-jdk-8-ubuntu-via-ppa/

* Installing Apache Ant
Mac
  Ant should be installed on most Macs. To verify installation via the command
  prompt, type
    ant -version
  If Ant is not on your Mac, install it via homebrew. At the command prompt, type
    brew update
    brew install ant
  You may need to install Homebrew (http://brew.sh/) on your machine
  See the following Stackoverflow post for more details:
  http://stackoverflow.com/questions/3222804/how-can-i-install-apache-ant-on-mac-os-x

Windows
  Installing Ant requires some minor changes to your system environment. Follow
  the instructions in this article:
  http://www.nczonline.net/blog/2012/04/12/how-to-install-apache-ant-on-windows/

Linux
  In the command prompt, type
    sudo apt-get install ant
  or
    sudo yum install ant
  depending on your package installer
