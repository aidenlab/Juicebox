--------------------
Quick Start
--------------------
```
rm -r out
ant
ant sign
ant bundle
cd l4j
~/Downloads/launch4j/launch4j config_bcm.xml 
cd ../out/artifacts/Juicebox_jar/
codesign -s "Erez Aiden" Juicebox.app --deep
hdiutil create -srcfolder Juicebox.app Juicebox.dmg
cd ~/Dropbox\ \(Lab\ at\ Large\)/important_jars/
# move old jars, EXEs, and DMGs, update CHANGES
emacs -nw CHANGES
# XXXXXX should be date when file written i.e. 20170103
mv Juicebox_BCM.dmg Juicebox_BCMXXXXXXXX.dmg
mv Juicebox_BCM.exe Juicebox_BCMXXXXXXXX.exe
mv Juicebox.jar JuiceboxXXXXXXXX.jar
mv ~/Dropbox/Research/JuiceboxDev/l4j/Juicebox.exe Juicebox_BCM.exe
mv ~/Dropbox/Research/JuiceboxDev/out/artifacts/Juicebox_jar/Juicebox.dmg Juicebox_BCM.dmg
mv ~/Dropbox/Research/JuiceboxDev/out/artifacts/Juicebox_jar/Juicebox.jar Juicebox.jar
mv ~/Dropbox/Research/JuiceboxDev/out/artifacts/Juicebox_clt_jar/Juicebox.jar juicebox_tools.8.0.jar
```


--------------------
Creating Executables 
--------------------

0. Compile Jars from Source Files as described in Juicebox README.

* EXE Build (Windows)

1. Download the launch4j tarball 
   <https://sourceforge.net/projects/launch4j/files/launch4j-3/3.8/> and unzip it.
2. Run 

> ./launch4j/launch4j config.xml

3. Modify the config.xml file to set the properties file and other parameters.

* .app Build (Mac)

1. After making jars as described above, type 

> ant bundle

to make the Juicebox.app executable. 

2. Sign the app with codesign -s "Erez Aiden" Juicebox.app --deep
You must have things appropriately installed in your keychain.  Follow the instructions on the Apple Developer website.  Our csr and cer are in the Dropbox under important_jars but I'm not sure exactly how you would add both the csr and the cer to KeyChain so you might just have to do it from scratch (this is what I did, eventually).  Look for the ones created Wed Dec 23 2015

3. hdiutil create -size 210m -srcfolder Juicebox.app Juicebox.dmg
Size depends on what the size of your .app file is, make it big enough so this command doesn't fail.  You can also do Disk Utility -> New Image From Folder and choose Juicebox.app.

4. Modify the code under the bundle taskdef in the build.xml file to change properties and other parameters.

--------------------
Steps for creating Public Friendly Version
--------------------

1. Create a new branch
2. Delete anything under development (e.g. clustering)
3. Replace sensitive chrom.sizes
4. Remove any mention of assembly/sensitive projects
5. Change version number as appropriate - HiCGlobals, for display purposes only.

Note this is only in terms of jars/executables. For actaul code release / open-sourcing, we need to wipe other private files, .git histories, etc.

--------------------
Building a new IGV jar for use in Juicebox
-------------------
Two problems with IGV jar: signatures and classpaths in the MANIFEST.  If it was just the first, it would be a one-liner.  Instead: 
1 - Unzip IGV jar (be sure to get the "snapshot" build, currently at https://data.broadinstitute.org/igv/projects/snapshot/igv.jar )
```
mkdir tmp
mv igv.jar tmp
cd tmp
unzip igv.jar
```
2 - Remove META-INF/*.SF META-INF/*.DSA META-INF/*.RSA
```
rm META-INF/*.SF META-INF/*.DSA META-INF/*.RSA
```
3 - Go into the META-INF/MANIFEST file with your favorite editor and remove the Class-Path lines.
4 - Rezip the jar
```
rm igv.jar
jar cvf igv.jar ./*
```

--------------------
Building Different CUDA versions
--------------------
​​Right now JuiceboxDev (assuming what's in the repo now) is defaulting to 7.0

CUDA 7.0 is used on AWS
CUDA 7.5 is on most of our internal machines (Hailmary)
CUDA 8.0 is used on Adam now.
I think Rice was 7.5 as well, but not sure.


To change from 7.0 to 7.5, the build.xml and 
libraries for compilation need to be changed

---build.xml changes

Change the following line (has x2 occurrences, one for each jar build)

`<zipfileset src="${basedir}/lib/jcuda/jcuda-0.7.0.jar"/>`
to
`<zipfileset src="${basedir}/lib/jcuda/jcuda-0.7.5.jar"/>`
or
`<zipfileset src="${basedir}/lib/jcuda/jcuda-0.8.0RC.linux.jar"/>`

---jcuda lib changes

- Go to ~/lib/jcuda
- Delete all the non .zip files
- Unzip Archive.JCuda.0.7.5.zip
  this creates a new folder
- Move everything in this newly made folder to the folder above it
  (e.g. path should be ~/lib/jcuda/jcuda-0.7.5.jar, NOT ~/lib/jcuda/Archive.JCuda.0.7.5/jcuda-0.7.0.jar)


The jars are really the ones that matter, but I think it's easier to update everything here (especially if testing GPU stuff from within IntelliJ)

​And now it should be fine to build via ant​

