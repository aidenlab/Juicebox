--------------------
Quick Start
--------------------
1. Run `package.sh <versionNum>`
versionNum lives in src/juicebox/HiCGlobals.
This will compile, sign, bundle, and create the DMG and EXE.
You will be prompted to change the EXE with a HEX editor, just follow
the instructions.
2. The new executables will be in out/artifacts/Juicebox_jar
3. move old jars, EXEs, and DMGs, update CHANGES in important_jars

------------------
Previous version
------------------
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

Modify the config.xml file to set the properties file and other parameters.

3. Signing is complicated. You'll need openssl and osslsigncode and will need to do a two step procedure.

  - openssl pkcs12 -in ~/Dropbox\ \(Lab\ at\ Large\)/important_jars/ErezSLieberman.p12 -nocerts -nodes -out certificate.pem
    (You can save this certificate and use it later, just don't forget your password)
  - Sign for the first time
     osslsigncode sign -certs ~/Dropbox\ \(Lab\ at\ Large\)/important_jars/erez_s_lieberman.pem -key certificate.pem \
     -askpass -n "Juicebox" -i http://aidenlab.org/ -in ~/Dropbox\ \(Lab\ at\ Large\)/important_jars/Juicebox.exe \
     -out signed.exe
  - Find the size of the signature in bytes i.e. sizeInBytesOf(signed.exe) - sizeInBytesOf(Juicebox.exe)
    You can use ls -l for this
  - Edit Juicebox.exe with favorite HEX editor to change last two bytes of exe i.e. the jar i.e. the zip end of 
    central directory to the size using littleendian byte order and save.  File size should remain the same.
    For example, if the size difference is 4384, the hex number is 0x1120; in Little Endian, this will be 20 11
  - Sign the modified Juicebox.exe using above osslsigncode again
   

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
2. Remove sensitive chrom.sizes. First, delete all contents of the chrom.sizes folder (/src/juicebox/tools/chrom.sizes). Next, go to the top level directory. You will see a compressed folder called PublicFriendlyChromSizes.zip. Extract its contents and copy them into the chrom.sizes folder which was just emptied. Recompile the project in IntelliJ.
3. Delete anything under development (e.g. clustering, APAvsDistance, etc.). These should all be in the dev folder, so it should usually be sufficient to just delete it and all its contents (src/juicebox/tools/dev). After deleting the folder, compile in IntelliJ and fix all the bugs/warnings (i.e. remove any calls to the dev/private folder). This may be tricky for some parts (especially the restriction enzymes section) See https://github.com/theaidenlab/JuiceboxDev/commit/fd930f5fac9af3df9f44cd87d4fc31e8df5d3ac3 for an example of what was deleted. (Aside: Any new sensitive project should be created in this dev directory to simplify this entire process for us.)
4. Remove any mention of assembly/sensitive projects (should have technically been taken care on in step 3, but a quick search for the word assembly in the whole project is easy to do)
5. Change version number as appropriate - HiCGlobals, for display purposes only.

Note this is only in terms of jars/executables.
For actual code release / open-sourcing, we need to wipe other private files, especially the .git histories, hidden files, internalREADME (i.e. me!), etc.

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
CUDA 8.0 is used on Adam; linux-x64
CUDA 8.0 is used on Rice; linux-ppc
CUDA 8.0 for Mac; apple-x86


To change from 7.0 to 7.5, the build.xml and 
libraries for compilation need to be changed

---build.xml changes

Change the following line (has x2 occurrences, one for each jar build)

`<zipfileset src="${basedir}/lib/jcuda/jcuda-0.7.0.jar"/>`
to
`<zipfileset src="${basedir}/lib/jcuda/jcuda-0.7.5.jar"/>`
or
`<zipfileset src="${basedir}/lib/jcuda/jcuda-0.8.0.jar"/>`
and for Power (Rice) will need
`<zipfileset src="${basedir}/lib/jcuda/jcuda-natives-0.8.0-linux-ppc_64.jar"/>`
and otherwise
`<zipfileset src="${basedir}/lib/jcuda/jcuda-natives-0.8.0-linux-x86_64.jar"/>`

---jcuda lib changes

- Go to ~/lib/jcuda
- Delete all the non .zip files
- Unzip Archive.JCuda.0.7.5.zip
  this creates a new folder
- Move everything in this newly made folder to the folder above it
  (e.g. path should be ~/lib/jcuda/jcuda-0.7.5.jar, NOT ~/lib/jcuda/Archive.JCuda.0.7.5/jcuda-0.7.0.jar)
- For JCuda 0.8, you should define if this is for x86 or ppc (Power8) architecture and add those natives: jcuda-natives-0.8.0-linux-ppc_64.jar or  jcuda-natives-0.8.0-linux-x86_64.jar

The jars are really the ones that matter, but I think it's easier to update everything here (especially if testing GPU stuff from within IntelliJ)

​And now it should be fine to build via ant​

