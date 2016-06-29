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

Note this is only in terms of jars/executables. For actaul code release / open-sourcing, we need to wipe other private files, .git histories, etc.
