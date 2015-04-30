Juicebox
========

Visualization and analysis software for Hi-C data

`MainWindow.java` is the main method class for the visualization portion of the software.  
`HiCTools.java` is the main method class for the analysis portion.


Setup Instructions
========

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
* The GUI's main class is `MainWindow.java` - click the little `...` button next to the text box for main class, and type `MainWindow.java`.
* The CLT's main class is `HiCTools.java`.  
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
