## Internal script for compiling and packaging DMG and EXE
## Compiles with appropriate properties file depending on if
## -b flag is set (if set, compile for BCM)
## Set the two globals at the top for your system
#!/bin/bash

set -e
shopt -s extglob

## GLOBALS: Set for your system
# executable for launch4j, which bundles EXE
LAUNCH4J_EXE="/Users/nchernia/Downloads/launch4j/launch4j"
# Lab At Large location, needed for certificates
LAL_DROPBOX="/Users/nchernia/Dropbox (Lab at Large)/"

printHelpAndExit() {
    echo "Usage: ${0##*/} <version_number> -bh"
    echo "       -v <version_number>: 1.6.1 e.g.  Required"
    echo "       -b: BCM-only version"
    echo "       -h: Print this help and exit"
    exit "$1"
}

while getopts "v:bh" opt; do
    case $opt in
	v) VERSION=$OPTARG ;;
	h) printHelpAndExit 0;;
	b) BCM=1 ;;
	[?]) printHelpAndExit 1;;
	esac
done

if [ -z "$VERSION" ]
then
   printHelpAndExit 1
fi

if [ -z "${BCM}" ]
then
    CONFIG_FILE="config.xml"
else
    CONFIG_FILE="config_bcm.xml"
fi

# these shouldn't need to be changed
BASE_DIR=$(pwd)
APP_NAME="Juicebox"
ARTIFACT_DIR="${BASE_DIR}/out/artifacts/Juicebox_jar"
DMG_BACKGROUND_IMG="Juicebox_bg.png"

# compile, bundle, sign 
cd "${BASE_DIR}"
rm -r out
ant
#ant sign -- don't sign for DMG, problem with slowness 
if [ -z "${BCM}" ]
then
    ant bundle -Dversion="$VERSION"
else
    ant bundlebcm # this is deprecated
fi

cd "${ARTIFACT_DIR}"

APP_EXE="${APP_NAME}.app/Contents/MacOS/JavaAppLauncher" 
VOL_NAME="${APP_NAME}_${VERSION}"  
DMG_TMP="${VOL_NAME}-temp.dmg"
DMG_FINAL="${VOL_NAME}.dmg"        
STAGING_DIR="./Install"            

# clear out any old data
rm -rf "${STAGING_DIR}" "${DMG_TMP}" "${DMG_FINAL}"

codesign -s "Erez Aiden" "${APP_NAME}".app --deep
 
# copy over the stuff we want in the final disk image to our staging dir
mkdir -p "${STAGING_DIR}"
cp -rpf "${APP_NAME}.app" "${STAGING_DIR}"
# ... cp anything else you want in the DMG - documentation, etc.

pushd "${STAGING_DIR}"
 
# strip the executable
#echo "Stripping ${APP_EXE}..."
#strip -u -r "${APP_EXE}"
 
# compress the executable if we have upx in PATH
#  UPX: http://upx.sourceforge.net/
#if hash upx 2>/dev/null; then
#   echo "Compressing (UPX) ${APP_EXE}..."
#   upx -9 "${APP_EXE}"
#fi
 
# ... perform any other stripping/compressing of libs and executables
 
popd
# figure out how big our DMG needs to be
#  assumes our contents are at least 1M!
SIZE=`du -sh "${STAGING_DIR}" | sed 's/\([0-9]*\)M\(.*\)/\1/'`
SIZE=`echo "${SIZE} + 1.0" | bc | awk '{print int($1+0.5)}'`
 
if [ $? -ne 0 ]; then
   echo "Error: Cannot compute size of staging dir"
   exit
fi

# create the temp DMG file
hdiutil create -srcfolder "${STAGING_DIR}" -volname "${VOL_NAME}" -fs HFS+ \
      -fsargs "-c c=64,a=16,e=16" -format UDRW -size ${SIZE}M "${DMG_TMP}"
 
echo "Created DMG: ${DMG_TMP}"
 
# mount it and save the device
DEVICE=$(hdiutil attach -readwrite -noverify "${DMG_TMP}" | \
         egrep '^/dev/' | sed 1q | awk '{print $1}')
 
sleep 2

# add a link to the Applications dir
echo "Add link to /Applications"
pushd /Volumes/"${VOL_NAME}"
ln -s /Applications
popd
 
# add a background image
mkdir /Volumes/"${VOL_NAME}"/.background
cp "${BASE_DIR}/${DMG_BACKGROUND_IMG}" /Volumes/"${VOL_NAME}"/.background/
 
# tell the Finder to resize the window, set the background,
#  change the icon size, place the icons in the right position, etc.
echo '
   tell application "Finder"
     tell disk "'${VOL_NAME}'"
           open
           set current view of container window to icon view
           set toolbar visible of container window to false
           set statusbar visible of container window to false
           set the bounds of container window to {400, 100, 920, 460}
           set viewOptions to the icon view options of container window
           set arrangement of viewOptions to not arranged
           set icon size of viewOptions to 72
           set background picture of viewOptions to file ".background:'${DMG_BACKGROUND_IMG}'"
           set position of item "'${APP_NAME}'.app" of container window to {140, 170}
           set position of item "Applications" of container window to {380, 170}
           close
           open
           update without registering applications
           delay 2
     end tell
   end tell
' | osascript
 
sync

# unmount it
hdiutil detach "${DEVICE}"
 
# now make the final image a compressed disk image
echo "Creating compressed image"
hdiutil convert "${DMG_TMP}" -format UDZO -imagekey zlib-level=9 -o "${DMG_FINAL}"
 
# clean up
rm -rf "${DMG_TMP}"
rm -rf "${STAGING_DIR}"
 
echo 'Done creating DMG'

###
### BUNDLE EXE
###

cd "${BASE_DIR}"
ant sign
cd "${BASE_DIR}"/l4j

# clean up any old versions
if ls *.exe 1> /dev/null 2>&1 
then
    rm *.exe 
fi

# Package exe
"${LAUNCH4J_EXE}" "${CONFIG_FILE}"

# Sign.  Uncomment below to change signature
#openssl pkcs12 -in ${LAL_DROPBOX}/important_jars/ErezSLieberman.p12 -nocerts\
# -nodes -out ${LAL_DROPBOX}/important_jars/certificate.pem
# Sign for the first time.  Password juicebox. To create new password, 
# uncomment above line, change "-pass" below
osslsigncode sign -certs "${LAL_DROPBOX}"/important_jars/erez_s_lieberman.pem \
     -key "${LAL_DROPBOX}"/important_jars/certificate.pem -pass juicebox \
     -n ${APP_NAME} -i http://aidenlab.org/ -in "${BASE_DIR}"/l4j/Juicebox.exe \
     -out "${BASE_DIR}"/l4j/signed.exe
# Find the size of the signature in bytes 
# i.e. sizeInBytesOf(signed.exe) - sizeInBytesOf(Juicebox.exe)
sizeInBytesSigned=$(ls -l "${BASE_DIR}"/l4j/signed.exe | awk '{print $5}')
sizeInBytesUnsigned=$(ls -l "${BASE_DIR}"/l4j/Juicebox.exe | awk '{print $5}')
# Print difference in Little Endian hex
echo "Edit ${BASE_DIR}/l4j/Juicebox.exe "
echo "with your favorite HEX editor to change last two bytes of the exe."
# This tells the zip file that there is a signature and what size. 
echo "Edit with the following value, save, and press enter to continue:"
awk -v s1=$sizeInBytesSigned -v s2=$sizeInBytesUnsigned  'BEGIN{str=sprintf("%x", s1-s2); x=substr(str,3,2); x=x substr(str,1,2); print x}'
read line
# Sign the modified Juicebox.exe using above osslsigncode again
osslsigncode sign -certs "${LAL_DROPBOX}"/important_jars/erez_s_lieberman.pem \
     -key "${LAL_DROPBOX}"/important_jars/certificate.pem -pass juicebox \
     -n "Juicebox" -i http://aidenlab.org/ -in "${BASE_DIR}"/l4j/Juicebox.exe \
     -out "${BASE_DIR}"/l4j/signed.exe
mv "${BASE_DIR}"/l4j/signed.exe "${ARTIFACT_DIR}"/Juicebox\ ${VERSION}.exe
mv "${ARTIFACT_DIR}"/Juicebox.jar "${ARTIFACT_DIR}"/Juicebox\ ${VERSION}.jar
echo "Done. The packaged executables live in ${ARTIFACT_DIR}"