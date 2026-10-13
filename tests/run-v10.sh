#!/usr/bin/env sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
jar="$root/out/artifacts/Juicebox_jar/Juicebox.jar"
if [ ! -f "$jar" ]; then
  printf '%s\n' 'Build first: ant -Djdk.home.1.8="$(/usr/libexec/java_home)" desktop-v10' >&2
  exit 1
fi
classes=$(mktemp -d "${TMPDIR:-/tmp}/desktop-v10-tests.XXXXXX")
trap 'rm -rf "$classes"' EXIT HUP INT TERM
javac -proc:none -cp "$jar" -d "$classes" \
  "$root/tests/DesktopV10Regression.java" \
  "$root/tests/DesktopScaledAllRegression.java" \
  "$root/tests/DesktopRenderRegression.java"
java -ea -Djava.awt.headless=true -Djava.io.tmpdir="$classes" -cp "$classes:$jar" DesktopV10Regression "$root/tests/data" "$@"
java -ea -Djava.awt.headless=true -Djava.io.tmpdir="$classes" -cp "$classes:$jar" DesktopScaledAllRegression "$root/tests/data/pseudo-all.hic" "$root/tests/data/kilobase-all.hic"
java -ea -Djava.awt.headless=true -Djava.io.tmpdir="$classes" -cp "$classes:$jar" juicebox.mapcolorui.DesktopRenderRegression "$root/tests/data/expected.hic"
