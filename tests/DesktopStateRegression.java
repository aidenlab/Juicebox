package juicebox.state;

import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.DirectoryManager;
import juicebox.assembly.*;
import juicebox.gui.SuperAdapter;
import juicebox.track.*;
import juicebox.track.feature.*;
import juicebox.windowui.NormalizationHandler;
import org.broad.igv.renderer.DataRange;
import org.broad.igv.util.ResourceLocator;

import java.awt.Color;
import java.io.File;
import java.lang.reflect.Field;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public class DesktopStateRegression {
    private static void check(boolean value, String message) {
        if (!value) throw new AssertionError(message);
    }

    public static void main(String[] args) throws Exception {
        Path temp = Files.createTempDirectory("desktop-state-");
        org.broad.igv.DirectoryManager.setIgvDirectory(temp.toFile());
        // Keep autosaves isolated without changing the user's Java Preferences.
        Field directory = DirectoryManager.class.getDeclaredField("HIC_DIRECTORY");
        directory.setAccessible(true);
        directory.set(null, temp.toFile());
        HiCGlobals.xmlSavedStatesFile = temp.resolve("states.xml").toFile();
        StringBuilder xml = new StringBuilder("<SavedMaps><STATE SelectedPath=\"test\">");
        for (String field : State.stateVarNames) xml.append('<').append(field).append('>')
                .append(field).append(" value</").append(field).append('>');
        xml.append("</STATE></SavedMaps>");
        Files.write(HiCGlobals.xmlSavedStatesFile.toPath(), xml.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8));
        String[] parsed = XMLFileParser.parseXML("test");
        for (int i = 0; i < parsed.length; i++) check((State.stateVarNames[i] + " value").equals(parsed[i]),
                "compact XML reads every field: " + State.stateVarNames[i]);

        HiC hic = new HiC(null);
        HiCDataTrack first = track(hic, "first", Color.RED, 2, 20, true);
        HiCDataTrack second = track(hic, "second", Color.GREEN, -5, 5, false);
        String config = XMLFileHandling.serializeTrackConfig(Arrays.asList(first, second));
        HiCDataTrack reloadedFirst = track(hic, "first", Color.BLACK, 0, 1, false);
        HiCDataTrack reloadedSecond = track(hic, "second", Color.BLACK, 0, 1, true);
        LoadStateFromXMLFile.restoreTrackConfig(reloadedFirst, config);
        LoadStateFromXMLFile.restoreTrackConfig(reloadedSecond, config);
        check(reloadedFirst.getPosColor().equals(Color.RED) && reloadedFirst.getDataRange().getMaximum() == 20
                && reloadedFirst.getDataRange().isLog(), "first track settings restored on loaded instance");
        check(reloadedSecond.getPosColor().equals(Color.GREEN) && reloadedSecond.getDataRange().getMinimum() == -5
                && !reloadedSecond.getDataRange().isLog(), "second track retains its own settings");

        HiCGlobals.savedStatesList.clear();
        String state = "sample--name--currentState:$$map@@map.hic@@null$$chr1$$chr1$$BP$$1000$$0$$0$$1$$OBSERVED$$NONE$$0$$0$$1$$1$$1$$none$$none$$none";
        HiCGlobals.savedStatesList.add(state);
        HiCGlobals.savedStatesList.add(state.replace("$$1000$$", "$$2000$$"));
        XMLFileWriter.overwriteXMLFile();
        check(XMLFileParser.parseXML("sample--name")[7].equals("2000"), "overwritten state uses latest values and preserves hyphens");
        HiCGlobals.savedStatesList.clear(); // model a fresh desktop session
        String nextState = state.replace("sample--name", "new-state");
        XMLFileWriter.saveState(nextState + "$$KR");
        check(XMLFileParser.parseXML("sample--name")[7].equals("2000"), "saving after restart preserves old states");
        check(XMLFileParser.parseXML("new-state")[21].equals("KR"), "control normalization persisted");
        XMLFileWriter.saveState(nextState.replace("$$1000$$", "$$4000$$") + "$$VC");
        check(XMLFileParser.parseXML("new-state")[7].equals("4000")
                && XMLFileParser.parseXML("new-state")[21].equals("VC"), "state overwrite replaces only matching name");
        byte[] beforeFailure = Files.readAllBytes(HiCGlobals.xmlSavedStatesFile.toPath());
        try {
            XMLFileWriter.saveState("invalid state");
            throw new AssertionError("invalid state should fail");
        } catch (IllegalStateException expected) { }
        check(Arrays.equals(beforeFailure, Files.readAllBytes(HiCGlobals.xmlSavedStatesFile.toPath())),
                "failed state save leaves previous XML intact");

        Feature2D original = feature(10, Collections.singletonMap("label", "original"));
        Feature2DList input = new Feature2DList();
        input.add(0, 1, original);
        AnnotationLayer layer = new AnnotationLayer(input);
        layer.undo(); // no edit yet: should not create a backup
        Path backup = temp.resolve(HiCGlobals.BACKUP_FILE_STEM + layer.getId() + ".bedpe");
        check(!Files.exists(backup), "empty undo leaves autosave alone");
        layer.add(0, 1, feature(20, Collections.emptyMap()));
        check(Files.readAllLines(backup).size() == 3, "first edit backs up loaded and new features immediately");
        check(original.getAttribute("label").equals("original"), "imported attributes remain intact");
        layer.undo();
        check(layer.getNumberOfFeatures() == 1 && Files.readAllLines(backup).size() == 2,
                "undo supports chromosome zero and rewrites backup");
        Field dirty = AnnotationLayer.class.getDeclaredField("unsavedEdits");
        dirty.setAccessible(true);
        check(dirty.getBoolean(layer), "rewriting autosave retains dirty state");
        layer.add(0, 1, feature(30, Collections.singletonMap("new", "value")));
        java.util.List<String> lines = Files.readAllLines(backup);
        check(lines.size() == 3 && lines.get(0).endsWith("\tlabel\tnew"), "new attribute rewrites backup header");
        for (String line : lines) check(line.split("\t", -1).length == lines.get(0).split("\t", -1).length,
                "autosave columns match schema");
        check(layer.exportAnnotations(temp.resolve("export.bedpe").toString()), "annotation export");
        check(!Files.exists(backup), "export clears backup");
        layer.add(0, 1, feature(40, Collections.emptyMap()));
        check(Files.readAllLines(backup).size() == 4, "editing after export reopens writer and saves all features");
        check(!layer.exportAnnotations(temp.toString()) && Files.exists(backup),
                "failed export neither exits desktop nor discards backup");
        layer.clearAnnotations();

        Path lateBackup = temp.resolve(HiCGlobals.BACKUP_FILE_STEM + "42.bedpe");
        Files.write(lateBackup, Collections.singletonList("late layer"));
        check(juicebox.windowui.layers.UnsavedAnnotationWarning.getBackupFiles().length == 1,
                "backup discovery includes layer IDs above nine");
        java.lang.reflect.Method move = juicebox.windowui.layers.UnsavedAnnotationWarning.class
                .getDeclaredMethod("moveOldAnnotationFiles");
        move.setAccessible(true);
        String prefix = (String) move.invoke(new juicebox.windowui.layers.UnsavedAnnotationWarning(null));
        check(!Files.exists(lateBackup) && Files.exists(temp.resolve(prefix + "42.bedpe")),
                "saving backups moves every layer");

        Feature2DList empty = new Feature2DList();
        empty.parallelizedProcessLists((key, features) -> { throw new AssertionError("empty list has no jobs"); });
        Feature2DList many = new Feature2DList();
        for (int i = 0; i < 100; i++) many.add(i, i, feature(i, Collections.emptyMap()));
        AtomicInteger processed = new AtomicInteger();
        many.parallelizedProcessLists((key, features) -> processed.incrementAndGet());
        check(processed.get() == 100, "bounded feature workers finish all lists");

        SuperAdapter.setDatasetTitle("sample.hic");
        AssemblyStateTracker tracker = new AssemblyStateTracker(null, null);
        Field autosave = AssemblyStateTracker.class.getDeclaredField("autoSaveFileName");
        autosave.setAccessible(true);
        check(new File((String) autosave.get(tracker)).getName().equals("sample.review.autosave"),
                "assembly autosave accepts basename-only titles");
        Path assembly = temp.resolve("sample.assembly");
        Files.write(assembly, Arrays.asList("", ">scaffold\t1   3000000000", "  1  ", ""));
        AssemblyFileImporter importer = new AssemblyFileImporter(assembly.toString(), false);
        importer.importAssembly();
        check(importer.getAssemblyScaffoldHandler().getListOfScaffolds().get(0).getLength() == 3000000000L,
                "assembly whitespace and long lengths");
        Files.write(assembly, Arrays.asList(">scaffold 1 invalid", "1"));
        try {
            new AssemblyFileImporter(assembly.toString(), false).importAssembly();
            throw new AssertionError("invalid assembly silently installed partial data");
        } catch (IllegalArgumentException expected) { }
        System.out.println("Desktop state: XML, track settings, overwrite, annotation backups, bounded workers and assembly import passed");
    }

    private static HiCDataTrack track(HiC hic, String name, Color color, float min, float max, boolean log) {
        HiCCoverageDataSource source = new HiCCoverageDataSource(hic, NormalizationHandler.VC, false);
        source.setName(name);
        source.setColor(color);
        DataRange range = new DataRange(min, max);
        range.setType(log ? DataRange.Type.LOG : DataRange.Type.LINEAR);
        source.setDataRange(range);
        return new HiCDataTrack(hic, new ResourceLocator(name + ".wig"), source);
    }

    private static Feature2D feature(long position, Map<String, String> attributes) {
        return new Feature2D(Feature2D.FeatureType.DOMAIN, "chr1", position, position + 5,
                "chr2", position, position + 5, Color.BLUE, new HashMap<>(attributes));
    }
}
