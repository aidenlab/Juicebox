import juicebox.HiC;
import juicebox.data.*;
import juicebox.data.basics.Chromosome;
import juicebox.windowui.*;
import java.util.*;
import java.math.BigInteger;

public class DesktopV10Regression {
    static Dataset load(String path) throws Exception {
        DatasetReader reader = DatasetReaderFactory.getReader(Collections.singletonList(path));
        if (!(reader instanceof DatasetReaderV10)) throw new AssertionError("Factory did not choose v10");
        return reader.read();
    }
    static Map<String, ContactRecord> contacts(MatrixZoomData zd, NormalizationType norm, long end) {
        Map<String, ContactRecord> result = new TreeMap<>();
        for (Block block : zd.getNormalizedBlocksOverlapping(0, 0, end, end, norm, false, false)) {
            for (ContactRecord record : block.getContactRecords()) {
                if (record.getBinX() <= end && record.getBinY() <= end)
                    result.put(record.getBinX() + "_" + record.getBinY(), record);
            }
        }
        return result;
    }
    static void check(boolean condition, String message) { if (!condition) throw new AssertionError(message); }
    public static void main(String[] args) throws Exception {
        String fixtures = args[0];
        Dataset ds = load(fixtures + "/cis.hic");
        Chromosome chr = ds.getChromosomeHandler().getChromosomeFromIndex(0);
        check(chr.getName().equals("chrA"), "chr0 was rewritten");
        check(ds.getChromosomeHandler().getChromosomeArrayWithoutAllByAll().length == 2, "chr0 was dropped");
        check(ds.getZoom(HiC.Unit.BP, 0).getBinSize() == 20, "display ladder is reversed");
        Matrix matrix = ds.getMatrix(chr, chr);
        MatrixZoomData fine = matrix.getZoomData(new HiCZoom(HiC.Unit.BP, 10));
        MatrixZoomData derived = matrix.getZoomData(new HiCZoom(HiC.Unit.BP, 20));
        check(fine.getXGridAxis().getBinCount() == 8, "wrong partial bin geometry");
        check(contacts(fine, NormalizationHandler.NONE, 7).size() == 3, "materialized cells missing");
        check(contacts(derived, NormalizationHandler.NONE, 3).size() == 3, "derived cells missing");
        check(contacts(fine, NormalizationHandler.VC, 7).get("3_3").getCounts() == 1.25f, "normalization incorrect");
        check(contacts(fine, new NormalizationType("missing", "missing"), 7).isEmpty(), "missing normalization exposed raw counts");
        int count = 0;
        for (Iterator<ContactRecord> it = ((V10MatrixZoomData)derived).directIterator(); it.hasNext();) { it.next(); count++; }
        check(count == 3, "derived iterator missing cells");
        java.io.StringWriter exported = new java.io.StringWriter();
        derived.dump(new java.io.PrintWriter(exported), null, NormalizationHandler.NONE, MatrixType.OBSERVED, false, null, null, false);
        check(exported.toString().trim().split("\n").length == 3, "derived sparse export empty");
        exported = new java.io.StringWriter();
        derived.dump(new java.io.PrintWriter(exported), null, NormalizationHandler.NONE, MatrixType.OBSERVED, false, null, null, true);
        check(exported.toString().trim().split("\n").length == 4, "derived dense export has wrong extent");
        Dataset transFixture = load(fixtures + "/trans.hic");
        Chromosome transFirst = transFixture.getChromosomeHandler().getChromosomeFromIndex(0);
        check(contacts(transFixture.getMatrix(transFirst, transFirst).getZoomData(fine.getZoom()), NormalizationHandler.NONE, 7).isEmpty(), "omitted pair did not produce an empty map");
        check(ds.getNormalizationVector(0, fine.getZoom(), NormalizationHandler.VC).getData().get(0) == 2, "vector adapter wrong");
        Dataset large = load(fixtures + "/bigint.hic");
        Chromosome first = large.getChromosomeHandler().getChromosomeFromIndex(0);
        ContactRecord exact = contacts(large.getMatrix(first, first).getZoomData(fine.getZoom()), NormalizationHandler.NONE, 7).get("0_0");
        check(exact.getExactCount().equals(BigInteger.ONE.shiftLeft(53).add(BigInteger.ONE)), "exact count lost");
        Dataset frag = load(fixtures + "/frag.hic");
        first = frag.getChromosomeHandler().getChromosomeFromIndex(0);
        MatrixZoomData fragmentZoom = frag.getMatrix(first, first).getZoomData(new HiCZoom(HiC.Unit.FRAG, 1));
        check(contacts(fragmentZoom, NormalizationHandler.NONE, 7).size() == 3, "FRAG contacts missing");
        check(fragmentZoom.getXGridAxis().getGenomicEnd(7) == first.getLength(), "last fragment omitted");
        Dataset expected = load(fixtures + "/expected.hic");
        check(expected.getExpectedValues(fine.getZoom(), NormalizationHandler.NONE).getExpectedValue(0, 0) == 5, "expected adapter wrong");
        Dataset summedExpected = DatasetReaderFactory.getReader(Arrays.asList(fixtures + "/expected.hic", fixtures + "/expected.hic")).read();
        check(summedExpected.getExpectedValues(fine.getZoom(), NormalizationHandler.NONE).getExpectedValue(0, 0) == 10, "combined expected lost chromosome scaling");
        ds.getStatistics(); // no recursion when the statistics attribute is absent
        Dataset combined = DatasetReaderFactory.getReader(Arrays.asList(fixtures + "/cis.hic", fixtures + "/cis.hic")).read();
        Chromosome combinedChr = combined.getChromosomeHandler().getChromosomeFromIndex(0);
        Matrix combinedMatrix = combined.getMatrix(combinedChr, combinedChr);
        check(contacts(combinedMatrix.getZoomData(fine.getZoom()), NormalizationHandler.NONE, 7).get("3_3").getCounts() == 10, "combined materialized did not sum");
        check(contacts(combinedMatrix.getZoomData(derived.getZoom()), NormalizationHandler.NONE, 3).get("1_1").getCounts() == 10, "combined derived did not sum");
        check(contacts(combinedMatrix.getZoomData(fine.getZoom()), NormalizationHandler.VC, 7).get("3_3").getCounts() == 2.5f, "combined normalized did not sum");
        // Resolve a foreign chromosome object by name, never by its numeric index.
        check(ds.getMatrix(new Chromosome(1, "chrA", 80), new Chromosome(1, "chrA", 80)) == matrix, "foreign chromosome indices were used");
        Dataset legacy = DatasetReaderFactory.getReader(Collections.singletonList(fixtures + "/legacy-v8.hic")).read();
        check(legacy.getVersion() == 8, "legacy factory broken");
        Dataset legacyAll = load(fixtures + "/legacy-all.hic");
        Chromosome legacyOverview = legacyAll.getChromosomeHandler().getChromosomeFromIndex(0);
        MatrixZoomData legacyOverviewZoom = legacyAll.getMatrix(legacyOverview, legacyOverview).getFirstZoomData(HiC.Unit.BP);
        check(legacyOverviewZoom.getBinSize() == 35 && contacts(legacyOverviewZoom, NormalizationHandler.NONE, 499).size() == 3,
              "legacy All selected an empty global resolution");
        if (args.length > 1) {
            Dataset real = load(args[1]);
            Chromosome a = real.getChromosomeHandler().getChromosomeFromIndex(0);
            Chromosome b = real.getChromosomeHandler().getChromosomeFromIndex(1);
            check(a.getName().equals("chr1"), "real chr1 replaced");
            check(real.getChromosomeHandler().getChromosomeArrayWithoutAllByAll().length == 25, "real chromosome lost");
            MatrixZoomData cis = real.getMatrix(a, a).getZoomData(new HiCZoom(HiC.Unit.BP, 5000));
            check(!contacts(cis, NormalizationHandler.NONE, 19).isEmpty(), "real cis empty");
            check(!contacts(cis, NormalizationHandler.VC, 19).isEmpty(), "real normalized cis empty");
            MatrixZoomData trans = real.getMatrix(a, b).getZoomData(new HiCZoom(HiC.Unit.BP, 2500000));
            check(contacts(trans, NormalizationHandler.NONE, 100).size() == 9118, "real trans mismatch");
            if (real.getChromosomeHandler().getAllOverview() != null) {
                Chromosome all = real.getChromosomeHandler().getChromosomeFromName("All");
                check(all != null && real.getChromosomeHandler().getChromosomeFromName("ALL") == all, "hg38 All alias lookup failed");
                MatrixZoomData overview = real.getMatrix(all, all).getFirstZoomData();
                check(overview.getBinSize() == real.getChromosomeHandler().getAllOverview().finestResolution, "real All selected collapsed level");
                check(!contacts(overview, NormalizationHandler.NONE, 3).isEmpty(), "real All empty");
            }
        }
        System.out.println("Desktop v10: factory, chromosomes, ladder, blocks, derivation, normalization, FRAG, expected, exact counts, legacy and real file passed");
    }
}
