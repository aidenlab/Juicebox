import juicebox.HiC;
import juicebox.data.*;
import juicebox.data.basics.Chromosome;
import juicebox.windowui.*;
import java.math.BigInteger;
import java.util.Collections;
public class DesktopScaledAllRegression {
    public static void main(String[] args) throws Exception {
        Dataset ds = DatasetReaderFactory.getReader(Collections.singletonList(args[0])).read();
        ChromosomeHandler h = ds.getChromosomeHandler();
        assert h.getAllOverview() != null && h.getAllOverview().chromosomeIndex == 2;
        assert h.getChromosomeArrayWithoutAllByAll().length == 2;
        Chromosome all = h.getChromosomeFromIndex(2);
        Matrix matrix = ds.getMatrix(all, all);
        MatrixZoomData zd = matrix.getFirstZoomData(HiC.Unit.BP);
        assert zd.getBinSize() == 1;
        assert zd.getXGridAxis().getBinCount() == 18;
        assert Double.isNaN(ds.getMatrix(h.getChromosomeFromIndex(0), all).getFirstZoomData().getExpectedValueForTrans());
        assert h.hasSameOverviewLayout(h);
        ChromosomeHandler withoutConvention = new ChromosomeHandler(java.util.Arrays.asList(h.getChromosomeArray()), "test", false, false, true);
        assert !h.hasSameOverviewLayout(withoutConvention);
        assert h.getOverviewChromosome(0).getIndex() == 0;
        assert h.getOverviewChromosome(10).getIndex() == 1;
        assert h.getOverviewPosition(h.getChromosomeFromIndex(1), 11) == 10;
        assert h.getChromosomeBoundaries()[0] == 10 && h.getChromosomeBoundaries()[1] == 18;
        BigInteger sum = BigInteger.ZERO;
        int n = 0;
        for (Block b : zd.getNormalizedBlocksOverlapping(0,0,17,17,NormalizationHandler.NONE,false,false))
            for (ContactRecord r : b.getContactRecords()) {
                assert r.getExactCount() != null;
                sum = sum.add(r.getExactCount()); n++;
            }
        assert n == 3 && sum.equals(BigInteger.ONE.shiftLeft(53).add(BigInteger.valueOf(49)));
        if (args.length > 1) {
            Dataset kb = DatasetReaderFactory.getReader(Collections.singletonList(args[1])).read();
            ChromosomeHandler kh = kb.getChromosomeHandler();
            MatrixZoomData kz = kb.getMatrix(kh.getChromosomeFromIndex(2), kh.getChromosomeFromIndex(2))
                .getFirstZoomData(HiC.Unit.BP);
            assert kh.getChromosomeFromIndex(2).getLength() == 170;
            assert kz.getBinSize() == 17 && kz.getXGridAxis().getBinCount() == 10;
            assert kh.getChromosomeBoundaries()[0] == 95 && kh.getChromosomeBoundaries()[1] == 170;
            assert kh.getOverviewPosition(kh.getChromosomeFromIndex(1), 100) == 5000;
            sum = BigInteger.ZERO; n = 0;
            for (Block block : kz.getNormalizedBlocksOverlapping(0, 0, 9, 9, NormalizationHandler.NONE, false, false))
                for (ContactRecord record : block.getContactRecords()) {
                    assert record.getBinX() < 10 && record.getBinY() < 10;
                    sum = sum.add(record.getExactCount()); ++n;
                }
            assert n == 2 && sum.equals(BigInteger.ONE.shiftLeft(53).add(BigInteger.valueOf(49)));
            n = 0;
            for (Block block : kz.getNormalizedBlocksOverlapping(5, 9, 5, 9, NormalizationHandler.NONE, false, false))
                for (ContactRecord record : block.getContactRecords())
                    if (record.getBinX() == 5 && record.getBinY() == 9) ++n;
            assert n == 1;
            Dataset combined = DatasetReaderFactory.getReader(java.util.Arrays.asList(args[1], args[1])).read();
            ChromosomeHandler ch = combined.getChromosomeHandler();
            MatrixZoomData combinedZoom = combined.getMatrix(ch.getChromosomeFromIndex(2), ch.getChromosomeFromIndex(2)).getFirstZoomData(HiC.Unit.BP);
            BigInteger doubled = BigInteger.ZERO;
            for (Block block : combinedZoom.getNormalizedBlocksOverlapping(0, 0, 9, 9, NormalizationHandler.NONE, false, false))
                for (ContactRecord record : block.getContactRecords()) doubled = doubled.add(record.getExactCount());
            assert doubled.equals(sum.multiply(BigInteger.valueOf(2)));
            java.io.StringWriter text = new java.io.StringWriter();
            kz.dump(new java.io.PrintWriter(text), null, NormalizationHandler.NONE, MatrixType.OBSERVED,
                    false, new long[4], null, false);
            assert text.toString().contains("85\t153\t32");
        }
        System.out.println("Desktop scaled All: finest zoom, exact counts, layout and boundaries passed");
    }
}
