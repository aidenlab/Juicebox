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
        System.out.println("Desktop scaled All: finest zoom, exact counts, layout and boundaries passed");
    }
}
