package juicebox.data;

import juicebox.track.HiCGridAxis;
import juicebox.windowui.NormalizationType;
import java.util.ArrayList;
import java.util.List;

/** A display view of a trans matrix in the requested chromosome-name order. */
public final class TransposedMatrixZoomData extends MatrixZoomData {
    private final MatrixZoomData source;

    private TransposedMatrixZoomData(MatrixZoomData source) {
        super(source.getChr2(), source.getChr1(), source.getZoom(), source.blockBinCount,
                source.blockColumnCount, new int[0], new int[0], source.reader);
        this.source = source;
        setAverageCount(source.getAverageCount());
        setSumCount(source.getSumCount());
    }

    public static MatrixZoomData orient(MatrixZoomData source, int xIndex) {
        return source == null || source.getChr1Idx() == source.getChr2Idx() ||
                source.getChr1Idx() == xIndex
                ? source : new TransposedMatrixZoomData(source);
    }

    @Override public HiCGridAxis getXGridAxis() { return source.getYGridAxis(); }
    @Override public HiCGridAxis getYGridAxis() { return source.getXGridAxis(); }
    @Override public float getObservedValue(int x, int y, NormalizationType norm) {
        return source.getObservedValue(y, x, norm);
    }
    @Override public List<Block> getNormalizedBlocksOverlapping(long x0, long y0, long x1, long y1,
            NormalizationType norm, boolean important, boolean fill) {
        List<Block> blocks = source.getNormalizedBlocksOverlapping(y0, x0, y1, x1, norm, important, fill);
        if (blocks == null) return null;
        List<Block> result = new ArrayList<>();
        for (Block block : blocks) {
            List<ContactRecord> records = new ArrayList<>();
            for (final ContactRecord record : block.getContactRecords()) {
                records.add(new ContactRecord(record.getBinY(), record.getBinX(), record.getCounts()) {
                    @Override public java.math.BigInteger getExactCount() { return record.getExactCount(); }
                });
            }
            result.add(new Block(block.getNumber(), records, block.getUniqueRegionID() + "_transposed"));
        }
        return result;
    }
}
