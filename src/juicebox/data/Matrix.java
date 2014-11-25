package juicebox.data;

import juicebox.HiC;
import juicebox.HiCZoom;

import java.util.*;

/**
 * @author jrobinso
 * @since Aug 12, 2010
 */
public class Matrix {

     final int chr1;
     final int chr2;
     List<MatrixZoomData> bpZoomData;
     List<MatrixZoomData> fragZoomData;

    /**
     * Constructor for creating a matrix from precomputed data.
     *
     * @param chr1
     * @param chr2
     * @param zoomDataList
     */
    public Matrix(int chr1, int chr2, List<MatrixZoomData> zoomDataList) {
        this.chr1 = chr1;
        this.chr2 = chr2;
        initZoomDataMap(zoomDataList);
    }

    public static String generateKey(int chr1, int chr2) {
        return "" + chr1 + "_" + chr2;
    }

    public String getKey() {
        return generateKey(chr1, chr2);
    }

    private void initZoomDataMap(List<MatrixZoomData> zoomDataList) {

        bpZoomData = new ArrayList<MatrixZoomData>();
        fragZoomData = new ArrayList<MatrixZoomData>();
        for (MatrixZoomData zd : zoomDataList) {
            if (zd.getZoom().getUnit() == HiC.Unit.BP) {
                bpZoomData.add(zd);
            } else {
                fragZoomData.add(zd);
            }

            // Zooms should be sorted, but in case they are not...
            Comparator<MatrixZoomData> comp = new Comparator<MatrixZoomData>() {
                @Override
                public int compare(MatrixZoomData o1, MatrixZoomData o2) {
                    return o2.getBinSize() - o1.getBinSize();
                }
            };
            Collections.sort(bpZoomData, comp);
            Collections.sort(fragZoomData, comp);
        }

    }

    public MatrixZoomData getFirstZoomData(HiC.Unit unit) {
        if(unit == HiC.Unit.BP) {
            return bpZoomData != null ? bpZoomData.get(0) : null;
        }
        else {
            return fragZoomData != null ? fragZoomData.get(0) : null;
        }

    }

    public MatrixZoomData getZoomData(HiCZoom zoom) {
        List<MatrixZoomData> zdList = (zoom.getUnit() == HiC.Unit.BP) ? bpZoomData : fragZoomData;
        //linear search for bin size, the lists are not large
        for (MatrixZoomData zd : zdList) {
            if (zd.getBinSize() == zoom.getBinSize()) {
                return zd;
            }
        }

        return null;
    }

    public int getNumberOfZooms(HiC.Unit unit) {
        return (unit == HiC.Unit.BP) ? bpZoomData.size() : fragZoomData.size();
    }

    public boolean isIntra() {
        return chr1 == chr2;
    }
}
