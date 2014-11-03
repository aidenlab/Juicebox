package juicebox.tools;

import org.broad.igv.feature.Chromosome;
import juicebox.data.Dataset;
import juicebox.data.DatasetReaderV2;
import juicebox.NormalizationType;
import juicebox.HiC;
import juicebox.HiCZoom;
import juicebox.data.NormalizationVector;
import juicebox.data.ExpectedValueFunction;
import juicebox.data.ExpectedValueFunctionImpl;

import java.io.IOException;
import java.util.Map;

/**
 * Utility functions to dump various bits of a hic file to stdout or file(s)
 *
 * @author jrobinso
 *         Date: 2/11/13
 *         Time: 2:01 PM
 */

public class HiCFileUtils {

    DatasetReaderV2 reader;
    Dataset dataset;

    public static void main(String[] args) throws IOException {
        HiCFileUtils utils = new HiCFileUtils(args[0]);
        //utils.dumpNormalizationVectors("KR", "1", "BP", 250000);
        utils.dumpExpectedVectors("KR", "BP", 1000000);
    }

    public HiCFileUtils(String hicfile) throws IOException {
        reader = new DatasetReaderV2(hicfile);
        dataset = reader.read();
    }

    public void dumpNormalizationVectors(String type, String chrName, String unitName, int binSize) {

        NormalizationType no = NormalizationType.valueOf(type);

        Chromosome chromosome = findChromosome(chrName);
        HiC.Unit unit = HiC.Unit.valueOf(unitName);
        HiCZoom zoom = new HiCZoom(unit, binSize);
        NormalizationVector nv = dataset.getNormalizationVector(chromosome.getIndex(),zoom, no);
        String label = "Normalization vector: type = " + type + " chr = " + chrName +
                " resolution = " + binSize + " " + unitName;
        System.out.println(label);
        double [] data = nv.getData();
        for(int i=0; i<data.length; i++) {
            System.out.println(data[i]);
        }

    }

    public void dumpExpectedVectors(String type, String unit, int binSize) throws IOException {


        Map<String, ExpectedValueFunction> expValFunMap = dataset.getExpectedValueFunctionMap();
        for (Map.Entry<String, ExpectedValueFunction> entry : expValFunMap.entrySet()) {


            ExpectedValueFunctionImpl ev = (ExpectedValueFunctionImpl) entry.getValue();

            if (ev.getUnit().toString().equals(unit) && ev.getBinSize() == binSize && ev.getNormalizationType().equals(type)) {
                String label = ev.getNormalizationType() + "\t" + ev.getUnit().toString() + "\t" + ev.getBinSize();

                System.out.println("Norm factors: " + label);
                for (Map.Entry<Integer, Double> nf : ev.getNormFactors().entrySet()) {
                    System.out.println(nf.getKey() + "\t" + nf.getValue());
                }

                System.out.println("Expected values: " + label);
                double[] values = ev.getExpectedValues();
                for (int i = 0; i < values.length; i++) {
                    System.out.println(values[i]);
                }

                System.out.println("End expected values: " + label);
                System.out.println();
            }

        }

    }


    private Chromosome findChromosome(String name) {

        for(Chromosome chr : dataset.getChromosomes()) {
            if(chr.getName().equals(name)) return chr;
        }
        return null;
    }

}
