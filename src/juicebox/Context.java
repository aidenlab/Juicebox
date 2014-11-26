package juicebox;

//import juicebox.data.Chromosome;

import juicebox.windowui.HiCZoom;
import org.broad.igv.feature.Chromosome;

/**
 * @author jrobinso
 * @date Aug 11, 2010
 */
public class Context {


    private final Chromosome chromosome;
    private HiCZoom zoom;

     private double binOrigin = 0;

    public Context(Chromosome chromosome) {
        this.chromosome = chromosome;
    }

    public double getBinOrigin() {
        return binOrigin;
    }

    public void setBinOrigin(double binOrigin) {
        this.binOrigin = binOrigin;
    }

    public HiCZoom getZoom() {
        return zoom;
    }

    public void setZoom(HiCZoom zoom) {
        this.zoom = zoom;
    }

    public int getChrLength() {
        return chromosome.getLength();
    }


    public Chromosome getChromosome() {
        return chromosome;
    }

}
