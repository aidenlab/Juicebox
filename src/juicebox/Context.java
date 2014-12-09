/*
 * Copyright (C) 2011-2014 Aiden Lab - All Rights Reserved
 * SOFTWARE COPYRIGHT NOTICE
 * This software and its documentation are the copyright of the Aiden Lab All rights are reserved.
 *
 * This software is supplied without any warranty or guaranteed support whatsoever. Aiden Lab is not responsible for its use, misuse, or functionality.
 *
 * This software is licensed under the terms of the GNU Lesser General Public License (LGPL),
 * Version 2.1 which is available at http://www.opensource.org/licenses/lgpl-2.1.php.
 */

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
