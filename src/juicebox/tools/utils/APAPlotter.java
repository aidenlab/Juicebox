/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2015 Broad Institute, Aiden Lab
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.tools.utils;

import juicebox.data.Matrix;
import org.tc33.jheatchart.HeatChart;

import javax.imageio.IIOImage;
import javax.imageio.ImageIO;
import javax.imageio.ImageWriteParam;
import javax.imageio.ImageWriter;
import javax.imageio.stream.FileImageOutputStream;
import java.awt.*;
import java.awt.font.FontRenderContext;
import java.awt.font.GlyphVector;
import java.awt.geom.AffineTransform;
import java.awt.geom.Arc2D;
import java.awt.image.BufferedImage;
import java.awt.image.BufferedImageOp;
import java.awt.image.ImageObserver;
import java.awt.image.RenderedImage;
import java.awt.image.renderable.RenderableImage;
import java.io.File;
import java.io.IOException;
import java.text.AttributedCharacterIterator;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.Map;

/**
 * Created by muhammadsaadshamim on 2/19/15.     956 227 8502
 */
public class APAPlotter {

    static double[][] data =
            {{0, 0, 0, 0, 1, 1, 0, 1, 2, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1},
            {1, 0, 0, 0, 1, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
            {0, 0, 1, 0, 0, 1, 0, 1, 0, 3, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0},
            {1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0},
            {0, 0, 0, 0, 2, 0, 0, 0, 1, 1, 0, 2, 1, 3, 0, 0, 0, 2, 1, 0, 1},
            {0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0},
            {0, 2, 0, 0, 1, 0, 1, 2, 1, 0, 1, 0, 1, 2, 1, 2, 1, 0, 0, 0, 0},
            {0, 0, 0, 0, 1, 0, 1, 1, 0, 3, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1},
            {1, 0, 0, 0, 0, 1, 0, 2, 0, 0, 5, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0},
            {2, 2, 0, 1, 0, 1, 2, 0, 2, 1, 4, 1, 2, 0, 1, 1, 1, 1, 1, 0, 0},
            {0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 3, 2, 0, 0, 0, 1, 0, 1, 2},
            {1, 2, 0, 0, 1, 1, 1, 4, 1, 1, 0, 0, 0, 3, 0, 2, 0, 0, 1, 0, 2},
            {0, 1, 0, 0, 0, 0, 0, 2, 0, 3, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0},
            {1, 2, 0, 0, 1, 0, 2, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 2, 0, 1},
            {1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0},
            {0, 1, 0, 1, 0, 2, 1, 1, 2, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0},
            {0, 0, 0, 0, 1, 2, 2, 1, 0, 1, 1, 1, 0, 0, 0, 1, 2, 1, 0, 0, 0},
            {0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
            {1, 1, 1, 0, 2, 2, 1, 0, 1, 1, 3, 0, 1, 0, 0, 0, 0, 1, 0, 0, 2},
            {1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 3, 1, 1, 1, 1, 3, 1, 1, 0, 1}};

    public static void run(){

        HeatChart map = new HeatChart(data);
        map.setLowValueColour(Color.WHITE);
        map.setHighValueColour(Color.RED);
        map.setTitle("N=2330(2330)/3331, P2LL: 5.538");
        map.setXValues(-data.length/2,1);
        map.setYValues(-data.length/2,1);

        HeatChart colorBar = generateColorBar(map);

        //double min = APAUtils.
        // Step 3: Output the chart to a file.
        try {

            BufferedImage apaImage = new BufferedImage(600,500, BufferedImage.TYPE_3BYTE_BGR);

            Graphics2D apaGraphics = apaImage.createGraphics();

            apaGraphics.drawImage(map.getChartImage(),0,0,500,500,null );
            apaGraphics.drawImage(colorBar.getChartImage(),500,0,100,500,null );

            saveJpeg(apaImage,new File("/Users/muhammadsaadshamim/Desktop/img.jpg"),1.0f);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static int nColors = 10;

    private static HeatChart generateColorBar(HeatChart map) {

        double minVal = HeatChart.min(data);
        double maxVal = HeatChart.max(data);

        HeatChart colorBar = new HeatChart(generateColumnData(minVal, maxVal, nColors+1));
        colorBar.setHighValueColour(map.getHighValueColour());
        colorBar.setLowValueColour(map.getLowValueColour());
        colorBar.setYValues(maxVal, -(maxVal - minVal)/nColors);

        return colorBar;
    }

    private static double[][] generateColumnData(double minVal, double maxVal, int n) {
        double[][] newData = new double[n][1];
        int iter = 0;
        for(double i = maxVal; i > minVal; i -= (maxVal-minVal)/(n- 1)){
            newData[iter][0] = i;
            iter++;
        }
        return newData;
    }

    public static Double[] convert(double[] row) {
        Double[] copy = new Double[row.length];
        System.arraycopy(row,0,copy,0,row.length);
        return copy;
    }

    private static void saveJpeg(BufferedImage image, File outputFile, float quality) throws IOException {
        Iterator<ImageWriter> iter = ImageIO.getImageWritersByFormatName("jpeg");
        ImageWriter writer = (ImageWriter) iter.next();
        ImageWriteParam iwp = writer.getDefaultWriteParam();
        iwp.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
        iwp.setCompressionQuality(quality);

        FileImageOutputStream output = new FileImageOutputStream(outputFile);
        writer.setOutput(output);
        IIOImage imageOut = new IIOImage(image, null, null);
        writer.write(null, imageOut, iwp);
        writer.dispose();

    }
}