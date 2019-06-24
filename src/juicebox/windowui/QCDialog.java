/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2019 Broad Institute, Aiden Lab
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

package juicebox.windowui;

import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.MainWindow;
import juicebox.data.Dataset;
import juicebox.data.ExpectedValueFunction;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.LogarithmicAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import javax.swing.text.html.HTMLEditorKit;
import javax.swing.text.html.StyleSheet;
import java.awt.*;
import java.util.NoSuchElementException;
import java.util.Scanner;

public class QCDialog extends JDialog {
    private static final long serialVersionUID = -436533197805525691L;
    private static final long[] logXAxis = {10, 12, 15, 19, 23, 28, 35, 43, 53, 66, 81, 100, 123, 152, 187, 231,
            285, 351, 433, 534, 658, 811, 1000, 1233,
            1520, 1874, 2310, 2848, 3511, 4329, 5337, 6579, 8111, 10000, 12328, 15199, 18738, 23101, 28480, 35112,
            43288, 53367, 65793, 81113, 100000, 123285, 151991, 187382, 231013, 284804, 351119, 432876, 533670,
            657933, 811131, 1000000, 1232847, 1519911, 1873817, 2310130, 2848036, 3511192, 4328761, 5336699,
            6579332, 8111308, 10000000, 12328467, 15199111, 18738174, 23101297, 28480359, 35111917, 43287613,
            53366992, 65793322, 81113083, 100000000, 123284674, 151991108, 187381742, 231012970, 284803587,
            351119173, 432876128, 533669923, 657933225, 811130831, 1000000000, 1232846739, 1519911083,
            1873817423, 2310129700L, 2848035868L, 3511191734L, 4328761281L, 5336699231L, 6579332247L, 8111308308L,
            10000000000L};

    public QCDialog(MainWindow mainWindow, HiC hic, String title, boolean isControl) {
        super(mainWindow);

        Dataset dataset = hic.getDataset();
        if (isControl) {
            dataset = hic.getControlDataset();
        }

        String text = dataset.getStatistics();
        String textDescription = null;
        String textStatistics = null;
        String graphs = dataset.getGraphs();
        JTextPane description = null;
        JTabbedPane tabbedPane = new JTabbedPane();
        HTMLEditorKit kit = new HTMLEditorKit();


        StyleSheet styleSheet = kit.getStyleSheet();
        styleSheet.addRule("table { border-collapse: collapse;}");
        styleSheet.addRule("body {font-family: Sans-Serif; font-size: 12;}");
        styleSheet.addRule("td { padding: 2px; }");
        styleSheet.addRule("th {border-bottom: 1px solid #000; text-align: left; background-color: #D8D8D8; font-weight: normal;}");

        if (text != null) {
            int split = text.indexOf("</table>") + 8;
            textDescription = text.substring(0, split);
            textStatistics = text.substring(split);
            description = new JTextPane();
            description.setPreferredSize(new Dimension(400,400));
            description.setEditable(false);
            description.setContentType("text/html");
            description.setEditorKit(kit);

            description.setText(textDescription);
            JScrollPane pane1 = new JScrollPane(description);
            tabbedPane.addTab("About Library", pane1);


            JTextPane textPane = new JTextPane();
            textPane.setEditable(false);
            textPane.setContentType("text/html");

            textPane.setEditorKit(kit);
            textPane.setText(textStatistics);
            JScrollPane pane = new JScrollPane(textPane);
            tabbedPane.addTab("Statistics", pane);
        }
        boolean success = true;
        if (graphs != null) {

            long[] A = new long[2000];
            long sumA = 0;
            long[] mapq1 = new long[201];
            long[] mapq2 = new long[201];
            long[] mapq3 = new long[201];
            long[] intraCount = new long[100];
            final XYSeries intra = new XYSeries("Intra Count");
            final XYSeries leftRead = new XYSeries("Left");
            final XYSeries rightRead = new XYSeries("Right");
            final XYSeries innerRead = new XYSeries("Inner");
            final XYSeries outerRead = new XYSeries("Outer");
            final XYSeries allMapq = new XYSeries("All MapQ");
            final XYSeries intraMapq = new XYSeries("Intra MapQ");
            final XYSeries interMapq = new XYSeries("Inter MapQ");

            Scanner scanner = new Scanner(graphs);
            try {
                while (!scanner.next().equals("[")) ;

                for (int idx = 0; idx < 2000; idx++) {
                    A[idx] = scanner.nextLong();
                    sumA += A[idx];
                }

                while (!scanner.next().equals("[")) ;
                for (int idx = 0; idx < 201; idx++) {
                    mapq1[idx] = scanner.nextInt();
                    mapq2[idx] = scanner.nextInt();
                    mapq3[idx] = scanner.nextInt();

                }

                for (int idx = 199; idx >= 0; idx--) {
                    mapq1[idx] = mapq1[idx] + mapq1[idx + 1];
                    mapq2[idx] = mapq2[idx] + mapq2[idx + 1];
                    mapq3[idx] = mapq3[idx] + mapq3[idx + 1];
                    allMapq.add(idx, mapq1[idx]);
                    intraMapq.add(idx, mapq2[idx]);
                    interMapq.add(idx, mapq3[idx]);
                }
                while (!scanner.next().equals("[")) ;
                for (int idx = 0; idx < 100; idx++) {
                    int tmp = scanner.nextInt();
                    if (tmp != 0) innerRead.add(logXAxis[idx], tmp);
                    intraCount[idx] = tmp;
                    tmp = scanner.nextInt();
                    if (tmp != 0) outerRead.add(logXAxis[idx], tmp);
                    intraCount[idx] += tmp;
                    tmp = scanner.nextInt();
                    if (tmp != 0) rightRead.add(logXAxis[idx], tmp);
                    intraCount[idx] += tmp;
                    tmp = scanner.nextInt();
                    if (tmp != 0) leftRead.add(logXAxis[idx], tmp);
                    intraCount[idx] += tmp;
                    if (idx > 0) intraCount[idx] += intraCount[idx - 1];
                    if (intraCount[idx] != 0) intra.add(logXAxis[idx], intraCount[idx]);
                }
            } catch (NoSuchElementException exception) {
                JOptionPane.showMessageDialog(getParent(), "Graphing file improperly formatted", "Error", JOptionPane.ERROR_MESSAGE);
                success = false;
            }

            if (success) {
                final XYSeriesCollection readTypeCollection = new XYSeriesCollection();
                readTypeCollection.addSeries(innerRead);
                readTypeCollection.addSeries(outerRead);
                readTypeCollection.addSeries(leftRead);
                readTypeCollection.addSeries(rightRead);

                final JFreeChart readTypeChart = ChartFactory.createXYLineChart(
                        "Types of reads vs distance",          // chart title
                        "Distance (log)",               // domain axis label
                        "Binned Reads (log)",                  // range axis label
                        readTypeCollection,                  // data
                        PlotOrientation.VERTICAL,
                        true,                     // include legend
                        true,
                        false
                );

                final XYPlot readTypePlot = readTypeChart.getXYPlot();

                readTypePlot.setDomainAxis(new LogarithmicAxis("Distance (log)"));
                readTypePlot.setRangeAxis(new LogarithmicAxis("Binned Reads (log)"));
                Color backgroundColor = HiCGlobals.isDarkulaModeEnabled ? Color.BLACK : Color.WHITE;
                readTypePlot.setBackgroundPaint(backgroundColor);
                readTypePlot.setRangeGridlinePaint(Color.lightGray);
                readTypePlot.setDomainGridlinePaint(Color.lightGray);
                readTypeChart.setBackgroundPaint(backgroundColor);
                readTypePlot.setOutlinePaint(Color.DARK_GRAY);
                final ChartPanel chartPanel = new ChartPanel(readTypeChart);

                final XYSeriesCollection reCollection = new XYSeriesCollection();
                final XYSeries reDistance = new XYSeries("Distance");

                for (int i = 0; i < A.length; i++) {
                    if (A[i] != 0) reDistance.add(i, A[i] / (float) sumA);
                }
                reCollection.addSeries(reDistance);

                final JFreeChart reChart = ChartFactory.createXYLineChart(
                        "Distance from closest restriction enzyme site",          // chart title
                        "Distance (bp)",               // domain axis label
                        "Fraction of Reads (log)",                  // range axis label
                        reCollection,                  // data
                        PlotOrientation.VERTICAL,
                        true,                     // include legend
                        true,
                        false
                );

                final XYPlot rePlot = reChart.getXYPlot();
                rePlot.setDomainAxis(new NumberAxis("Distance (bp)"));
                rePlot.setRangeAxis(new LogarithmicAxis("Fraction of Reads (log)"));
                rePlot.setBackgroundPaint(backgroundColor);
                rePlot.setRangeGridlinePaint(Color.lightGray);
                rePlot.setDomainGridlinePaint(Color.lightGray);
                reChart.setBackgroundPaint(backgroundColor);
                rePlot.setOutlinePaint(Color.darkGray);
                final ChartPanel chartPanel2 = new ChartPanel(reChart);

                final XYSeriesCollection intraCollection = new XYSeriesCollection();

                intraCollection.addSeries(intra);

                final JFreeChart intraChart = ChartFactory.createXYLineChart(
                        "Intra reads vs distance",          // chart title
                        "Distance (log)",               // domain axis label
                        "Cumulative Sum of Binned Reads (log)",                  // range axis label
                        intraCollection,                  // data
                        PlotOrientation.VERTICAL,
                        true,                     // include legend
                        true,
                        false
                );

                final XYPlot intraPlot = intraChart.getXYPlot();
                intraPlot.setDomainAxis(new LogarithmicAxis("Distance (log)"));
                intraPlot.setRangeAxis(new NumberAxis("Cumulative Sum of Binned Reads (log)"));
                intraPlot.setBackgroundPaint(backgroundColor);
                intraPlot.setRangeGridlinePaint(Color.lightGray);
                intraPlot.setDomainGridlinePaint(Color.lightGray);
                intraChart.setBackgroundPaint(backgroundColor);
                intraPlot.setOutlinePaint(Color.darkGray);
                final ChartPanel chartPanel3 = new ChartPanel(intraChart);

                final XYSeriesCollection mapqCollection = new XYSeriesCollection();
                mapqCollection.addSeries(allMapq);
                mapqCollection.addSeries(intraMapq);
                mapqCollection.addSeries(interMapq);

                final JFreeChart mapqChart = ChartFactory.createXYLineChart(
                        "MapQ Threshold Count",          // chart title
                        "MapQ threshold",               // domain axis label
                        "Count",                  // range axis label
                        mapqCollection,                  // data
                        PlotOrientation.VERTICAL,
                        true,                     // include legend
                        true,                     // include tooltips
                        false
                );

                final XYPlot mapqPlot = mapqChart.getXYPlot();
                mapqPlot.setBackgroundPaint(backgroundColor);
                mapqPlot.setRangeGridlinePaint(Color.lightGray);
                mapqPlot.setDomainGridlinePaint(Color.lightGray);
                mapqChart.setBackgroundPaint(backgroundColor);
                mapqPlot.setOutlinePaint(Color.darkGray);
                final ChartPanel chartPanel4 = new ChartPanel(mapqChart);


                tabbedPane.addTab("Pair Type", chartPanel);
                tabbedPane.addTab("Restriction", chartPanel2);
                tabbedPane.addTab("Intra vs Distance", chartPanel3);
                tabbedPane.addTab("MapQ", chartPanel4);
            }
        }

        final ExpectedValueFunction df;
        if (isControl) {
            df = dataset.getExpectedValues(hic.getZoom(), hic.getControlNormalizationType());
        } else {
            df = dataset.getExpectedValues(hic.getZoom(), hic.getObsNormalizationType());
        }


        if (df != null) {
            double[] expected = df.getExpectedValues();
            final XYSeriesCollection collection = new XYSeriesCollection();
            final XYSeries expectedValues = new XYSeries("Expected");
            for (int i = 0; i < expected.length; i++) {
                if (expected[i] > 0) expectedValues.add(i + 1, expected[i]);
            }
            collection.addSeries(expectedValues);
            String title1 = "Expected at " + hic.getZoom() + " norm " + hic.getObsNormalizationType();
            final JFreeChart readTypeChart = ChartFactory.createXYLineChart(
                    title1,          // chart title
                    "Distance between reads (log)",               // domain axis label
                    "Genome-wide expected (log)",                  // range axis label
                    collection,                  // data
                    PlotOrientation.VERTICAL,
                    false,                     // include legend
                    true,
                    false
            );
            final XYPlot readTypePlot = readTypeChart.getXYPlot();

            readTypePlot.setDomainAxis(new LogarithmicAxis("Distance between reads (log)"));
            readTypePlot.setRangeAxis(new LogarithmicAxis("Genome-wide expected (log)"));
            Color backgroundColor = HiCGlobals.isDarkulaModeEnabled ? Color.BLACK : Color.WHITE;
            readTypePlot.setBackgroundPaint(backgroundColor);
            readTypePlot.setRangeGridlinePaint(Color.lightGray);
            readTypePlot.setDomainGridlinePaint(Color.lightGray);
            readTypeChart.setBackgroundPaint(backgroundColor);
            readTypePlot.setOutlinePaint(Color.darkGray);
            final ChartPanel chartPanel5 = new ChartPanel(readTypeChart);

            tabbedPane.addTab("Expected", chartPanel5);
        }

        if (text == null && graphs == null) {
            JOptionPane.showMessageDialog(this, "Sorry, no metrics are available for this dataset", "Error", JOptionPane.ERROR_MESSAGE);
            setVisible(false);
            dispose();

        } else {
            getContentPane().add(tabbedPane);
            pack();
            setModal(false);
            setLocation(100, 100);
            setTitle(title);
            setVisible(true);
        }
    }
}
