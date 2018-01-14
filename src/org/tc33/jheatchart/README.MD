Note

The APA heatmap plotting module in Juicebox Tools uses a modified version of JHeatChart developed by Tom Castle.
JHeatChart is open source under an LGPL license.
The main heatmap plotting of the standard Juicebox GUI does not use JHeatChart.

JHeatChart - https://github.com/tc33/jheatchart

- MSS


*******************************************************************************

            JHeatChart v.0.6            Tom Castle: www.tc33.org
                                                                           
*******************************************************************************
|                                                                             |
| JHeatChart is made available under a GPL license, for full details of the   |
| license see the included LICENSE text file.                                 |
|                                                                             |
|-----------------------------------------------------------------------------|

For the latest version of JHeatChart and any tutorials and examples that exist,
see the JHeatChart project pages at http://www.tc33.org. 

Contents of release:

    jheatchart-0.6
        | jheatchart-0.6.jar - the library. Put this anywhere on your class path.
        | src - directory containing all the java source.
            | ...
        | javadoc - directory containing the javadoc for this release.
            | ...
        | README - this file.
        | LICENSE - details of the GPL license this software is covered by.

Usage:

    Simply put the jheatchart jar file on your class path and then you are free 
    to create instances of HeatChart.
   
Example code:

    // Create some dummy data.
    double[][] data = new double[][]{{3,2,3,4,5,6},
                                     {2,3,4,5,6,7},
                                     {3,4,5,6,7,6},
                                     {4,5,6,7,6,5}};
    
    // Create our heat chart using our data.
    HeatChart chart = new HeatChart(data);
    
    // Customise the chart.
    chart.setTitle("This is my chart title");
    chart.setXAxisLabel("X Axis");
    chart.setYAxisLabel("Y Axis");
    
    // Output the chart to a file.
    chart.saveToFile(new File("my-chart.png"));
    