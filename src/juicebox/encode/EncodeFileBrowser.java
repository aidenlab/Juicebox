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


package juicebox.encode;

import com.jidesoft.swing.JideBoxLayout;
import juicebox.HiCGlobals;
import org.apache.log4j.Logger;
import org.broad.igv.Globals;
import org.broad.igv.ui.IGV;
import org.broad.igv.util.Pair;
import org.broad.igv.util.ResourceLocator;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import javax.swing.text.NumberFormatter;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.text.ParseException;
import java.util.*;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * @author Jim Robinson
 */
public class EncodeFileBrowser extends JDialog {

    private static final long serialVersionUID = 3049287764412827292L;
    private static final Map<String, EncodeFileBrowser> instanceMap = Collections.synchronizedMap(new HashMap<String, EncodeFileBrowser>());
    private static final NumberFormatter numberFormatter = new NumberFormatter();
    private static final HashSet<String> supportedGenomes = new HashSet<String>(Arrays.asList("hg19", "mm9", "hic"));
    private static Logger log = Logger.getLogger(EncodeFileBrowser.class);
    private final EncodeTableModel model;
    private JTable table;
    private JTextField filterTextField;
    private JLabel rowCountLabel;
    private boolean canceled;

    private EncodeFileBrowser(Frame owner, EncodeTableModel model) {
        super(owner);
        this.model = model;
        setModal(true);
        initComponents();
        init(model);
    }

    public synchronized static EncodeFileBrowser getInstance(String genomeId) throws IOException {

        String encodeGenomeId = getEncodeGenomeId(genomeId);
        EncodeFileBrowser instance = instanceMap.get(encodeGenomeId);
        if (instance == null) {
            Pair<String[], List<EncodeFileRecord>> records = getEncodeFileRecords(encodeGenomeId);
            if (records == null) {
                return null;
            }
            Frame parent = IGV.hasInstance() ? IGV.getMainFrame() : null;
            instance = new EncodeFileBrowser(parent, new EncodeTableModel(records.getFirst(), records.getSecond()));
            instanceMap.put(encodeGenomeId, instance);
        }

        return instance;
    }

    public synchronized static EncodeFileBrowser getHiCInstance() throws IOException {
        EncodeFileBrowser instance = instanceMap.get("hic");
        if (instance == null) {
            Pair<String[], List<EncodeFileRecord>> records = getEncodeFileRecords("hic");
            if (records == null) {
                return null;
            }
            Frame parent = IGV.hasInstance() ? IGV.getMainFrame() : null;
            instance = new EncodeFileBrowser(parent, new EncodeTableModel(records.getFirst(), records.getSecond()));
            instanceMap.put("hic", instance);
        }
        return instance;
    }

    public static boolean genomeSupported(String genomeId) {
        return genomeId != null && supportedGenomes.contains(getEncodeGenomeId(genomeId));
    }

    private static String getEncodeGenomeId(String genomeId) {
        if (genomeId.equals("b37") || genomeId.equals("1kg_v37")) return "hg19";
        else return genomeId;
    }

    private static Pair<String[], List<EncodeFileRecord>> getEncodeFileRecords(String genomeId) throws IOException {

        InputStream is = null;

        try {

            is = EncodeFileBrowser.class.getResourceAsStream("encode." + genomeId + ".txt");
            if (is == null) {
                return null;
            }
            BufferedReader reader = new BufferedReader(new InputStreamReader(is), HiCGlobals.bufferSize);

            String[] headers = Globals.tabPattern.split(reader.readLine());

            List<EncodeFileRecord> records = new ArrayList<EncodeFileRecord>(20000);
            String nextLine;
            while ((nextLine = reader.readLine()) != null) {
                if (!nextLine.startsWith("#")) {

                    String[] tokens = Globals.tabPattern.split(nextLine, -1);
                    String path = tokens[0];

                    // Filter BAMs for hic
                    if (path == null || path.endsWith("bam")) continue;

                    Map<String, String> attributes = new HashMap<String, String>();
                    for (int i = 0; i < headers.length; i++) {
                        String value = i < tokens.length ? tokens[i] : "";
                        if (value.length() > 0) {
                            attributes.put(headers[i], value);
                        }
                    }

                    final EncodeFileRecord record = new EncodeFileRecord(path, attributes);
                    if (record.hasMetaData()) records.add(record);

                }

            }
            return new Pair<String[], List<EncodeFileRecord>>(headers, records);
        } finally {
            if (is != null) is.close();
        }
    }

    public static void main(String[] args) throws IOException {
        getInstance("hg19").setVisible(true);
    }

    private void init(final EncodeTableModel model) {
        setModal(true);
        setTitle("Encode Production Data");

        table.setAutoCreateRowSorter(true);
        table.setModel(model);
        table.setRowSorter(model.getSorter());
        try {
            rowCountLabel.setText(numberFormatter.valueToString(table.getRowCount()) + " rows");
        } catch (ParseException e) {
            e.printStackTrace();
        }

        table.setRowSelectionAllowed(false);
        table.setColumnSelectionAllowed(false);

        filterTextField.getDocument().addDocumentListener(
                new DocumentListener() {
                    public void changedUpdate(DocumentEvent e) {
                        updateFilter();
                    }

                    public void insertUpdate(DocumentEvent e) {
                        updateFilter();
                    }

                    public void removeUpdate(DocumentEvent e) {
                        updateFilter();
                    }
                });

    }

    /**
     * Update the row filter regular expression from the expression in
     * the text box.
     */
    private void updateFilter() {


        RowFilter<EncodeTableModel, Object> rf = null;
        //If current expression doesn't parse, don't update.
        try {
            rf = new RegexFilter(filterTextField.getText());
        } catch (java.util.regex.PatternSyntaxException e) {
            return;
        } catch (ClassCastException e) {
            return;
        }
        model.getSorter().setRowFilter(rf);

        try {
            rowCountLabel.setText(numberFormatter.valueToString(table.getRowCount()) + " rows");
        } catch (ParseException e) {
            e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
        }
    }

    private void loadButtonActionPerformed(ActionEvent e) {
        canceled = false;
        setVisible(false);
    }

    private void cancelButtonActionPerformed(ActionEvent e) {
        canceled = true;
        setVisible(false);
    }

    public boolean isCanceled() {
        return canceled;
    }

    /**
     * @return the list of VISIBLE selected records.  Filtered records are not returned even if record.selected == true
     * @throws java.io.IOException
     */
    public List<EncodeFileRecord> getSelectedRecords() {

        List<EncodeFileRecord> selectedRecords = new ArrayList<EncodeFileRecord>();
        List<EncodeFileRecord> allRecords = model.getRecords();

        int rowCount = table.getRowCount();
        for (int i = 0; i < rowCount; i++) {
            int modelIdx = table.convertRowIndexToModel(i);
            EncodeFileRecord record = allRecords.get(modelIdx);
            if (record.isSelected()) {
                selectedRecords.add(record);
            }
        }

        return selectedRecords;
    }

    public void checkEncodeTracks(String track) {
        List<EncodeFileRecord> allRecords = model.getRecords();
        int rowCount = table.getRowCount();

        for (int i = 0; i < rowCount; i++) {
            int modelIdx = table.convertRowIndexToModel(i);
            EncodeFileRecord record = allRecords.get(modelIdx);
            if (record.getTrackName().contains(track)) {
                record.setSelected(true);
            }
        }
    }

    public void remove(ResourceLocator locator) {
        List<EncodeFileRecord> allRecords = model.getRecords();
        int rowCount = table.getRowCount();
        int i = 0;
        boolean notFound = true;
        while (i < rowCount && notFound) {
            int modelIdx = table.convertRowIndexToModel(i);
            EncodeFileRecord record = allRecords.get(modelIdx);
            ResourceLocator rl = new ResourceLocator(record.getPath());
            rl.setName(record.getTrackName());
            if (rl.equals(locator)) {
                record.setSelected(false);
                notFound = false;
            }
            i++;
        }
    }

    private void initComponents() {

        JPanel dialogPane = new JPanel();
        JPanel contentPanel = new JPanel();
        JScrollPane scrollPane1 = new JScrollPane();
        table = new JTable();
        JPanel filterPanel = new JPanel();
        JLabel filterLabel = new JLabel();
        filterTextField = new JTextField();
        rowCountLabel = new JLabel();
        JPanel buttonBar = new JPanel();
        JButton okButton = new JButton();
        JButton cancelButton = new JButton();

        getRootPane().setDefaultButton(okButton);

        final String filterToolTip = "Enter multiple filter strings separated by commas.  e.g.  GM12878, ChipSeq";
        filterLabel.setToolTipText(filterToolTip);
        filterTextField.setToolTipText(filterToolTip);

        //======== this ========
        Container contentPane = getContentPane();
        contentPane.setLayout(new BorderLayout());

        //======== dialogPane ========

        dialogPane.setBorder(new EmptyBorder(12, 12, 12, 12));
        dialogPane.setLayout(new BorderLayout());

        //======== contentPanel ========

        contentPanel.setLayout(new BorderLayout(0, 10));

        //======== scrollPane1 ========

        scrollPane1.setViewportView(table);

        contentPanel.add(scrollPane1, BorderLayout.CENTER);

        //======== panel1 ========

        filterPanel.setLayout(new JideBoxLayout(filterPanel, JideBoxLayout.X_AXIS, 5));

        //---- label1 ----
        filterLabel.setText("Filter:");
        filterPanel.add(filterLabel, JideBoxLayout.FIX);

        //---- filterTextField ----
        filterPanel.add(filterTextField, JideBoxLayout.VARY);

        rowCountLabel.setHorizontalAlignment(JLabel.RIGHT);
        JPanel sillyPanel = new JPanel();
        sillyPanel.setLayout(new JideBoxLayout(sillyPanel, JideBoxLayout.X_AXIS, 0));
        sillyPanel.setPreferredSize(new Dimension(100, 28));
        sillyPanel.add(rowCountLabel, JideBoxLayout.VARY);

        filterPanel.add(sillyPanel, JideBoxLayout.FIX);

        contentPanel.add(filterPanel, BorderLayout.NORTH);

        dialogPane.add(contentPanel, BorderLayout.CENTER);

        //======== buttonBar ========

        buttonBar.setBorder(new EmptyBorder(12, 0, 0, 0));
        buttonBar.setLayout(new GridBagLayout());
        ((GridBagLayout) buttonBar.getLayout()).columnWidths = new int[]{0, 85, 80};
        ((GridBagLayout) buttonBar.getLayout()).columnWeights = new double[]{1.0, 0.0, 0.0};

        //---- okButton ----
        okButton.setText("Load");
        okButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                loadButtonActionPerformed(e);
            }
        });
        buttonBar.add(okButton, new GridBagConstraints(1, 0, 1, 1, 0.0, 0.0,
                GridBagConstraints.CENTER, GridBagConstraints.BOTH,
                new Insets(0, 0, 0, 5), 0, 0));

        //---- cancelButton ----
        cancelButton.setText("Cancel");
        cancelButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                cancelButtonActionPerformed(e);
            }
        });
        buttonBar.add(cancelButton, new GridBagConstraints(2, 0, 1, 1, 0.0, 0.0,
                GridBagConstraints.CENTER, GridBagConstraints.BOTH,
                new Insets(0, 0, 0, 0), 0, 0));

        dialogPane.add(buttonBar, BorderLayout.SOUTH);

        contentPane.add(dialogPane, BorderLayout.CENTER);
        setSize(700, 620);
        setLocationRelativeTo(getOwner());
    }

    private class RegexFilter extends RowFilter<EncodeTableModel, Object> {

        List<Pair<String, Matcher>> matchers;

        RegexFilter(String text) {

            if (text == null) {
                throw new IllegalArgumentException("Pattern must be non-null");
            }
            matchers = new ArrayList<Pair<String, Matcher>>();
            String[] tokens = Globals.whitespacePattern.split(text);
            for (String t : tokens) {
                // If token contains an = sign apply to specified column only
                String column = "*";
                String value = t.trim();
                if (t.contains("=")) {
                    String[] kv = Globals.equalPattern.split(t);
                    if (kv.length > 1) {
                        column = kv[0].trim();
                        value = kv[1].trim();
                    } else {
                        value = kv[0];  // Value is column name until more input is entered
                    }
                }

                matchers.add(new Pair<String, Matcher>(column, Pattern.compile("(?i)" + value).matcher("")));
            }

        }

        /**
         * Include row if each matcher succeeds in at least one column.  In other words all the conditions
         * are combined with "and"
         *
         * @param value value to check
         * @return if matched
         */
        @Override
        public boolean include(Entry<? extends EncodeTableModel, ?> value) {

            for (Pair<String, Matcher> entry : matchers) {
                String column = entry.getFirst();
                Matcher matcher = entry.getSecond();


                // Search for a match in at least one column.  The first column is the checkbox.
                boolean found = false;  // Pessimistic
                int nColumns = table.getColumnCount();
                for (int index = 1; index < nColumns; index++) {

                    // Include column headings in search.  This is to prevent premature filtering when entering a
                    // specific column condition (e.g. cataType=ChipSeq)
                    matcher.reset(table.getColumnName(index).toLowerCase());
                    if (matcher.find()) {
                        found = true;
                        break;
                    }

                    boolean wildcard = column.equals("*");
                    if (wildcard || column.equalsIgnoreCase(table.getColumnName(index))) {
                        matcher.reset(value.getStringValue(index));
                        if (matcher.find()) {
                            found = true;
                            break;
                        }
                    }
                }
                if (!found) return false;
            }
            return true;  // If we get here we matched them all
        }

    }

}
