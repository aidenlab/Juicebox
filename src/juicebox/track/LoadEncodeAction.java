package juicebox.track;

import juicebox.HiC;
import juicebox.data.Dataset;
import juicebox.encode.EncodeFileBrowser;
import juicebox.encode.EncodeFileRecord;
import org.apache.log4j.Logger;
import org.broad.igv.track.AttributeManager;
import org.broad.igv.ui.util.MessageUtils;
import org.broad.igv.util.ResourceLocator;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.io.IOException;
import java.util.*;
import java.util.List;

/**
 * @author jrobinso
 *         Date: 3/13/14
 *         Time: 9:57 AM
 */
public class LoadEncodeAction extends AbstractAction {

    private static final Logger log = Logger.getLogger(LoadEncodeAction.class);
    private static final long serialVersionUID = 42L;
    private static final Map<String, Color> colors;

    static {
        colors = new HashMap<String, Color>();
        colors.put("H3K27AC", new Color(200, 0, 0));
        colors.put("H3K27ME3", new Color(200, 0, 0));
        colors.put("H3K36ME3", new Color(0, 0, 150));
        colors.put("H3K4ME1", new Color(0, 150, 0));
        colors.put("H3K4ME2", new Color(0, 150, 0));
        colors.put("H3K4ME3", new Color(0, 150, 0));
        colors.put("H3K9AC", new Color(100, 0, 0));
        colors.put("H3K9ME1", new Color(100, 0, 0));
    }

    private final Component owner;
    private final HiC hic;
    private String genome;
    private HashSet<ResourceLocator> loadedLocators;

    public LoadEncodeAction(String s, Component owner, HiC hic) {
        super(s);
        this.owner = owner;
        this.hic = hic;
        this.genome = null;

    }

    private LoadEncodeAction(String s, Component owner, HiC hic, String genome) {
        super(s);
        this.owner = owner;
        this.hic = hic;
        this.genome = genome;

    }

    @Override
    public void actionPerformed(ActionEvent e) {
        if (hic.getDataset() == null) {
            JOptionPane.showMessageDialog(owner, "File must be loaded to load annotations", "Error", JOptionPane.ERROR_MESSAGE);
            return;
        }

        if (genome == null) {
            genome = "hg19";   // initial guess
            Dataset ds = hic.getDataset();
            if (ds != null && ds.getGenomeId() != null) {
                genome = ds.getGenomeId();
            }
        }
        hic.setEncodeAction(this);
        String[] visibleAttributes = {"dataType", "cell", "antibody", "lab"};
        try {
            EncodeFileBrowser browser = EncodeFileBrowser.getInstance(genome);

            if (browser == null) {
                MessageUtils.showMessage("Encode tracks are not available for " + genome);
                return;
            }

            browser.setVisible(true);
            if (browser.isCanceled()) return;

            java.util.List<EncodeFileRecord> records = browser.getSelectedRecords();

            if (records.size() > 0) {
                if (loadedLocators == null) {
                    loadedLocators = new HashSet<ResourceLocator>();
                }

                List<ResourceLocator> locators = null;
                for (EncodeFileRecord record : records) {
                    ResourceLocator rl = new ResourceLocator(record.getPath());
                    rl.setName(record.getTrackName());

                    final String antibody = record.getAttributeValue("antibody");
                    if (antibody != null) {
                        rl.setColor(colors.get(antibody.toUpperCase()));
                    }

                    for (String name : visibleAttributes) {
                        String value = record.getAttributeValue(name);
                        if (value != null) {
                            AttributeManager.getInstance().addAttribute(rl.getName(), name, value);
                        }
                    }
                    if (!loadedLocators.contains(rl)) {
                        if (locators == null) {
                            locators = new ArrayList<ResourceLocator>();
                        }

                        locators.add(rl);
                        loadedLocators.add(rl);
                    }

                }
                if (locators != null) {
                    hic.loadHostedTracks(locators);
                }
            }


        } catch (IOException exc) {
            log.error("Error opening Encode browser", exc);
        }

    }

    public void remove(ResourceLocator locator) {
        try {
            EncodeFileBrowser browser = EncodeFileBrowser.getInstance(genome);
            browser.remove(locator);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


}
