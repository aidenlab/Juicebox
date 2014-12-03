package juicebox.encode;

import java.io.File;
import java.util.Collection;
import java.util.Map;

/**
 * @author jrobinso
 *         Date: 10/31/13
 *         Time: 10:11 PM
 */
public class EncodeFileRecord {

    private final String path;
    private final Map<String, String> attributes;
    private boolean selected = false;
    private String trackName;

    public EncodeFileRecord(String path, Map<String, String> attributes) {
        this.path = path;
        this.attributes = attributes;
    }

    public String getPath() {
        return path;
    }

    public String getFileType() {
        //String off trailing gz, if present
        String filetype = path;
        if (filetype.endsWith(".gz")) {
            filetype = filetype.substring(0, filetype.length() - 3);
        }
        int idx = filetype.lastIndexOf(".");
        return filetype.substring(idx + 1);
    }

    public String getAttributeValue(String name) {
        String value = attributes.get(name);
        if (name.equals("type") && value == null) value = getFileType();
        return value;
    }

    public Collection<String> getAttributeNames() {
        return attributes.keySet();
    }

    public boolean containsText(String filter) {
        for (String value : attributes.values()) {
            if (value.contains(filter)) return true;
        }
        return false;
    }

    boolean isSelected() {
        return selected;
    }

    void setSelected(boolean selected) {
        this.selected = selected;
    }

    /**
     * Return a friendly name for the track.  Unfortunately it is neccessary to hardcode certain attributes.
     *
     * @return
     */
    public String getTrackName() {

        if (trackName == null) {
            StringBuilder sb = new StringBuilder();
            if (attributes.containsKey("cell")) sb.append(attributes.get("cell")).append(" ");
            if (attributes.containsKey("antibody")) sb.append(attributes.get("antibody")).append(" ");
            if (attributes.containsKey("dataType")) sb.append(attributes.get("dataType")).append(" ");
            if (attributes.containsKey("view")) sb.append(attributes.get("view")).append(" ");
            if (attributes.containsKey("replicate")) sb.append("rep ").append(attributes.get("replicate"));

            trackName = sb.toString().trim();
            if (sb.length() == 0) trackName = (new File(path)).getName();
        }

        return trackName;

    }

    /**
     * Test if record has a eough of meta-data to be interpretable
     *
     * @return
     */
    public boolean hasMetaData() {

        return (attributes.containsKey("cell")) || (attributes.containsKey("antibody"));

    }
}
