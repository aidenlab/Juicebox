package juicebox.data;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * @author jrobinso
 *         Date: 12/22/12
 *         Time: 1:06 PM
 */
public class DatasetReaderFactory {

    public static DatasetReader getReader(List<String> fileList) throws IOException {

        if (fileList.size() == 1) {
            String file = fileList.get(0);
            return getReaderForFile(file);
        } else {
            List<DatasetReaderV2> readers = new ArrayList<DatasetReaderV2>(fileList.size());
            for (String f : fileList) {
                DatasetReaderV2 r = getReaderForFile(f);
                if (r != null) {
                    readers.add(r);
                }

            }
            return new CombinedDatasetReader(readers);
        }
    }

    private static DatasetReaderV2 getReaderForFile(String file) throws IOException {
        String magicString = DatasetReaderV2.getMagicString(file);

        DatasetReaderV2 reader;
        if (magicString.equals("HIC")) {
            reader = new DatasetReaderV2(file);
        } else {
            System.err.println("This version is deprecated and is no longer supported.");
            //reader = new DatasetReaderV1(file);
            // file not actually read, usually canceled the read of password-protected file
            //if (reader.getVersion() == -1)
            return null;
        }
        return reader;
    }

}
