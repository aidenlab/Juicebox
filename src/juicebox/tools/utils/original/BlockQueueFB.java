/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2021 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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
 *  FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.tools.utils.original;

import htsjdk.tribble.util.LittleEndianInputStream;

import java.awt.*;
import java.io.*;
import java.util.HashMap;
import java.util.Map;

class BlockQueueFB implements BlockQueue {

    final File file;
    BlockPP block;
    long filePosition;
    long fileLength;

    BlockQueueFB(File file) {
        this.file = file;
        fileLength = file.length();
        try {
            advance();
        } catch (IOException e) {
            e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
        }
    }

    public void advance() throws IOException {
        if (filePosition >= fileLength) {
            block = null;
            return;
        }

        FileInputStream fis = null;

        try {
            fis = new FileInputStream(file);
            fis.getChannel().position(filePosition);

            LittleEndianInputStream lis = new LittleEndianInputStream(fis);
            int blockNumber = lis.readInt();
            int nRecords = lis.readInt();

            byte[] bytes = new byte[nRecords * 12];
            readFully(bytes, fis);

            ByteArrayInputStream bis = new ByteArrayInputStream(bytes);
            lis = new LittleEndianInputStream(bis);


            Map<Point, ContactCount> contactRecordMap = new HashMap<>(nRecords);
            for (int i = 0; i < nRecords; i++) {
                int x = lis.readInt();
                int y = lis.readInt();
                float v = lis.readFloat();
                ContactCount rec = new ContactCount(v);
                contactRecordMap.put(new Point(x, y), rec);
            }
            block = new BlockPP(blockNumber, contactRecordMap);

            // Update file position based on # of bytes read, for next block
            filePosition = fis.getChannel().position();

        } finally {
            if (fis != null) fis.close();
        }
    }

    public BlockPP getBlock() {
        return block;
    }

    /**
     * Read enough bytes to fill the input buffer
     */
    void readFully(byte[] b, InputStream is) throws IOException {
        int len = b.length;
        if (len < 1) throw new IndexOutOfBoundsException();
        int n = 0;
        while (n < len) {
            int count = is.read(b, n, len - n);
            if (count < 0)
                throw new EOFException();
            n += count;
        }
    }
}
