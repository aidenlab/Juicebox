/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2016 Broad Institute, Aiden Lab
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

import juicebox.gui.SuperAdapter;
import org.broad.igv.util.HttpUtils;

import java.awt.datatransfer.DataFlavor;
import java.awt.datatransfer.Transferable;
import java.awt.dnd.*;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Listener for drag&drop actions
 */
public class FileDropTargetListener implements DropTargetListener {

    private final SuperAdapter superAdapter;

    public FileDropTargetListener(SuperAdapter superAdapter) {
        this.superAdapter = superAdapter;
    }

    public void dragEnter(DropTargetDragEvent event) {

        if (!isDragAcceptable(event)) {
            event.rejectDrag();
        }
    }

    public void dragExit(DropTargetEvent event) {
    }

    public void dragOver(DropTargetDragEvent event) {
        // you can provide visual feedback here
    }

    public void dropActionChanged(DropTargetDragEvent event) {
        if (!isDragAcceptable(event)) {
            event.rejectDrag();
        }
    }

    public void drop(DropTargetDropEvent event) {
        if (!isDropAcceptable(event)) {
            event.rejectDrop();
            return;
        }

        event.acceptDrop(DnDConstants.ACTION_COPY);

        Transferable transferable = event.getTransferable();

        try {
            @SuppressWarnings("unchecked") // Transferable when called with DataFlavor javaFileList is guaranteed to return a File List.
                    java.util.List<File> files = (java.util.List<File>) transferable.getTransferData(DataFlavor.javaFileListFlavor);
            List<String> paths = new ArrayList<String>();
            for (File f : files) {
                paths.add(f.getAbsolutePath());
            }
            superAdapter.safeLoad(paths, false, files.get(0).getName());

        } catch (Exception e) {
            String obj;
            try {
                obj = transferable.getTransferData(DataFlavor.stringFlavor).toString();
                if (HttpUtils.isRemoteURL(obj)) {
                    superAdapter.safeLoad(Arrays.asList(obj), false, obj);
                }
            } catch (Exception e1) {
                e1.printStackTrace();
            }

        }
        superAdapter.repaint();
        event.dropComplete(true);
    }

    private boolean isDragAcceptable(DropTargetDragEvent event) {
        //  Check the  available data flavors here
        //  Currently accepting all flavors
        return (event.getDropAction() & DnDConstants.ACTION_COPY_OR_MOVE) != 0;
    }

    private boolean isDropAcceptable(DropTargetDropEvent event) {
        //  Check the  available data flavors here
        //  Currently accepting all flavors
        return (event.getDropAction() & DnDConstants.ACTION_COPY_OR_MOVE) != 0;
    }
}