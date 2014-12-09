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

package juicebox.windowui;

import juicebox.MainWindow;
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

    private final MainWindow mainWindow;

    public FileDropTargetListener(MainWindow mainWindow) {
        this.mainWindow = mainWindow;
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
            @SuppressWarnings("unchecked") // Transferable when called with DataFlavor javaFileList is guaranteed to retunr a File List.
                    java.util.List<File> files = (java.util.List<File>) transferable.getTransferData(DataFlavor.javaFileListFlavor);
            List<String> paths = new ArrayList<String>();
            for (File f : files) {
                paths.add(f.getAbsolutePath());
            }
            mainWindow.load(paths, false);

        } catch (Exception e) {
            String obj;
            try {
                obj = transferable.getTransferData(DataFlavor.stringFlavor).toString();
                if (HttpUtils.isRemoteURL(obj)) {
                    mainWindow.load(Arrays.asList(obj), false);
                }
            } catch (Exception e1) {
                e1.printStackTrace();
            }

        }
        mainWindow.repaint();
        event.dropComplete(true);
    }

    public boolean isDragAcceptable(DropTargetDragEvent event) {
        //  Check the  available data flavors here
        //  Currently accepting all flavors
        return (event.getDropAction() & DnDConstants.ACTION_COPY_OR_MOVE) != 0;
    }

    public boolean isDropAcceptable(DropTargetDropEvent event) {
        //  Check the  available data flavors here
        //  Currently accepting all flavors
        return (event.getDropAction() & DnDConstants.ACTION_COPY_OR_MOVE) != 0;
    }
}