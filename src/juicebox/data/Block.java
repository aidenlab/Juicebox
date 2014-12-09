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

package juicebox.data;

//import java.awt.*;
//import java.util.List;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;


/**
 * @author jrobinso
 * @date Aug 10, 2010
 */
public class Block {

    private final int number;

    private final List<ContactRecord> records;

    public Block(int number) {
        this.number = number;
        records = new ArrayList<ContactRecord>();
    }

    public Block(int number, List<ContactRecord> records) {
        this.number = number;
        this.records = records;
    }

    public int getNumber() {
        return number;
    }


    public Collection<ContactRecord> getContactRecords() {
        return records;
    }


}
