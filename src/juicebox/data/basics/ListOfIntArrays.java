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

package juicebox.data.basics;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * can't use <T> because we need to instantiate the array, otherwise that would have been nice
 */
public class ListOfIntArrays {
	
	private final int DEFAULT_LENGTH = Integer.MAX_VALUE - 10;
	private final long overallLength;
	private final List<int[]> internalList = new ArrayList<>();
	
	public ListOfIntArrays(long length) {
		this.overallLength = length;
		long tempLength = length;
		while (tempLength > 0) {
			if (tempLength < DEFAULT_LENGTH) {
				internalList.add(new int[(int) tempLength]);
				break;
			} else {
				internalList.add(new int[DEFAULT_LENGTH]);
				tempLength -= DEFAULT_LENGTH;
			}
		}
	}
	
	public ListOfIntArrays(long totSize, int defaultValue) {
		this(totSize);
		for (int[] array : internalList) {
			Arrays.fill(array, defaultValue);
		}
	}
	
	public void clear() {
		internalList.clear();
	}
	
	public int get(long index) {
		if (index < overallLength) {
			int pseudoRow = (int) (index / DEFAULT_LENGTH);
			int pseudoCol = (int) (index % DEFAULT_LENGTH);
			return internalList.get(pseudoRow)[pseudoCol];
		} else {
			System.err.println("long index exceeds max size of list of int arrays while getting");
			return -Integer.MAX_VALUE;
		}
	}
	
	public void set(long index, int value) {
		long tempIndex = index;
		if (index < overallLength) {
			int pseudoRow = (int) (index / DEFAULT_LENGTH);
			int pseudoCol = (int) (index % DEFAULT_LENGTH);
			internalList.get(pseudoRow)[pseudoCol] = value;
		} else {
			System.err.println("long index exceeds max size of list of arrays while setting");
			return;
		}
		//System.err.println("unusual - long index exceeds max size of list of arrays while setting");
	}
	
	public long getLength() {
		return overallLength;
	}
	
	public ListOfIntArrays deepClone() {
		ListOfIntArrays clone = new ListOfIntArrays(overallLength);
		for (int k = 0; k < internalList.size(); k++) {
			System.arraycopy(internalList.get(k), 0, clone.internalList.get(k), 0, internalList.get(k).length);
		}
		return clone;
	}
	
	public void divideBy(long index, int value) {
		if (index < overallLength) {
			int pseudoRow = (int) (index / DEFAULT_LENGTH);
			int pseudoCol = (int) (index % DEFAULT_LENGTH);
			internalList.get(pseudoRow)[pseudoCol] /= value;
		} else {
			System.err.println("long index exceeds max size of list of arrays while dividing");
			return;
		}
		System.err.println("unusual - long index exceeds max size of list of arrays while dividing");
	}
	
	public void addValuesFrom(ListOfIntArrays other) {
		if (overallLength == other.overallLength) {
			for (int i = 0; i < internalList.size(); i++) {
				for (int j = 0; j < internalList.get(i).length; j++) {
					internalList.get(i)[j] += other.internalList.get(i)[j];
				}
			}
		} else {
			System.err.println("Adding objects of different sizes!");
		}
	}
	
	public void addTo(long index, int value) {
		if (index < overallLength) {
			int pseudoRow = (int) (index / DEFAULT_LENGTH);
			int pseudoCol = (int) (index % DEFAULT_LENGTH);
			internalList.get(pseudoRow)[pseudoCol] += value;
		} else {
			System.err.println("long index exceeds max size of list of arrays while adding");
		}
	}
	
	public List<int[]> getValues() {
		return internalList;
	}
}
