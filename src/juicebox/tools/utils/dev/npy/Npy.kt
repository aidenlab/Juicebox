/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2019 Broad Institute, Aiden Lab
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

package org.jetbrains.bio.npy

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.nio.channels.FileChannel.MapMode
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardOpenOption
import java.util.*

/**
 * A file in NPY format.
 *
 * Currently unsupported types:
 *
 *   * unsigned integral types (treated as signed)
 *   * bit field,
 *   * complex,
 *   * object,
 *   * Unicode
 *   * void*
 *   * intersections aka types for structured arrays.
 *
 * See http://docs.scipy.org/doc/numpy-dev/neps/npy-format.html
 */
object NpyFile {
    /**
     * NPY file header.
     *
     * Presently NumPy implements two version of the NPY format: 1.0 and 2.0.
     * The difference between the two is the maximum size of the NPY header.
     * Version 1.0 requires it to be <=2**16 while version 2.0 allows <=2**32.
     *
     * The appropriate NPY format is chosen automatically based on the
     * header size.
     */
    internal data class Header(val order: ByteOrder? = null,
                               val type: Char, val bytes: Int,
                               val shape: IntArray) {
        /** Major version number. */
        val major: Int
        /** Minor version number. */
        private val minor: Int = 0
        /** Meta-data formatted as a Python dict and 16-byte-padded. */
        private val meta: String

        /** Header size in bytes. */
        private val size: Int

        init {
            val metaUnpadded = StringJoiner(", ", "{", "}")
                    .add("'descr': '${order.toChar()}$type$bytes'")
                    .add("'fortran_order': False")
                    .add("'shape': (${shape.joinToString(",")}, )")
                    .toString()

            // According to the spec the total meta size should be
            // evenly divisible by 16 for alignment purposes. +1 here
            // accounts for the newline.
            // XXX despite the fact that the HEADER_LEN is 4 bytes in
            //     NPY2.0 the padding is always computed assuming 2 bytes.
            val totalUnpadded = MAGIC.size + 2 + java.lang.Short.BYTES +
                    metaUnpadded.length + 1
            val padding = 16 - totalUnpadded % 16

            var total = totalUnpadded + padding
            if (total <= NPY_10_20_SIZE_BOUNDARY) {
                major = 1
            } else {
                total += 2  // fix for the XXX above.
                major = 2
            }

            meta = metaUnpadded + " ".repeat(padding) + '\n'
            size = total
        }

        /** Allocates a [ByteBuffer] for this header. */
        internal fun allocate() = ByteBuffer.allocateDirect(size).apply {
            order(ByteOrder.LITTLE_ENDIAN)
            put(MAGIC)
            put(major.toByte())
            put(minor.toByte())

            when (major to minor) {
                1 to 0 -> putShort(meta.length.toShort())
                2 to 0 -> putInt(meta.length)
            }

            put(meta.toByteArray(Charsets.US_ASCII))

            rewind()
        }

        override fun equals(other: Any?) = when {
            this === other -> true
            other == null || other !is Header -> false
            else -> {
                order == other.order &&
                        type == other.type && bytes == other.bytes &&
                        Arrays.equals(shape, other.shape)
            }
        }

        override fun hashCode() = Objects.hash(order, type, bytes, Arrays.hashCode(shape))

        companion object {
            /** Each NPY file *must* start with this byte sequence. */
            internal val MAGIC = byteArrayOf(0x93.toByte()) + "NUMPY".toByteArray()
            /** Maximum byte size of the header to be written as NPY1.0. */
            // XXX this is a var only for testing purposes.
            internal var NPY_10_20_SIZE_BOUNDARY = 65535

            @Suppress("unchecked_cast")
            fun read(input: ByteBuffer) = with(input.order(ByteOrder.LITTLE_ENDIAN)) {
                val buf = ByteArray(6)
                get(buf)
                check(Arrays.equals(MAGIC, buf)) { "bad magic: ${String(buf)}" }

                val major = get().toInt()
                val minor = get().toInt()
                val size = when (major to minor) {
                    1 to 0 -> short.toInt()
                    2 to 0 -> int
                    else -> error("unsupported version: $major.$minor")
                }

                val header = ByteArray(size)
                get(header)

                val s = String(header)
                val meta = parseDict(s)
                val type = meta["descr"] as String
                check(!(meta["fortran_order"] as Boolean)) {
                    "Fortran-contiguous arrays are not supported"
                }

                val shape = (meta["shape"] as List<Int>).toIntArray()
                val order = type[0].toByteOrder()
                Header(order = order, type = type[1],
                        bytes = type.substring(2).toInt(), shape = shape)
            }
        }
    }

    /**
     * Reads an array in NPY format from a given path.
     *
     * The caller is responsible for coercing the resulting array to
     * an appropriate type via [NpyArray] methods.
     */
    @JvmStatic
    fun read(path: Path, step: Int = Int.MAX_VALUE): NpyArray {
        return FileChannel.open(path).use {
            var remaining = Files.size(path)
            var chunk = ByteBuffer.allocate(0)
            read(generateSequence {
                // Make sure we don't miss any unaligned bytes.
                remaining += chunk.remaining()
                if (remaining == 0L) {
                    null
                } else {
                    val offset = Files.size(path) - remaining
                    chunk = it.map(
                            MapMode.READ_ONLY, offset,
                            if (remaining > step) step.toLong() else remaining)

                    remaining -= chunk.capacity()
                    chunk
                }
            })
        }
    }

    internal fun read(chunks: Sequence<ByteBuffer>): NpyArray {
        // XXX we have to make it peeking, because otherwise
        //     the first chunk would be gone.
        val it = PeekingIterator(chunks.iterator())
        val header = Header.read(it.peek())
        val size = header.shape.reduce { a, b -> a * b }
        val merger = when (header.type) {
            'b' -> {
                check(header.bytes == 1)
                BooleanArrayMerger(size)
            }
            'u', 'i' -> when (header.bytes) {
                1 -> ByteArrayMerger(size)
                2 -> ShortArrayMerger(size)
                4 -> IntArrayMerger(size)
                8 -> LongArrayMerger(size)
                else -> error("invalid number of bytes for ${header.type}: ${header.bytes}")
            }
            'f' -> when (header.bytes) {
                4 -> FloatArrayMerger(size)
                8 -> DoubleArrayMerger(size)
                else -> error("invalid number of bytes for ${header.type}: ${header.bytes}")
            }
            'S' -> StringArrayMerger(size, header.bytes)
            else -> error("unsupported type: ${header.type}")
        }

        for (chunk in it) {
            chunk.order(header.order)
            merger(chunk)
        }

        return NpyArray(merger.result(), header.shape)
    }

    /**
     * Writes an array in NPY format to a given path.
     */
    @JvmOverloads
    @JvmStatic
    fun write(path: Path, data: BooleanArray,
              shape: IntArray = intArrayOf(data.size)) {
        write(path, allocate(data, shape))
    }

    @JvmOverloads
    @JvmStatic
    fun write(path: Path, data: ByteArray,
              shape: IntArray = intArrayOf(data.size)) {
        write(path, allocate(data, shape))
    }

    @JvmOverloads
    @JvmStatic
    fun write(path: Path, data: ShortArray,
              shape: IntArray = intArrayOf(data.size),
              order: ByteOrder = ByteOrder.nativeOrder()) {
        write(path, allocate(data, shape, order))
    }

    @JvmOverloads
    @JvmStatic
    fun write(path: Path, data: IntArray,
              shape: IntArray = intArrayOf(data.size),
              order: ByteOrder = ByteOrder.nativeOrder()) {
        write(path, allocate(data, shape, order))
    }

    @JvmOverloads
    @JvmStatic
    fun write(path: Path, data: LongArray,
              shape: IntArray = intArrayOf(data.size),
              order: ByteOrder = ByteOrder.nativeOrder()) {
        write(path, allocate(data, shape, order))
    }

    @JvmOverloads
    @JvmStatic
    fun write(path: Path, data: FloatArray,
              shape: IntArray = intArrayOf(data.size),
              order: ByteOrder = ByteOrder.nativeOrder()) {
        write(path, allocate(data, shape, order))
    }

    @JvmOverloads
    @JvmStatic
    fun write(path: Path, data: DoubleArray,
              shape: IntArray = intArrayOf(data.size),
              order: ByteOrder = ByteOrder.nativeOrder()) {
        write(path, allocate(data, shape, order))
    }

    @JvmOverloads
    @JvmStatic
    fun write(path: Path, data: Array<String>,
              shape: IntArray = intArrayOf(data.size)) {
        write(path, allocate(data, shape))
    }

    private fun write(path: Path, chunks: Sequence<ByteBuffer>) {
        FileChannel.open(path,
                StandardOpenOption.WRITE,
                StandardOpenOption.CREATE).use {

            for (chunk in chunks) {
                while (chunk.hasRemaining()) {
                    it.write(chunk)
                }
            }

            it.truncate(it.position())
        }
    }

    internal fun allocate(data: BooleanArray, shape: IntArray): Sequence<ByteBuffer> {
        val header = Header(order = null, type = 'b', bytes = 1, shape = shape)
        return sequenceOf(header.allocate()) + BooleanArrayChunker(data)
    }

    internal fun allocate(data: ByteArray, shape: IntArray): Sequence<ByteBuffer> {
        val header = Header(order = null, type = 'i', bytes = 1, shape = shape)
        return sequenceOf(header.allocate()) + ByteBuffer.wrap(data)
    }

    internal fun allocate(data: ShortArray, shape: IntArray,
                          order: ByteOrder): Sequence<ByteBuffer> {
        val header = Header(order = order, type = 'i',
                bytes = java.lang.Short.BYTES, shape = shape)
        return sequenceOf(header.allocate()) + ShortArrayChunker(data, order)
    }

    internal fun allocate(data: IntArray, shape: IntArray,
                          order: ByteOrder): Sequence<ByteBuffer> {
        val header = Header(order = order, type = 'i',
                bytes = java.lang.Integer.BYTES, shape = shape)
        return sequenceOf(header.allocate()) + IntArrayChunker(data, order)
    }

    internal fun allocate(data: LongArray, shape: IntArray,
                          order: ByteOrder): Sequence<ByteBuffer> {
        val header = Header(order = order, type = 'i',
                bytes = java.lang.Long.BYTES, shape = shape)
        return sequenceOf(header.allocate()) + LongArrayChunker(data, order)
    }

    internal fun allocate(data: FloatArray, shape: IntArray,
                          order: ByteOrder): Sequence<ByteBuffer> {
        val header = Header(order = order, type = 'f',
                bytes = java.lang.Float.BYTES, shape = shape)
        return sequenceOf(header.allocate()) + FloatArrayChunker(data, order)
    }

    internal fun allocate(data: DoubleArray, shape: IntArray,
                          order: ByteOrder): Sequence<ByteBuffer> {
        val header = Header(order = order, type = 'f',
                bytes = java.lang.Double.BYTES, shape = shape)
        return sequenceOf(header.allocate()) + DoubleArrayChunker(data, order)
    }

    internal fun allocate(data: Array<String>, shape: IntArray): Sequence<ByteBuffer> {
        val bytes = data.asSequence().map { it.length }.max() ?: 0
        val header = Header(order = null, type = 'S', bytes = bytes, shape = shape)
        return sequenceOf(header.allocate()) + StringArrayChunker(data)
    }
}

/** A wrapper for NPY array data. */
class NpyArray(
        /** Array data. */
        val data: Any,
        /** Array dimensions. */
        val shape: IntArray) {

    fun asBooleanArray() = data as BooleanArray

    fun asByteArray() = data as ByteArray

    fun asShortArray() = data as ShortArray

    fun asIntArray() = data as IntArray

    fun asLongArray() = data as LongArray

    fun asFloatArray() = data as FloatArray

    fun asDoubleArray() = data as DoubleArray

    @Suppress("unchecked_cast")
    fun asStringArray() = data as Array<String>

    override fun toString() = StringJoiner(", ", "NpyArray{", "}")
            .add("data=" + Arrays.deepToString(arrayOf(data))
                    .removeSurrounding("[", "]"))
            .add("shape=" + Arrays.toString(shape))
            .toString()
}

private fun Char.toByteOrder() = when (this) {
    '<' -> ByteOrder.LITTLE_ENDIAN
    '>' -> ByteOrder.BIG_ENDIAN
    '|' -> null
    else -> error(this)
}

private fun ByteOrder?.toChar() = when (this) {
    ByteOrder.LITTLE_ENDIAN -> '<'
    ByteOrder.BIG_ENDIAN -> '>'
    null -> '|'
    else -> error(this)
}