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

/** Default buffer size for [ArrayChunker] subclasses. */
private const val DEFAULT_BUFFER_SIZE = 65536

/**
 * A chunked iterator for primitive array types.
 *
 * The maximum chunk size is currently a constant and is defined by
 * [DEFAULT_BUFFER_SIZE].
 *
 * Why? Java does not provide an API for coercing a primitive buffer
 * to a [ByteBuffer] without copying, because a primitive buffer might
 * have a non-native byte ordering. This class implements
 * constant-memory iteration over a primitive array without resorting
 * to primitive buffers.
 *
 * Invariant: buffers produced by the iterator must be consumed
 * **in order**, because their content is invalidated between
 * [Iterator.next] calls.
 *
 * @since 0.3.1
 */
internal abstract class ArrayChunker<T>(
        /** The array. */
        protected val data: T,
        /** Number of elements in the array. */
        private val size: Int,
        /** Byte order for the produced buffers. */
        private val order: ByteOrder) : Sequence<ByteBuffer> {

    /** Byte size of the element of [T]. */
    abstract val bytes: Int

    /**
     * Populates this buffer using elements from [data].
     *
     * @see ByteBuffer.put
     */
    abstract fun ByteBuffer.put(data: T, offset: Int, size: Int)

    override fun iterator() = object : Iterator<ByteBuffer> {
        private var offset = 0  // into the [data].
        private var step = DEFAULT_BUFFER_SIZE / bytes
        // Only allocated 'cache' if the [data] is bigger than [step].
        private val cache by lazy {
            // [DEFAULT_BUFFER_SIZE] is rounded down to be divisible by [bytes].
            ByteBuffer.allocateDirect(step * bytes).order(order)
        }

        override fun hasNext() = offset < size

        override fun next(): ByteBuffer {
            val available = Math.min(size - offset, step)
            val result = if (available == step) {
                cache.apply { rewind() }
            } else {
                ByteBuffer.allocateDirect(available * bytes).order(order)
            }

            with(result) {
                put(data, offset, available)
                rewind()
            }

            offset += available
            return result
        }
    }
}

internal class BooleanArrayChunker(data: BooleanArray) :
        ArrayChunker<BooleanArray>(data, data.size, ByteOrder.nativeOrder()) {
    override val bytes: Int get() = 1

    override fun ByteBuffer.put(data: BooleanArray, offset: Int, size: Int) {
        for (i in offset until offset + size) {
            put(if (data[i]) 1.toByte() else 0.toByte())
        }
    }
}

internal class ShortArrayChunker(data: ShortArray, order: ByteOrder) :
        ArrayChunker<ShortArray>(data, data.size, order) {
    override val bytes: Int get() = java.lang.Short.BYTES

    override fun ByteBuffer.put(data: ShortArray, offset: Int, size: Int) {
        asShortBuffer().put(data, offset, size)
    }
}

internal class IntArrayChunker(data: IntArray, order: ByteOrder) :
        ArrayChunker<IntArray>(data, data.size, order) {
    override val bytes: Int get() = java.lang.Integer.BYTES

    override fun ByteBuffer.put(data: IntArray, offset: Int, size: Int) {
        asIntBuffer().put(data, offset, size)
    }
}

internal class LongArrayChunker(data: LongArray, order: ByteOrder) :
        ArrayChunker<LongArray>(data, data.size, order) {
    override val bytes: Int get() = java.lang.Long.BYTES

    override fun ByteBuffer.put(data: LongArray, offset: Int, size: Int) {
        asLongBuffer().put(data, offset, size)
    }
}

internal class FloatArrayChunker(data: FloatArray, order: ByteOrder) :
        ArrayChunker<FloatArray>(data, data.size, order) {
    override val bytes: Int get() = java.lang.Float.BYTES

    override fun ByteBuffer.put(data: FloatArray, offset: Int, size: Int) {
        asFloatBuffer().put(data, offset, size)
    }
}

internal class DoubleArrayChunker(data: DoubleArray, order: ByteOrder) :
        ArrayChunker<DoubleArray>(data, data.size, order) {
    override val bytes: Int get() = java.lang.Double.BYTES

    override fun ByteBuffer.put(data: DoubleArray, offset: Int, size: Int) {
        asDoubleBuffer().put(data, offset, size)
    }
}

internal class StringArrayChunker(data: Array<String>) :
        ArrayChunker<Array<String>>(data, data.size, ByteOrder.nativeOrder()) {
    override val bytes: Int by lazy { data.asSequence().map { it.length }.max() ?: 0 }

    override fun ByteBuffer.put(data: Array<String>, offset: Int, size: Int) {
        for (i in offset until offset + size) {
            put(data[i].toByteArray(Charsets.US_ASCII).copyOf(bytes))
        }
    }
}

/**
 * A chunked initializer for primitive array types.
 *
 * JVM does not allow mapping files larger that `Int.MAX_SIZE` bytes.
 * As a result, one cannot simply read a primitive array from a memory
 * mapped file via the usual [ByteBuffer] magic.
 *
 * This class allows to incrementally initialize an array from multiple
 * buffers.
 *
 * @since 0.3.2
 */
internal abstract class ArrayMerger<out T>(protected val data: T) : (ByteBuffer) -> Unit {
    protected var offset = 0

    fun result() = data
}

internal class BooleanArrayMerger(size: Int) :
        ArrayMerger<BooleanArray>(BooleanArray(size)) {
    override fun invoke(chunk: ByteBuffer) {
        while (chunk.hasRemaining()) {
            data[offset++] = chunk.get() == 1.toByte()
        }
    }
}

internal class ByteArrayMerger(size: Int) : ArrayMerger<ByteArray>(ByteArray(size)) {
    override fun invoke(chunk: ByteBuffer) = with(chunk) {
        while (hasRemaining()) {
            val size = remaining()
            get(data, offset, size)
            offset += size
        }
    }
}

/** Adjusts this buffer position after executing [block]. */
private inline fun ByteBuffer.linked(bytes: Int, block: (ByteBuffer) -> Unit) {
    val tick = position()
    block(this)
    val consumedCeiling = capacity() - tick
    position(position() + (consumedCeiling - consumedCeiling % bytes))
}

internal class ShortArrayMerger(size: Int) : ArrayMerger<ShortArray>(ShortArray(size)) {
    override fun invoke(chunk: ByteBuffer) = chunk.linked(java.lang.Short.BYTES) {
        with(it.asShortBuffer()) {
            while (hasRemaining()) {
                val size = remaining()
                get(data, offset, size)
                offset += size
            }
        }
    }
}

internal class IntArrayMerger(size: Int) : ArrayMerger<IntArray>(IntArray(size)) {
    override fun invoke(chunk: ByteBuffer) = chunk.linked(java.lang.Integer.BYTES) {
        with(it.asIntBuffer()) {
            while (hasRemaining()) {
                val size = remaining()
                get(data, offset, size)
                offset += size
            }
        }
    }
}

internal class LongArrayMerger(size: Int) : ArrayMerger<LongArray>(LongArray(size)) {
    override fun invoke(chunk: ByteBuffer) = chunk.linked(java.lang.Long.BYTES) {
        with(it.asLongBuffer()) {
            while (hasRemaining()) {
                val size = remaining()
                get(data, offset, size)
                offset += size
            }
        }
    }
}

internal class FloatArrayMerger(size: Int) : ArrayMerger<FloatArray>(FloatArray(size)) {
    override fun invoke(chunk: ByteBuffer) = chunk.linked(java.lang.Float.BYTES) {
        with(it.asFloatBuffer()) {
            while (hasRemaining()) {
                val size = remaining()
                get(data, offset, size)
                offset += size
            }
        }
    }
}

internal class DoubleArrayMerger(size: Int) : ArrayMerger<DoubleArray>(DoubleArray(size)) {
    override fun invoke(chunk: ByteBuffer) = chunk.linked(java.lang.Double.BYTES) {
        with(it.asDoubleBuffer()) {
            while (hasRemaining()) {
                val size = remaining()
                get(data, offset, size)
                offset += size
            }
        }
    }
}

internal class StringArrayMerger(size: Int, private val bytes: Int) :
        ArrayMerger<Array<String>>(Array(size) { "" }) {

    override fun invoke(chunk: ByteBuffer) = with(chunk) {
        // Iterate until there is not more data or the next value is
        // split between chunks, e.g.
        //             chunk2
        //          .........
        //     "foo|bar\0\0\0"
        //      ...
        //   chunk1
        while (remaining() >= bytes) {
            val b = ByteArray(bytes)
            get(b)
            data[offset++] = String(b, Charsets.US_ASCII).trimEnd('\u0000')
        }
    }
}