/*
    Copyright (c) 2015 Rolf W. Rasmussen <rolfwr@gmail.com>

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.
*/

#include "../../challenge/point_search.h"
#include <vector>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include "immintrin.h"

#pragma intrinsic(memcpy)

#ifdef NDEBUG
#define search_fast search
#endif

/*
 * The basic strategy for this code is to construct a tree of blocks
 * containing point data. The rank values of the points in each each block are
 * smaller than the rank values of any decendant blocks. Each block has four
 * child blocks, representing the four quadrants of the coordinate space as
 * divided by a pivot point also stored in the parent block.
 */

/** The number of points that fit into a single SIMD register.
 *
 * "Single Instruction, Multiple Data" (SIMD) instructions are used to process
 * several points at a time, rather than doing them one by one.
 *
 * While the Intel i5-4250U CPU of target machine supports the AVX2 SIMD
 * instruction set which supports 512 bit wide registers, the newst development
 * machine I could get my hands on unfortunately only has AVX which only
 * supports 256 bit wide registers.
 * 
 * A 256 bit wide register can contain eight float values, e.g. eight x
 * coordinate values or eight y coordinate values.
 */
const int points_per_vectorset = 8;

/** The number of SIMD passes that will be performed on a single block.
 *
 * The size of each block is a tradeoff between quickly reducing the search
 * space by only tranversing down specific branches of the subtree, and
 * performing processing of blocks of memory that have good cache locality.
 */
const int vectorsets_per_block = 1;

/** Maximum number of blocks that needs to be processed during a search.
 *
 * There is an upper limit to the number of blocks that needs to be processed.
 * Blocks that are entirely contained within the search bounds will quickly
 * fulfil the number of points requested. Only blocks lying at the edge
 * of the search bounds have a chance of increasing the number of blocks to
 * be processed significantly. The worst case scenario is a long and thin
 * search rect that spans all the points in one dimension while straddeling the
 * edges pairs of bottommost blocks in the other dimension.
 *
 * Given perfect partitioning of the data the worst case would result in the
 * number of blocks reaching something like:
 *
 *     2^(ceil(ln(10^7/points_per_vectorset/vectorsets_per_block)/ln(4))+2)
 *
 * We double this to have a safe margin of error.
 * 
 */
const int max_block_process = 4096*2;

/** The number of points a search is expected to return.
 *
 * By specializing on returning a specific number of points allows us to
 * optimize further.
 */
const int points_requested = 20;

/**
 * To find pivot points that split the data set up pretty evenly, we want to 
 * find the median. As a naive way of preventing the calculation of the median
 * from making the loading too slow, we limit calculate an estimated median
 * based on a limited number of samples.
 *
 * This aspect of the loading hasn't been analysed properly yet. There exists
 * better median calculation algorithms that will give better estimates or
 * even exact result in less time. It is also unclear whether using medians
 * for pivot points is the best approach. Using medians assumes that the
 * distribution function used for point generation is similar to the
 * distribution function used for search bounds generation. Analysis of the
 * data sets has not been done yet to verify that this is the case.
 */
const int median_sample_count = 100000;


const uint8_t lxbit = 1;
const uint8_t lybit = 2;
const uint8_t nhxbit = 4;
const uint8_t nhybit = 8;
const uint8_t lxlybit = 16;
const uint8_t lxhybit = 32;
const uint8_t hxlybit = 64;
const uint8_t hxhybit = 128;

/** The data that will be processed by a single SIMD pass.
 */
struct vectorset {
    /// X coordinate values that fit in a single SIMD register.
    float xs[points_per_vectorset];

    /// Y coordinate values that fit in a single SIMD register.
    float ys[points_per_vectorset];

    /** Combined rank and id values for the point.
     *
     * The rank is stored in the upper 24 bits, while the id is stored in the
     * lower 8 bits.
     */
    uint32_t rankid[points_per_vectorset];
};

struct coordinate {
    float x;
    float y;
};

/** A block which makes up a node in a tree of blocks.
 */
struct block {
    /// One vectorset for each SIMD pass that will be performed on the block.
    vectorset vectors[vectorsets_per_block];

    uint32_t best_block_rank;
    uint32_t best_child_rank;

    /// Pivot coordinate value.
    coordinate pivot;

    /// Indicies to the blocks for each quadrant.
    uint32_t children[4];
};

/** Point representation stored in priority queue.
 *
 * To be able to return a specified number of best ranked points, the search
 * algorithm will maintain a collection of that number of points, which
 * represent the best ranked points found so far.
 *
 * The priority queue is implemented as an array based heap data structure
 * which makes it efficient to identify and replace the point with the highest
 * rank value. The compressed_point struct is the representation of the point
 * inside of this priority queue.
 */
struct compressed_point {
    uint32_t rankid;
    float x;
    float y;
};

/** Contains the preprocessed data and preallocated buffers.
 */
struct SearchContext {
#ifndef NDEBUG
    // In debug builds we keep all the points in their original for, so that
    // we can run a naive search algorithm to easily validate and compare the
    // output from the fast search algorithm.
    std::vector<Point> points;
#endif

    compressed_point heap[points_requested];

    /** Allocated memory for storing the blocks of the search tree.
     *
     * During creation of the tree, we will use the memory of the vector as
     * intended, but before finalizing the search context, we will shift the
     * blocks to be cache line aligned. The memory allocated by vector will
     * still be used after doing this, even tough element access through the
     * vector API no longer makes sense.
     */
    std::vector<block> blocks;

    /** Pointer to the first tree block after alignment shift.
     *
     * If the tree has no blocks, i.e. there are no points, this is a nullptr.
     */
    block* aligned_begin;

    /** Queue of unprocessed blocks.
     *
     * During search, applicable child buffer indicies will be added to one end
     * of the queue, while the search loop removes buffer indicies one by one
     * from the other end. To remove any overhead in maintaining the queue, the
     * read and write pointers into the queue will always monotonically
     * increase. I.e. there is no reuse what so ever of the memory in the
     * queue. Each memory location is only ever written once, and read once.
     */
    uint32_t process_queue_buffer[max_block_process];

    SearchContext() {
    }
};

/** This is the order the children block for the quadrants are stored.
 */
enum quadrant : int {
    lxly = 0,
    lxhy = 1,
    hxly = 2,
    hxhy = 3
};

coordinate find_pivot(Point* begin, int remaining) {
    int stride = 1;
    if (remaining > median_sample_count * 2) {
        stride = remaining / median_sample_count;
    }

    float sepx = 0;
    float sepy = 0;
    // Find a suitable pivit point by locating median coordinate values.
    std::vector<float> xrem;
    int samplesize = remaining / stride;
    xrem.reserve(samplesize + 1);
    for (int i = 0; i < remaining; i += stride) {
        xrem.push_back(begin[i].x);
    }

    std::sort(xrem.begin(), xrem.end());
    auto mid = xrem.size() / 2;
    sepx = xrem[mid];

    std::vector<float> yrem1;

    yrem1.reserve(samplesize / 2 + 1);
    std::vector<float> yrem2;
    yrem2.reserve(samplesize / 2 + 1);

    for (int i = 0; i < remaining; i += stride) {
        if (begin[i].x < sepx) {
            yrem1.push_back(begin[i].y);
        }
        else {
            yrem2.push_back(begin[i].y);
        }
    }

    std::sort(yrem1.begin(), yrem1.end());
    std::sort(yrem2.begin(), yrem2.end());

    if (yrem1.empty()) {
        sepy = yrem2[yrem2.size() / 2];
    }
    else if (yrem2.empty())
    {
        sepy = yrem1[yrem1.size() / 2];
    }
    else
    {
        sepy = (yrem1[yrem1.size() / 2] + yrem2[yrem2.size() / 2]) / 2;
    }

    return coordinate{ sepx, sepy };
}

struct enblock_result {
    uint32_t block_index;
    uint32_t bestrankid;
};

/** Create block tree for the given range of rank ordered points.
 *
 * This function is called during the creation phase, and has therefore not
 * been optimized for speed.
 *
 * This function will create a block containing the points with the lowest
 * rank value in the range, then partition the remaining points into four
 * quadrants divided by a pivot point selected to ensure even distribution of
 * points in each quadrant. The function is then invoked recursively to create
 * child blocks for each quadrant.
 *
 * @returns index of topmost block created.
 */
enblock_result enblock(std::vector<block>& blocks, Point* begin, Point* end)
{
    enblock_result result{ 0, 0xFFFFFFFF };

    const int maxpoints = points_per_vectorset * vectorsets_per_block;
    int available = (int)std::distance(begin, end);
    int count;
    if (available > maxpoints) {
            count = maxpoints;
    }
    else
    {
        count = available;
    }

    int remaining = available - count;
    if (count == 0) {
        return result;
    }

    result.bestrankid = begin->rank << 8;

    result.block_index = (uint32_t)blocks.size();
    blocks.push_back(block{});
    block& b = blocks.back();
    b.best_block_rank = result.bestrankid;

    std::vector<Point> candidates(begin, begin + count);
    begin += count;

    if (remaining) {
        b.pivot = find_pivot(begin, remaining);
    }

    // Make all point values initally inert, so that the search loop will
    // ignore them without the need for specific conditionals to exclude them
    // from the search. The search loop can therefore assume that all blocks
    // are fully filled.
    for (int vi = 0; vi < vectorsets_per_block; ++vi) {
        vectorset& vs = b.vectors[vi];

        for (int i = 0; i < points_per_vectorset; ++i) {
            vs.xs[i] = std::numeric_limits<float>::max();
        }
    }

    // Fill in the vector sets with available points.
    for (int i = 0; i < candidates.size(); ++i) {
        Point& p = candidates[i];
        vectorset& vs = b.vectors[i/points_per_vectorset];
        vs.xs[i % points_per_vectorset] = p.x;
        vs.ys[i % points_per_vectorset] = p.y;
        vs.rankid[i % points_per_vectorset] = (((uint32_t)p.rank) << 8) | (((uint32_t)p.id) & 0xFF);
    }

    // Partition remaining points and enblock them into children.
    if (remaining) {
        float sepx = b.pivot.x;
        float sepy = b.pivot.y;

        // We need to use stable_partition rather than reqular partition,
        // because we need to ensure that each partition is still ordered
        // by rank.
        Point* xsplit = std::stable_partition(begin, end, [=](const Point& pt) {
            return pt.x < sepx;
        });

        Point* ysplit1 = std::stable_partition(begin, xsplit, [=](const Point& pt) {
            return pt.y < sepy;
        });

        Point* ysplit2 = std::stable_partition(xsplit, end, [=](const Point& pt) {
            return pt.y < sepy;
        });

        enblock_result lxly = enblock(blocks, begin, ysplit1);
        enblock_result lxhy = enblock(blocks, ysplit1, xsplit);
        enblock_result hxly = enblock(blocks, xsplit, ysplit2);
        enblock_result hxhy = enblock(blocks, ysplit2, end);

        // The vector may have been resized, invalidating the previous
        // reference to the block. Therefore, look it up by index after
        // generating children.
        block& parent = blocks[result.block_index];
        parent.children[quadrant::lxly] = lxly.block_index;
        parent.children[quadrant::lxhy] = lxhy.block_index;
        parent.children[quadrant::hxly] = hxly.block_index;
        parent.children[quadrant::hxhy] = hxhy.block_index;

        uint8_t child_flags = (lxly.block_index ? lxlybit : 0)
            | (lxhy.block_index ? lxhybit : 0)
            | (hxly.block_index ? hxlybit : 0)
            | (hxhy.block_index ? hxhybit : 0);
        assert((child_flags & 0x0f) == 0);

        parent.best_child_rank = (std::min({ lxly.bestrankid, lxhy.bestrankid, hxly.bestrankid, hxhy.bestrankid }) & 0xffffff00) | child_flags;
    }
    else
    {
        b.best_child_rank = 0xffffff00;
    }

    assert((blocks[result.block_index].best_child_rank & 0x0f) == 0);

    return result;
}

extern "C" {

__declspec(dllexport) SearchContext* __stdcall create(const Point* points_begin, const Point* points_end) {
    // Make sure our blocks suited for being cache line aligned.
    assert((sizeof(block) % 64) == 0);
    auto sc = new SearchContext();

    auto count = std::distance(points_begin, points_end);
    std::vector<Point> points;
    points.resize(count);
    std::copy(points_begin, points_end, points.begin());
    std::sort(points.begin(), points.end(), [](const Point& a, const Point& b) {
        return a.rank < b.rank;
    });

    int bindex = enblock(sc->blocks, &points.data()[0], &points.data()[count]).block_index;
    assert(bindex == 0);
    assert(sc->blocks.size() >= (size_t)(count / (points_per_vectorset*vectorsets_per_block)));

    size_t blockcount = sc->blocks.size();
    sc->blocks.push_back(block{}); // alignment padding;

    size_t begin = (size_t)(&sc->blocks.data()[0]);
    size_t aligned = (begin + 63) & ~((size_t)63);

    std::memmove((void*)aligned, (void*)begin, blockcount * sizeof(block));
    sc->aligned_begin = blockcount ? (block*)aligned : nullptr;

#ifndef NDEBUG
    sc->points = points;
    std::sort(sc->points.begin(), sc->points.end(), [](const Point& a, const Point& b) {
        return a.rank < b.rank;
    });
#endif

    return sc;
};

}

/** Get the combined rank and id of a compressed_point at a memory location.
 *
 * While the C++ standard library already have an implementation of a priority
 * queue, we obtain faster execution by implementing a special purpose priority
 * queue. The elements in the priority queue are ordered by the combined rank
 * and id value. In reality, the we only care about the rank, but since the
 * rank is stored in the upper bits of the uint32_t we get the same result
 * as we would if we had just ordered by rank.
 *
 */
static __forceinline uint32_t& getrankid(char* p) {
    return *((uint32_t*)(p + offsetof(compressed_point, rankid)));
}

/** Copy a compressed_point from one memory location to another. */
static __forceinline void copy_point(char* dest, char* src) {
    *((compressed_point*)dest) = *((compressed_point*)src);
}

/** Remove point with the highest rank value from the priority queue.
 *
 * This removes the point at the start of the heap, and frees up a slot
 * for a new point at the end of the heap.
 */
static __forceinline void pop_heap_raw(char* heap, int last_index) {
    int end = last_index * sizeof(compressed_point);

    uint32_t value = getrankid(heap + end);
    // Find insert point.
    int i = 0;
    int c1;
    while (true) {
        c1 = i * 2 + sizeof(compressed_point) * 1;
        int c2 = i * 2 + sizeof(compressed_point) * 2;
        if (c1 >= end) {
            goto insert;
        }

        if (c2 >= end) {
            goto last;
        }

        int highc = (getrankid(heap + c2) > getrankid(heap + c1)) ? c2 : c1;

        if (value >= getrankid(heap + highc)) {
            goto insert;
        }

        copy_point(heap + i, heap + highc);

        i = highc;
    }

last:
    if (value < getrankid(heap + c1)) {
        copy_point(heap + i, heap + c1);
        i = c1;
    }

insert:
    copy_point(heap + i, heap + end);
}

/** Remove point with the highest rank value from the priority queue.
 *
 * This is a specialized version of pop_heap_raw() which assumes the
 * heap contains the number of elements specified by points_requested.
 * 
 * This removes the point at the start of the heap, and frees up a slot
 * for a new point at the end of the heap.
 */
static __forceinline void pop_heap_raw_specialized(char* heap) {
    const int end = (points_requested - 1) * sizeof(compressed_point);

    uint32_t value = getrankid(heap + end);
    // Find insert point.
    int i = 0;
    int c1;
    while (true) {
        c1 = i * 2 + sizeof(compressed_point) * 1;
        int c2 = i * 2 + sizeof(compressed_point) * 2;
        if (c1 >= end) {
            goto insert;
        }

        if (c2 >= end) {
            goto last;
        }

        int highc = (getrankid(heap + c2) > getrankid(heap + c1)) ? c2 : c1;

        if (value >= getrankid(heap + highc)) {
            goto insert;
        }

        copy_point(heap + i, heap + highc);

        i = highc;
    }

last:
    if (value < getrankid(heap + c1)) {
        copy_point(heap + i, heap + c1);
        i = c1;
    }

insert:
    copy_point(heap + i, heap + end);
}


/** Add point to priority queue which has a free slot at the end of the heap.
 */
static __forceinline void push_heap(compressed_point* heap, uint32_t newrankid, float x, float y) {
    int lastpos = points_requested - 1;
    assert(lastpos != 0);
    do {
        int parent = (lastpos - 1) / 2;
        if (heap[parent].rankid >= newrankid) {
            break;
        }

        heap[lastpos] = heap[parent];
        lastpos = parent;
    } while (lastpos != 0);

    
    heap[lastpos].rankid = newrankid;
    heap[lastpos].x = x;
    heap[lastpos].y = y;
}

struct process_queue {
    int dequeue_index;
    int enqueue_index;

    uint32_t* buffer;

    process_queue(uint32_t* buffer_)
        : dequeue_index(0), enqueue_index(0), buffer(buffer_)
    {
    }

    /** Add index of block to process later to the ring buffer. */
    __forceinline void enqueue(SearchContext* sc, uint32_t value) {
        buffer[enqueue_index++] = value;        
    }

    __forceinline bool contains_values() {
        return dequeue_index != enqueue_index;
    }

    __forceinline uint32_t front() {
        return buffer[dequeue_index];
    }

    __forceinline void pop() {
        ++dequeue_index;
    }
};

extern "C" {

/** Release build search API entry-point.
*/
__declspec(dllexport) int32_t __stdcall search_fast(SearchContext* sc, const Rect rect, const int32_t count_, Point* out_points)
{
    // Verify specialization assumptions.
    assert(vectorsets_per_block == 1);
    assert(points_per_vectorset == 8);
    assert(count_ == points_requested);

    // Prepare search bounds for SIMD point vector comparison.
    __m256 lxs = _mm256_set1_ps(rect.lx);
    __m256 hxs = _mm256_set1_ps(rect.hx);
    __m256 lys = _mm256_set1_ps(rect.ly);
    __m256 hys = _mm256_set1_ps(rect.hy);

    // Prepare search bounds for SIMD pivot comparison.
    __m128 rect_lxlyhxhy = _mm_set_ps(rect.hy, rect.hx, rect.ly, rect.lx);
    assert(rect_lxlyhxhy.m128_f32[0] == rect.lx);

    process_queue queue(&sc->process_queue_buffer[0]);

    const block* aligned_begin = sc->aligned_begin;
    if (aligned_begin == nullptr) {
        return 0;
    }

    compressed_point* bestheap = sc->heap;

    // Fill the priority queue initially with invalid points, which will be
    // replaced as the search commences, and will be filtered out from the
    // search result if they remain after the search loop completes.    
    for (compressed_point* ptr = bestheap; ptr < bestheap + points_requested; ++ptr) {
        ptr->rankid = std::numeric_limits<uint32_t>::max();
    }
 
    // Start on the root block. All other blocks to be searched will be found
    // from there.
    queue.enqueue(sc, 0);

    // Loop until no more blocks remain in queue.
    while (queue.contains_values()) {
        const block& b = aligned_begin[queue.front()];
        queue.pop();

        if (b.best_block_rank >= bestheap->rankid)
        {
            continue;
        }

        if (queue.contains_values()) {
            // Try to prefetch the memory region for the next block to be
            // proccessed into the CPU cache. We do this before starting to
            // process the current block, so that the fetch occur concurrently.
            const char* memloc = (const char*)(&aligned_begin[queue.front()]);
            _mm_prefetch(memloc, _MM_HINT_T0);
            _mm_prefetch(memloc + 64, _MM_HINT_T0);

            if ((queue.dequeue_index + 1) != queue.enqueue_index) {
                const char* memloc = (const char*)(&aligned_begin[queue.buffer[queue.dequeue_index + 1]]);
                _mm_prefetch(memloc, _MM_HINT_T0);
                _mm_prefetch(memloc + 64, _MM_HINT_T0);
            }
        }

        // This is the inner loop of SIMD instructions. This instructions
        // basically checks each point in a single vectorset for the
        // following boolean properties:
        //
        //   1. Has rank value lower than the highest rank value in the
        //      priority queue.
        //   2. Has a coordinate which is inside the search bounds.

        const vectorset& vs = b.vectors[0];
        __m256 xs = _mm256_load_ps(&vs.xs[0]);
        __m256 ys = _mm256_load_ps(&vs.ys[0]);

        __m256 inbound = _mm256_and_ps(
            _mm256_and_ps(_mm256_cmp_ps(lxs, xs, _CMP_LE_OQ), _mm256_cmp_ps(xs, hxs, _CMP_LE_OQ)),
            _mm256_and_ps(_mm256_cmp_ps(lys, ys, _CMP_LE_OQ), _mm256_cmp_ps(ys, hys, _CMP_LE_OQ)));

        int inboundi = _mm256_movemask_ps(inbound);
        int pushmask = inboundi;

        unsigned long index;
        while (_BitScanForward(&index, pushmask)) {
            // On machines with BMI1 instruction set support, this could be
            // replaced with _blsr_u32().
            pushmask &= pushmask - 1;

            if (bestheap->rankid > vs.rankid[index]) {
                assert(b.best_block_rank < bestheap->rankid);

                pop_heap_raw_specialized((char*)(void*)bestheap);
                push_heap(bestheap, vs.rankid[index], vs.xs[index], vs.ys[index]);
            }
            else
            {
                goto nextinqueue;
            }

        }

        // If we don't see any rank values that are lower than the rank
        // highest value we've already found, then we know that there is no
        // reason to examine any of the child blocks, since they will only
        // contain even higher rank values.
        //
        // Queue up child blocks only if there is a possiblity for finding
        // points with lower rank values in them.
        if (b.best_child_rank < bestheap->rankid) {
            __m128 pivot_xyxy = { 0 };
            pivot_xyxy = _mm_loadl_pi(pivot_xyxy, (const __m64*)&b.pivot.x);
            pivot_xyxy = _mm_loadh_pi(pivot_xyxy, (const __m64*)&b.pivot.x);

            assert(pivot_xyxy.m128_f32[0] == b.pivot.x);
            assert(pivot_xyxy.m128_f32[1] == b.pivot.y);
            assert(pivot_xyxy.m128_f32[2] == b.pivot.x);
            assert(pivot_xyxy.m128_f32[3] == b.pivot.y);

            __m128 lessthan = _mm_cmplt_ps(rect_lxlyhxhy, pivot_xyxy);
            uint8_t mask = ((uint8_t)_mm_movemask_ps(lessthan)) | ((uint8_t)b.best_child_rank);

            if ((mask & (lxbit | lybit | lxlybit)) == (lxbit | lybit | lxlybit)) {
                assert(b.children[lxly]);
                queue.enqueue(sc, b.children[lxly]);
            }

            if ((mask & (lxbit | nhybit | lxhybit)) == (lxbit |lxhybit)) {
                assert(b.children[lxhy]);
                queue.enqueue(sc, b.children[lxhy]);
            }

            if ((mask & (nhxbit | lybit | hxlybit)) == (lybit | hxlybit)) {
                assert(b.children[hxly]);
                queue.enqueue(sc, b.children[hxly]);
            }

            if ((mask & (nhxbit | nhybit | hxhybit)) == hxhybit) {
                assert(b.children[hxhy]);
                queue.enqueue(sc, b.children[hxhy]);
            }
        }

    nextinqueue:
        ;
    }

    assert(!queue.contains_values());

    int left = points_requested;

    // Remove any invalid points which has been retained from when the priority
    // queue was initialized. If any still exists, that means that the search
    // found less points than the number that was requested.
    while (left != 0 && bestheap->rankid == std::numeric_limits<uint32_t>::max()) {
        pop_heap_raw((char*)(void*)bestheap, --left);
    }

    // Write the remaining points from the priority queue out to the result
    // buffer. The priority queue outputs the points in the reverse order of what
    // is expected.
    uint32_t result = left;
    while (left > 0) {
        --left;
        out_points[left].id = (int8_t)(bestheap->rankid);
        out_points[left].rank = bestheap->rankid >> 8;
        out_points[left].x = bestheap->x;
        out_points[left].y = bestheap->y;
        pop_heap_raw((char*)(void*)bestheap, left);
    }

    return result;
};

/** Free memory used by search context.
*/
__declspec(dllexport) SearchContext* __stdcall destroy(SearchContext* sc) {
    delete sc;
    return nullptr;
};

// Debugging support that is not compiled in release builds follows.

#ifndef NDEBUG 
/** Naive search algorithm used for comparison during debugging. */
__declspec(dllexport) int32_t __stdcall search_good(SearchContext* sc, const Rect rect, const int32_t count, Point* out_points) {
    auto end = sc->points.cend();
    auto pos = sc->points.cbegin();
    auto outptr = out_points;
    auto outend = outptr + count;
    while (pos != end) {
        if (pos->x >= rect.lx && pos->x <= rect.hx && pos->y >= rect.ly && pos->y <= rect.hy) {
            *outptr++ = *pos;
            if (outptr == outend) {
                break;
            }
        }
        ++pos;
    }
    auto result = (int32_t)std::distance(out_points, outptr);

    return result;
};

/** Debug build search API entry-point.
 */
__declspec(dllexport) int32_t __stdcall search(SearchContext* sc, const Rect rect, const int32_t count, Point* out_points) {
    std::vector<Point> goodbuf(count);
    auto goodresult = search_good(sc, rect, count, &goodbuf.front());
    auto result = search_fast(sc, rect, count, out_points);

    assert(result == goodresult);

    for (int i = 0; i < result; ++i) {
        Point good = goodbuf[i];

        Point point = out_points[i];

        assert(good.id == point.id);
        assert(good.rank == point.rank);
        assert(good.x == point.x);
        assert(good.y == point.y);
    }

    assert(memcmp(goodbuf.data(), out_points, result * sizeof(Point)) == 0);

    return result;
};
#endif

}
