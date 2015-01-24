#include "../../challenge/point_search.h"
#include <vector>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include "immintrin.h"

#pragma intrinsic(memcpy)
/*
 * The basic strategy for this code is to construct a tree of blocks
 * containing point data. The rank values of the points in each each block are
 * smaller than the rank values of any of the decendant blocks. Each block has
 * four child blocks, representing the four quadrants of the coordinate space
 * as divided by a pivot point also stored in the parent block.
 */

/** The number of points that fit into a single SIMD register.
 *
 * "Single Instruction, Multiple Data" (SIMD) instructions are used to process
 * several points at a time, rather than doing them one by one.
 *
 * While the Intel i5-4250U CPU of target machine supports the AVX2 SIMD
 * instruction set which supports 512 bit wide registers, the machine I'm
 * developing on unfortunately only has SSE4.2 which only supports 128 bit wide
 * registers.
 * 
 * An 128 bit wide register can contain four float values, e.g. four x
 * coordinate values or four y coordinate values.
 */
const int points_per_vectorset = 4;

/** The number of SIMD passes that will be performed on a single block.
 *
 * The size of each block is a tradeoff between quickly reducing the search
 * space by only tranversing down specific branches of the subtree, and
 * performing processing of blocks of memory that have good cache locality.
 */
const int vectorsets_per_block = 2;

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

/** A block which makes up a node in a tree of blocks.
 */
struct block {
    /// One vectorset for each SIMD pass that will be performed on the block.
    vectorset vectors[vectorsets_per_block];

    /// Indicies to the blocks for each quadrant.
    uint32_t children[4];

    /// X coordinate value pivot.
    float sepx;

    /// Y coordinate value pivot.
    float sepy;

    /** Padding to bring the size of the block up to a multiple of 64.
     *
     * This allows blocks that are store consequatively to all be aligned to
     * cache line width.
     */
    char pad[8];
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

    /** Allocated memory for containing a ring buffer of unprocessed blocks.
     *
     * During search, applicable child buffer indicies will be added to one
     * end of the ring buffer, while the search loop removes buffer indicies
     * one by one from the other end.
     *
     * This buffer is always a power of two, to allow ring buffer wrapping to
     * be implemented using bit masking.
     */
    std::vector<uint32_t> remaining_buffer;

    SearchContext() : remaining_buffer(1024) {
        // The ring buffer typically never needs more than 1024 elements.
    }

    /** Add more space to the ring buffer.
     *
     * If for some reason the original allocation of the ring buffer is not
     * large enough, increase its size.
     */
    uint32_t* enlarge(int& read_point) {
        int oldsize = (int)remaining_buffer.size();
        int endcount = oldsize - read_point;
        remaining_buffer.resize(oldsize * 2);
        auto begin = remaining_buffer.begin();
        std::copy(begin + read_point, begin + oldsize, begin + read_point + oldsize);
        read_point += oldsize;
        return remaining_buffer.data();
    }

    /** Get bit mask that will wrap the current ring buffer size.
     */
    int get_mask() {
        return (int)remaining_buffer.size() - 1;
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
uint32_t enblock(SearchContext& sc, Point* begin, Point* end)
{
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
        return 0;
    }

    uint32_t result = (uint32_t)sc.blocks.size();
    sc.blocks.push_back(block{});
    block& b = sc.blocks.back();

    std::vector<Point> candidates(begin, begin + count);
    begin += count;

    float sepx = 0;
    float sepy = 0;

    if (remaining) {
        // Find a suitable pivit point by locating median coordinate values.
        std::vector<float> xrem;
        xrem.reserve(remaining);
        for (int i = 0; i < remaining; ++i) {
            xrem.push_back(begin[i].x);
        }

        std::sort(xrem.begin(), xrem.end());
        int mid = remaining / 2;
        sepx = xrem[mid];

        std::vector<float> yrem1;
        yrem1.reserve(remaining / 2 + 1);
        std::vector<float> yrem2;
        yrem2.reserve(remaining / 2 + 1);

        for (int i = 0; i < remaining; ++i) {
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

        b.sepx = sepx;
        b.sepy = sepy;
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

        uint32_t lxly = enblock(sc, begin, ysplit1);
        uint32_t lxhy = enblock(sc, ysplit1, xsplit);
        uint32_t hxly = enblock(sc, xsplit, ysplit2);
        uint32_t hxhy = enblock(sc, ysplit2, end);

        block& parent = sc.blocks[result];
        parent.children[quadrant::lxly] = lxly;
        parent.children[quadrant::lxhy] = lxhy;
        parent.children[quadrant::hxly] = hxly;
        parent.children[quadrant::hxhy] = hxhy;
    }

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

    int bindex = enblock(*sc, &points.data()[0], &points.data()[count]);
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

#ifndef NDEBUG
int32_t __stdcall search_good(SearchContext* sc, const Rect rect, const int32_t count, Point* out_points) {
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
#endif

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
static __forceinline void pop_heap_raw(char* heap, int count) {
    int end = (count - 1) * sizeof(compressed_point);

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
static __forceinline void push_heap(compressed_point* heap, int lastpos, uint32_t newrankid, float x, float y) {
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

/** Add index of block to process later to the ring buffer. */
static __forceinline void enqueue(SearchContext* sc, uint32_t*& queue, int& enqueue_index, int& dequeue_index, int& queuemask, uint32_t value) {
    queue[enqueue_index] = value;
    enqueue_index = (enqueue_index + 1) & queuemask;
    if (enqueue_index == dequeue_index) {
        // The initial queue should be sized so that it is unlikely this will ever be called.
        queue = sc->enlarge(dequeue_index);
        queuemask = sc->get_mask();
    }
}

/** The actual search algorithm. */
int32_t __stdcall search_alt(SearchContext* sc, const Rect rect, const int32_t count, Point* out_points)
{
    // Prepare search bounds for SIMD comparison.
    __m128 lxs = _mm_load1_ps(&rect.lx);
    __m128 hxs = _mm_load1_ps(&rect.hx);
    __m128 lys = _mm_load1_ps(&rect.ly);
    __m128 hys = _mm_load1_ps(&rect.hy);

    // Read and write indiceies into the ring buffer used to queue up blocks
    // for processing.
    int dequeue_index = 0;
    int enqueue_index = 0;

    uint32_t* queue = sc->remaining_buffer.data();
    int queuemask = sc->get_mask();

    const block* aligned_begin = sc->aligned_begin;
    if (aligned_begin == nullptr) {
        return 0;
    }

    compressed_point* bestheap = (compressed_point*)alloca(count * sizeof(compressed_point));

    // Fill the priority queue initially with invalid points, which will be
    // replaced as the search commences, and will be filtered out from the
    // search result if they remain after the search loop completes.
    for (int i = 0; i < count; ++i) {
        bestheap[i].rankid = std::numeric_limits<uint32_t>::max();
    }
 
    // Start on the root block. All other blocks to be searched will be found
    // from there.
    enqueue(sc, queue, enqueue_index, dequeue_index, queuemask, 0);

    // Loop until no more blocks remain in queue.
    while (enqueue_index != dequeue_index) {
        const block& b = aligned_begin[queue[dequeue_index]];
        dequeue_index = (dequeue_index + 1) & queuemask;

        if (enqueue_index != dequeue_index) {
            // Try to prefetch the memory region for the next block to be
            // proccessed into the CPU cache. We do this before starting to
            // process the current block, so that the fetch occur concurrently.
            _mm_prefetch((const char*)(&aligned_begin[queue[dequeue_index]]), _MM_HINT_T0);
            _mm_prefetch(((const char*)(&aligned_begin[queue[dequeue_index]]) + 64), _MM_HINT_T0);
        }

        // If we don't see any rank values that are lower than the rank
        // highest value we've already found, then we know that there is no
        // reason to examine any of the child blocks, since they will only
        // contain even higher rank values.
        __m128i seen_better = _mm_setzero_si128();

        for (int vi = 0; vi < vectorsets_per_block; ++vi) {

            // This is the inner loop of SIMD instructions. This instructions
            // basically checks each point in a single vectorset for the
            // following boolean properties:
            //
            //   1. Has rank value lower than the highest rank value in the
            //      priority queue.
            //   2. Has a coordinate which is inside the search bounds.

            uint32_t ranklimit = bestheap->rankid >> 8;
            __m128i ranklimitv = _mm_set_epi32(ranklimit, ranklimit, ranklimit, ranklimit);
            const vectorset& vs = b.vectors[vi];
            __m128i ranks = _mm_srli_epi32(_mm_load_si128((const __m128i*)&vs.rankid[0]), 8);
            __m128i betterv = _mm_cmpgt_epi32(ranklimitv, ranks);
            seen_better = _mm_or_si128(seen_better, betterv);
            __m128 xs = _mm_load_ps(&vs.xs[0]);
            __m128 ys = _mm_load_ps(&vs.ys[0]);

            __m128i inboundsi = _mm_castps_si128(
                _mm_and_ps(
                    _mm_and_ps(_mm_cmple_ps(lxs, xs), _mm_cmple_ps(xs, hxs)),
                    _mm_and_ps(_mm_cmple_ps(lys, ys), _mm_cmple_ps(ys, hys))));

            __m128i dopush = _mm_and_si128(inboundsi, betterv);

            // Add better matches to priority queue.
            for (int i = 0; i < points_per_vectorset; ++i) {
                if ((dopush.m128i_i32[i] != 0) && bestheap->rankid > vs.rankid[i]) {
                    pop_heap_raw((char*)(void*)bestheap, count);
                    push_heap(bestheap, count - 1, vs.rankid[i], vs.xs[i], vs.ys[i]);
                }
            }
        }

        // Queue up child blocks only if there is a possiblity for finding
        // points with lower rank values in them.
        if (!(_mm_test_all_zeros(seen_better, seen_better))) {
            bool islx = rect.lx < b.sepx;
            bool ishx = rect.hx >= b.sepx;
            bool isly = rect.ly < b.sepy;
            bool ishy = rect.hy >= b.sepy;

            // Only add the blocks whose quadrant intersect with the search
            // bounds.

            if (islx && isly && b.children[lxly]) {
                enqueue(sc, queue, enqueue_index, dequeue_index, queuemask, b.children[lxly]);
            }

            if (islx && ishy && b.children[lxhy]) {
                enqueue(sc, queue, enqueue_index, dequeue_index, queuemask, b.children[lxhy]);
            }

            if (ishx && isly && b.children[hxly]) {
                enqueue(sc, queue, enqueue_index, dequeue_index, queuemask, b.children[hxly]);
            }

            if (ishx && ishy && b.children[hxhy]) {
                enqueue(sc, queue, enqueue_index, dequeue_index, queuemask, b.children[hxhy]);
            }
        }
    }

    assert(enqueue_index == dequeue_index);

    int left = count;

    // Remove any invalid points which has been retained since the priority
    // queue was initialized. If any still exists, that means that the search
    // found less points than the number that was requested.
    while (left != 0 && bestheap->rankid == std::numeric_limits<uint32_t>::max()) {
        pop_heap_raw((char*)(void*)bestheap, left);
        --left;
    }

    // Write the remaining points from the priority queue out to the result
    // buffer. The priority queue outputs the points in the reverse order of what
    // is expected.
    uint32_t result = left;
    for (int i = ((int)result) - 1; i >= 0; --i) {
        out_points[i].id = (int8_t)(bestheap->rankid);
        out_points[i].rank = bestheap->rankid >> 8;
        out_points[i].x = bestheap->x;
        out_points[i].y = bestheap->y;
        pop_heap_raw((char*)(void*)bestheap, left);
        --left;
    }

    return result;
};

/** Search API entry-point.
 */
__declspec(dllexport) int32_t __stdcall search(SearchContext* sc, const Rect rect, const int32_t count, Point* out_points) {
#ifndef NDEBUG 
    std::vector<Point> goodbuf(count);
    auto goodresult = search_good(sc, rect, count, &goodbuf.front());
#endif
    auto result = search_alt(sc, rect, count, out_points);

#ifndef NDEBUG 
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
#endif

    return result;
};


/** Free memory used by search context.
 */
__declspec(dllexport) SearchContext* __stdcall destroy(SearchContext* sc) {
    delete sc;
    return nullptr;
};

}
