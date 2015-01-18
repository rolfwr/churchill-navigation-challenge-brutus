#include "../../challenge/point_search.h"
#include <vector>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <cmath>
#include <stack>
#include <queue>
#include <atomic>

#define NOMINMAX
#include "windows.h"
#include "immintrin.h"

#pragma intrinsic(memcpy)

#define NDUMP 1

// It's too early to nail down how many points per block we should have.
// Timing of some higher seems to on average be better, but also vary
// greatly between runs.

const int vectorsets_per_block = 1;
const int points_per_vectorset = 4;
const int workers = 3;

struct vectorset {
    float xs[points_per_vectorset];
    float ys[points_per_vectorset];
    uint32_t rankid[points_per_vectorset];
};

struct block {
    vectorset vectors[vectorsets_per_block];
    uint32_t children[4];
};

struct compressed_point {
    uint32_t rankid;
    float x;
    float y;
};

struct compressed_atomic_point {
    uint32_t rankid;
    float x;
    float y;
    std::atomic<bool> ready;
};


struct SearchContext {
    SearchContext() : remaining_buffer(1024) {
        memset(&found[0], 0, sizeof(compressed_atomic_point) * 100000);
    }

    uint32_t* enlarge(int& read_point) {
        int oldsize = (int)remaining_buffer.size();
        int endcount = oldsize - read_point;
        remaining_buffer.resize(oldsize * 2);
        auto begin = remaining_buffer.begin();
        std::copy(begin + read_point, begin + oldsize, begin + read_point + oldsize);
        read_point += oldsize;
        return remaining_buffer.data();
    }

    int get_mask() {
        return (int)remaining_buffer.size() - 1;
    }

    std::vector<Point> points;
    std::vector<block> blocks[workers];
    std::vector<uint32_t> remaining_buffer;
    block* aligned_begin[workers];
    block* aligned_end[workers];
    Rect rect;
    std::atomic<compressed_atomic_point*> filled;
    std::atomic<uint32_t> ranklimit;
    std::atomic<int> running;

    compressed_atomic_point found[100000];
};

enum quadrant : int {
    lxly = 0,
    lxhy = 1,
    hxly = 2,
    hxhy = 3
};

struct valueindex {
    float value;
    int index;
};


struct distance {
    int orderdist;
    float valuedist;
};


void add_distance_from_center(std::vector<valueindex>& values, std::vector<distance>& distances) {

    std::sort(values.begin(), values.end(), [](const valueindex& a, const valueindex& b) {
        return a.value < b.value;
    });

    int count = (int)values.size();
    int doublemid = count - 1;

    float valuescale = abs(1.0f / (values.front().value - values.back().value));
    float valuemid = (values.front().value + values.back().value) / 2;

    for (int i = 0; i < count; ++i) {
        int orderdist = abs(i * 2 - doublemid);
        float valuedist = abs(values[i].value - valuemid) * valuescale;
        int index = values[i].index;
        distances[index].orderdist += orderdist;
        distances[index].valuedist += valuedist;
    }
}


int find_centermost_candidate(const std::vector<Point>& candidates) {
    // Find centermost point in the list of candidate points.
    auto count = candidates.size();
    std::vector<valueindex> xsort(count);
    std::vector<valueindex> ysort(count);
    for (int i = 0; i < count; ++i) {
        xsort[i] = valueindex{ candidates[i].x, i };
        ysort[i] = valueindex{ candidates[i].y, i };
    }
    std::vector<distance> distances(count);
    add_distance_from_center(xsort, distances);
    add_distance_from_center(ysort, distances);

    auto best = std::min_element(distances.begin(), distances.end(),
        [](const distance& a, const distance& b) {
        return a.orderdist != b.orderdist ? a.orderdist < b.orderdist : a.valuedist < b.valuedist;
    });
    return (int)std::distance(distances.begin(), best);
}

uint32_t enblock(std::vector<block>& blocks, Point* begin, Point* end) {

    const int maxpoints = points_per_vectorset * vectorsets_per_block;

    auto count = std::min((int)std::distance(begin, end), maxpoints);
    if (count == 0) {
        return 0;
    }

    uint32_t result = (uint32_t)blocks.size();
    blocks.push_back(block{});
    block& b = blocks.back();

    std::vector<Point> candidates(begin, begin + count);
    begin += count;


    // Move most center point to end of block.
    int bestindex = find_centermost_candidate(candidates);
    if (bestindex != count - 1) {
        std::swap(candidates[bestindex], candidates[count - 1]);
    }

    // Make all point values initally inert.
    for (int vi = 0; vi < vectorsets_per_block; ++vi) {
        vectorset& vs = b.vectors[vi];

        for (int i = 0; i < points_per_vectorset; ++i) {
            vs.xs[i] = std::numeric_limits<float>::max();
        }
    }
 
    // Fill in the vector sets with available points.
    for (int i = 0; i < count; ++i) {
        Point& p = candidates[i];
        vectorset& vs = b.vectors[i/points_per_vectorset];
        vs.xs[i] = p.x;
        vs.ys[i] = p.y;
        vs.rankid[i] = (((uint32_t)p.rank) << 8) | (((uint32_t)p.id) & 0xFF);
    }

    // Partition remaining points and enblock them into children.
    if (count == maxpoints) {
        float sepx = candidates[count - 1].x;
        float sepy = candidates[count - 1].y;

        Point* xsplit = std::stable_partition(begin, end, [=](const Point& pt) {
            return pt.x < sepx;
        });

        Point* ysplit1 = std::stable_partition(begin, xsplit, [=](const Point& pt) {
            return pt.y < sepy;
        });

        Point* ysplit2 = std::stable_partition(xsplit, end, [=](const Point& pt) {
            return pt.y < sepy;
        });

        uint32_t lxly = enblock(blocks, begin, ysplit1);
        uint32_t lxhy = enblock(blocks, ysplit1, xsplit);
        uint32_t hxly = enblock(blocks, xsplit, ysplit2);
        uint32_t hxhy = enblock(blocks, ysplit2, end);
        
        block& parent = blocks[result];
        parent.children[quadrant::lxly] = lxly;
        parent.children[quadrant::lxhy] = lxhy;
        parent.children[quadrant::hxly] = hxly;
        parent.children[quadrant::hxhy] = hxhy;
    }

    return result;
}

extern "C" {


__declspec(dllexport) SearchContext* __stdcall create(const Point* points_begin, const Point* points_end) {

    assert((sizeof(block) % 64) == 0);
    auto sc = new SearchContext();

    auto count = std::distance(points_begin, points_end);
    sc->points.resize(count);
    std::copy(points_begin, points_end, sc->points.begin());

    auto shardcount = count / workers;

    Point* shard_begin = &sc->points.data()[0];
    for (int shard = 0; shard < workers; ++shard) {
        Point* shard_end = shard_begin + shardcount;
        if (shard + 1 == workers) {
            shard_end = &sc->points.data()[count];
        }


        std::sort(shard_begin, shard_end, [](const Point& a, const Point& b) {
            return a.rank < b.rank;
        });

        int bindex = enblock(sc->blocks[shard], shard_begin, shard_end);
        assert(bindex == 0);
        assert(sc->blocks[shard].size() >= (size_t)(shardcount / (points_per_vectorset*vectorsets_per_block)));

        size_t blockcount = sc->blocks[shard].size();
        sc->blocks[shard].push_back(block{}); // alignment padding;

        size_t begin = (size_t)(&sc->blocks[shard].data()[0]);
        size_t aligned = (begin + 63) & ~((size_t)63);

        std::memmove((void*)aligned, (void*)begin, blockcount * sizeof(block));
        sc->aligned_begin[shard] = (block*)aligned;
        sc->aligned_end[shard] = sc->aligned_begin[shard] + blockcount;

        shard_begin = shard_end;
    }


#ifdef NDEBUG
    sc->points.clear();
#else
    std::sort(sc->points.begin(), sc->points.end(), [](const Point& a, const Point& b) {
        return a.rank < b.rank;
    });
#endif

    return sc;
};

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

static __forceinline uint32_t& getrankid(char* p) {
    return *((uint32_t*)(p + offsetof(compressed_point, rankid)));
}

static __forceinline void copy_point(char* dest, char* src) {
    *((compressed_point*)dest) = *((compressed_point*)src);
}


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

static __forceinline void enqueue(SearchContext* sc, uint32_t*& queue, int& enqueue_index, int& dequeue_index, int& queuemask, uint32_t value) {
    queue[enqueue_index] = value;
    enqueue_index = (enqueue_index + 1) & queuemask;
    if (enqueue_index == dequeue_index) {
        // The initial queue should be sized so that it is unlikely this will ever be called.
        queue = sc->enlarge(dequeue_index);
        queuemask = sc->get_mask();
    }
}

struct work {
    SearchContext* sc;
    int id;
};


DWORD WINAPI scan_worker(LPVOID workptr) {
    work* w = (work*)workptr;
    SearchContext* sc = w->sc;

    __m128 lxs = _mm_load1_ps(&sc->rect.lx);
    __m128 hxs = _mm_load1_ps(&sc->rect.hx);
    __m128 lys = _mm_load1_ps(&sc->rect.ly);
    __m128 hys = _mm_load1_ps(&sc->rect.hy);

    int enqueue_index = 0;
    int dequeue_index = 0;
    uint32_t* queue = sc->remaining_buffer.data();
    int queuemask = sc->get_mask();

    const block* aligned_begin = sc->aligned_begin[w->id];
    const block* aligned_end = sc->aligned_end[w->id];

    if (aligned_begin == aligned_end) {
        goto done;
    }

    enqueue(sc, queue, enqueue_index, dequeue_index, queuemask, 0);

    while (enqueue_index != dequeue_index) {
        const block& b = aligned_begin[queue[dequeue_index]];
        dequeue_index = (dequeue_index + 1) & queuemask;
        if (enqueue_index != dequeue_index) {
            _mm_prefetch((const char*)(&aligned_begin[queue[dequeue_index]]), _MM_HINT_T0);
        }

        __m128i seen_better = _mm_setzero_si128();
        for (int vi = 0; vi < vectorsets_per_block; ++vi) {
            uint32_t ranklimit = sc->ranklimit.load() >> 8;
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

            for (int i = 0; i < points_per_vectorset; ++i) {
                if (inboundsi.m128i_i32[i]) {
                    compressed_atomic_point* writepoint = sc->filled.fetch_add(1);
                    writepoint->rankid = vs.rankid[i];
                    writepoint->x = vs.xs[i];
                    writepoint->y = vs.ys[i];
                    writepoint->ready.store(true);
                }
            }
        }

        if (!(_mm_test_all_zeros(seen_better, seen_better))) {
            float x = b.vectors[vectorsets_per_block - 1].xs[points_per_vectorset - 1];
            float y = b.vectors[vectorsets_per_block - 1].ys[points_per_vectorset - 1];
            bool islx = sc->rect.lx < x;
            bool ishx = sc->rect.hx >= x;
            bool isly = sc->rect.ly < y;
            bool ishy = sc->rect.hy >= y;

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

    done:
    sc->running.fetch_sub(1);
    return 0;
};



int32_t search_master(SearchContext* sc, const Rect rect, const int32_t count, Point* out_points) {
    sc->running.store(workers);
    sc->rect = rect;
    sc->filled.store(&sc->found[0]);
    sc->ranklimit = std::numeric_limits<uint32_t>::max();
    work ws[workers];
    compressed_atomic_point* pos = &sc->found[0];
    for (int shard = 0; shard < workers; ++shard){
        work* w = &ws[shard];
        w->id = shard;
        w->sc = sc;
        QueueUserWorkItem(&scan_worker, w, 0);
    }

    
    compressed_point* bestheap = (compressed_point*)alloca(count * sizeof(compressed_point));
    for (int i = 0; i < count; ++i) {
        bestheap[i].rankid = std::numeric_limits<uint32_t>::max();
    }
    int running;
    do {
        running = sc->running.load();
        while (pos != sc->filled.load()) {
            assert(pos < sc->filled.load());
            if (!pos->ready.load()) {
                continue;
            }

            pos->ready.store(false);

            if (pos->rankid < bestheap->rankid) {
                pop_heap_raw((char*)(void*)bestheap, count);
                push_heap(bestheap, count - 1, pos->rankid, pos->x, pos->y);
                sc->ranklimit.store(bestheap->rankid);
            }
            ++pos;
        }

    } while (running);

    assert(pos == sc->filled.load());

    int left = count;
    while (left != 0 && bestheap->rankid == std::numeric_limits<uint32_t>::max()) {
        pop_heap_raw((char*)(void*)bestheap, left);
        --left;
    }

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
}

__declspec(dllexport) int32_t __stdcall search(SearchContext* sc, const Rect rect, const int32_t count, Point* out_points) {
#ifndef NDEBUG 
    std::vector<Point> goodbuf(count);
    auto goodresult = search_good(sc, rect, count, &goodbuf.front());
#endif
    auto result = search_master(sc, rect, count, out_points);

#ifndef NDEBUG 
    assert(result == goodresult);
    assert(memcmp(goodbuf.data(), out_points, result * sizeof(Point)) == 0);
#endif
    return result;
};


__declspec(dllexport) SearchContext* __stdcall destroy(SearchContext* sc) {
    delete sc;
    return nullptr;
};

}
