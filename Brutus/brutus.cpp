#include "../../challenge/point_search.h"
#include <vector>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <cmath>
#include <stack>
#include <queue>
#include "immintrin.h"

#define NDUMP 1

// It's too early to nail down how many points per block we should have.
// Timing of some higher seems to on average be better, but also vary
// greatly between runs.

const int vectorsets_per_block = 1;
const int points_per_vectorset = 4;

struct compressed_point {
    float x;
    float y;
    uint32_t rankid;
};

struct vectorset {
    float xs[points_per_vectorset];
    float ys[points_per_vectorset];
    uint32_t rankid[points_per_vectorset];
};

struct block {
    vectorset vectors[vectorsets_per_block];
    uint32_t children[4];
};

struct SearchContext {
    SearchContext() : remaining_buffer(1024) {
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
    std::vector<block> blocks;
    std::vector<uint32_t> remaining_buffer;
    block* aligned_begin;
    block* aligned_end;

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

uint32_t enblock(SearchContext& sc, Point* begin, Point* end) {

    const int maxpoints = points_per_vectorset * vectorsets_per_block;

    auto count = std::min((int)std::distance(begin, end), maxpoints);
    if (count == 0) {
        return 0;
    }

    uint32_t result = (uint32_t)sc.blocks.size();
    sc.blocks.push_back(block{});
    block& b = sc.blocks.back();

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

    assert((sizeof(block) % 64) == 0);
    auto sc = new SearchContext();

    auto count = std::distance(points_begin, points_end);
    sc->points.resize(count);
    std::copy(points_begin, points_end, sc->points.begin());
    std::sort(sc->points.begin(), sc->points.end(), [](const Point& a, const Point& b) {
        return a.rank < b.rank;
    });

    int bindex = enblock(*sc, &sc->points.data()[0], &sc->points.data()[count]);
    assert(bindex == 0);
    assert(sc->blocks.size() >= (size_t)(count / (points_per_vectorset*vectorsets_per_block)));

    size_t blockcount = sc->blocks.size();
    sc->blocks.push_back(block{}); // alignment padding;

    size_t begin = (size_t)(&sc->blocks.data()[0]);
    size_t aligned = (begin + 63) & ~((size_t)63);

    std::memmove((void*)aligned, (void*)begin, blockcount * sizeof(block));
    sc->aligned_begin = (block*) aligned;
    sc->aligned_end = sc->aligned_begin + blockcount;

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
    return *((uint32_t*)(p + offsetof(compressed_point,rankid)));
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

        memcpy(heap + i, heap + highc, sizeof(compressed_point));
        i = highc;
    }

last:
    if (value < getrankid(heap + c1)) {
        memcpy(heap + i, heap + c1, sizeof(compressed_point));
        i = c1;
    }

insert:
    memcpy(heap + i, heap + end, sizeof(compressed_point));
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

    
    heap[lastpos].x = x;
    heap[lastpos].y = y;
    heap[lastpos].rankid = newrankid;
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

int32_t __stdcall search_alt(SearchContext* sc, const Rect rect, const int32_t count, Point* out_points)
{
    __m128 lxs = _mm_load1_ps(&rect.lx);
    __m128 hxs = _mm_load1_ps(&rect.hx);
    __m128 lys = _mm_load1_ps(&rect.ly);
    __m128 hys = _mm_load1_ps(&rect.hy);


    int enqueue_index = 0;
    int dequeue_index = 0;
    uint32_t* queue = sc->remaining_buffer.data();
    int queuemask = sc->get_mask();

    const block* aligned_begin = sc->aligned_begin;
    const block* aligned_end = sc->aligned_end;

    if (aligned_begin == aligned_end) {
        return 0;
    }

    compressed_point* bestheap = (compressed_point*)alloca(count * sizeof(compressed_point));

    for (int i = 0; i < count; ++i) {
        bestheap[i].rankid = std::numeric_limits<int32_t>::max();
    }
 
    enqueue(sc, queue, enqueue_index, dequeue_index, queuemask, 0);

    while (enqueue_index != dequeue_index) {
        const block& b = aligned_begin[queue[dequeue_index]];
        dequeue_index = (dequeue_index + 1) & queuemask;

        int seen_better = 0;

        for (int vi = 0; vi < vectorsets_per_block; ++vi) {
            __m128i ranklimit = _mm_set_epi32(bestheap->rankid, bestheap->rankid, bestheap->rankid, bestheap->rankid);
            const vectorset& vs = b.vectors[vi];
            //for (int i = 0; i < points_per_vectorset; ++i) {
            //    better[i] = bestheap->rankid > vs.rankid[i];

            __m128i rankids = _mm_load_si128((const __m128i*)&vs.rankid[0]);
            __m128i better = _mm_cmpgt_epi32(ranklimit, rankids);

            //    float x = vs.xs[i];
            __m128 xs = _mm_load_ps(&vs.xs[0]);

            // float y = vs.ys[i];
            __m128 ys = _mm_load_ps(&vs.ys[0]);

            //    lxc[i] = x >= rect.lx;
            //    lxc[i] = rect.lx <= x;
            __m128 lxc = _mm_cmple_ps(lxs, xs);

            //    hxc[i] = x <= rect.hx;
            __m128 hxc = _mm_cmple_ps(xs, hxs);

            //    lyc[i] = y >= rect.ly;
            //    lyc[i] = rect.ly <= y;
            __m128 lyc = _mm_cmple_ps(lys, ys);

            //    hyc[i] = y <= rect.hy;
            __m128 hyc = _mm_cmple_ps(ys, hys);

            //dopush[i] = better[i] && lxc[i] && hxc[i] && lyc[i] && hyc[i];
            __m128 inbounds = _mm_and_ps(_mm_and_ps(lxc, hxc), _mm_and_ps(lyc, hyc));
            __m128i inboundsi = _mm_castps_si128(inbounds);
            __m128i dopush = _mm_and_si128(inboundsi, better);


            //}



            for (int i = 0; i < points_per_vectorset; ++i) {
                seen_better |= better.m128i_i32[i];
                if (dopush.m128i_i32[i] && vs.rankid[i] < bestheap->rankid) {
                    pop_heap_raw((char*)(void*)bestheap, count);
                    push_heap(bestheap, count - 1, vs.rankid[i], vs.xs[i], vs.ys[i]);
                }
            }
        }

        if (seen_better) {
            float x = b.vectors[vectorsets_per_block - 1].xs[points_per_vectorset - 1];
            float y = b.vectors[vectorsets_per_block - 1].ys[points_per_vectorset - 1];
            bool islx = rect.lx < x;
            bool ishx = rect.hx >= x;
            bool isly = rect.ly < y;
            bool ishy = rect.hy >= y;

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
    while (left != 0 && bestheap->rankid == std::numeric_limits<int32_t>::max()) {
        pop_heap_raw((char*)(void*)bestheap, left);
        --left;
    }

    uint32_t result = left;
    for (int i = ((int)result) - 1; i >= 0; --i) {
        out_points[i].id = bestheap->rankid;
        out_points[i].rank = bestheap->rankid >> 8;
        out_points[i].x = bestheap->x;
        out_points[i].y = bestheap->y;
        pop_heap_raw((char*)(void*)bestheap, left);
        --left;
    }

    return result;
};


__declspec(dllexport) int32_t __stdcall search(SearchContext* sc, const Rect rect, const int32_t count, Point* out_points) {
#ifndef NDEBUG 
    std::vector<Point> goodbuf(count);
    auto goodresult = search_good(sc, rect, count, &goodbuf.front());
#endif
    auto result = search_alt(sc, rect, count, out_points);

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
