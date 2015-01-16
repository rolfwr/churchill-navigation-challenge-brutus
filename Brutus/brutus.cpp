#include "../../challenge/point_search.h"
#include <vector>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <cmath>
#include <stack>
#include <queue>
//#include "immintrin.h"

#define NDUMP 1

// It's too early to nail down how many points per block we should have.
// Timing of some higher seems to on average be better, but also vary
// greatly between runs.
const int points_per_block = 4;

struct block {
    float xs[points_per_block];
    float ys[points_per_block];
    uint32_t rankid[points_per_block];
    uint32_t children[4];
};

struct SearchContext {
    std::vector<Point> points;
    std::vector<block> blocks;
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
    auto count = std::min((int)std::distance(begin, end), points_per_block);
    if (count == 0) {
        return 0;
    }

    uint32_t result = (uint32_t)sc.blocks.size();
    sc.blocks.push_back(block{});
    block& b = sc.blocks.back();

    std::vector<Point> candidates(begin, begin + count);
    begin += count;

    for (int i = 0; i < points_per_block; ++i) {
        b.xs[i] = std::numeric_limits<float>::max();
    }

    // Move most center point to end of block.
    int bestindex = find_centermost_candidate(candidates);
    if (bestindex != count - 1) {
        std::swap(candidates[bestindex], candidates[count - 1]);
    }

    for (int i = 0; i < count; ++i) {
        Point& p = candidates[i];

        b.xs[i] = p.x;
        b.ys[i] = p.y;
        b.rankid[i] = (((uint32_t)p.rank) << 8) | (((uint32_t)p.id) & 0xFF);
    }

    if (count == points_per_block) {
        float sepx = b.xs[points_per_block - 1];
        float sepy = b.ys[points_per_block - 1];

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
    assert(sc->blocks.size() >= (size_t)(count / points_per_block));

    sc->aligned_begin = (block*)_aligned_malloc(sc->blocks.size() * sizeof(block), 64);
    assert(count == 0 || ((((uint32_t)(void*)sc->aligned_begin % 64) == 0)));
    sc->aligned_end = sc->aligned_begin + sc->blocks.size();
    std::memcpy(sc->aligned_begin, &sc->blocks.data()[0], sc->blocks.size() * sizeof(block));
    sc->blocks.clear();

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

static __forceinline int32_t& getrank(char* p) {
    return *((int32_t*)(p + offsetof(Point,rank)));
}

static __forceinline void pop_heap_raw(char* heap, int count) {
    int end = (count - 1) * sizeof(Point);

    int32_t value = getrank(heap + end);
    // Find insert point.
    int i = 0;
    int c1;
    while (true) {
        c1 = i * 2 + sizeof(Point) * 1;
        int c2 = i * 2 + sizeof(Point) * 2;
        if (c1 >= end) {
            goto insert;
        }

        if (c2 >= end) {
            goto last;
        }

        int highc = (getrank(heap + c2) > getrank(heap + c1)) ? c2 : c1;

        if (value >= getrank(heap + highc)) {
            goto insert;
        }

        memcpy(heap + i, heap + highc, sizeof(Point));
        i = highc;
    }

last:
    if (value < getrank(heap + c1)) {
        memcpy(heap + i, heap + c1, sizeof(Point));
        i = c1;
    }

insert:
    memcpy(heap + i, heap + end, sizeof(Point));
}

static __forceinline void push_heap(Point* heap, int lastpos, int8_t newid, int32_t newrank, float x, float y) {
    assert(lastpos != 0);
    do {
        int parent = (lastpos - 1) / 2;
        if (heap[parent].rank >= newrank) {
            break;
        }

        heap[lastpos] = heap[parent];
        lastpos = parent;
    } while (lastpos != 0);

    
    heap[lastpos].id = newid;
    heap[lastpos].rank = newrank;
    heap[lastpos].x = x;
    heap[lastpos].y = y;
}

int32_t __stdcall search_alt(SearchContext* sc, const Rect rect, const int32_t count, Point* out_points)
{
    if (sc->aligned_begin == sc->aligned_end) {
        return 0;
    }

    Point* bestheap = (Point*)alloca(count * sizeof(Point));

    // TODO: A deque size of 1000 should be sifficient for most workloads.
    std::deque<uint32_t> remaining;
    for (int i = 0; i < count; ++i) {
        bestheap[i].rank = std::numeric_limits<int32_t>::max();
    }
 
    remaining.push_back(0);
    while (!remaining.empty()) {

        block& b = sc->aligned_begin[remaining.front()];
        remaining.pop_front();
        bool seen_better = false;
        for (int i = 0; i < points_per_block; ++i) {
            float x = b.xs[i];
            float y = b.ys[i];
            int32_t rank = b.rankid[i] >> 8;
            if (bestheap->rank > rank) {
                seen_better = true;
                if (x >= rect.lx && x <= rect.hx && y >= rect.ly && y <= rect.hy) {
                    pop_heap_raw((char*)(void*)bestheap, count);
                    push_heap(bestheap, count - 1, (int8_t)(b.rankid[i] & 0xFF), rank, x, y);
                }
            }
        }

        if (seen_better) {
            float x = b.xs[points_per_block - 1];
            float y = b.ys[points_per_block - 1];
            bool islx = rect.lx < x;
            bool ishx = rect.hx >= x;
            bool isly = rect.ly < y;
            bool ishy = rect.hy >= y;

            if (islx && isly && b.children[lxly]) {
                remaining.push_back(b.children[lxly]);
            }

            if (islx && ishy && b.children[lxhy]) {
                remaining.push_back(b.children[lxhy]);
            }

            if (ishx && isly && b.children[hxly]) {
                remaining.push_back(b.children[hxly]);
            }

            if (ishx && ishy && b.children[hxhy]) {
                remaining.push_back(b.children[hxhy]);
            }
        }
    }

    int left = count;
    while (left != 0 && bestheap->rank == std::numeric_limits<int32_t>::max()) {
        pop_heap_raw((char*)(void*)bestheap, left);
        --left;
    }

    uint32_t result = left;
    for (int i = ((int)result) - 1; i >= 0; --i) {
        out_points[i] = *bestheap;
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
