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
    uint32_t pad;
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

float median(std::vector<float>& values) {
    std::sort(values.begin(), values.end());

    auto size = values.size();
    if (size % 2 == 1) {
        return values[size / 2];
    }

    return (values[size / 2 - 1] + values[size / 2]) / 2;
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

    // Find centermost point in the list of candidate points.
    std::vector<float> xsort(count);
    std::vector<float> ysort(count);
    for (int i = 0; i < count; ++i) {
        xsort[i] = candidates[i].x;
        ysort[i] = candidates[i].y;
    }

    float medianx = median(xsort);
    float mediany = median(ysort);

    float bestdist = std::numeric_limits<float>::max();
    int bestindex = -1;
    for (int i = 0; i < count; ++i) {
        float dist = abs(medianx - candidates[i].x) + abs(mediany - candidates[i].y);

        if (dist < bestdist) {
            bestdist = dist;
            bestindex = i;
        }
    }

    // Move most center point to end of block.
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

    //assert((sizeof(block) % 64) == 0);
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

int32_t __stdcall search_alt(SearchContext* sc, const Rect rect, const int32_t count, Point* out_points) {
    if (sc->aligned_begin == sc->aligned_end) {
        return 0;
    }

    std::deque<uint32_t> remaining;
    auto comp = [](const Point& a, const Point& b) {
        return a.rank < b.rank;
    };

    std::priority_queue<Point, std::vector<Point>, decltype(comp)> best;
    Point bad;
    bad.rank = std::numeric_limits<int32_t>::max();
    for (int i = 0; i < count; ++i) {
        best.push(bad);
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
            if (best.top().rank > rank) {
                seen_better = true;
                if (x >= rect.lx && x <= rect.hx && y >= rect.ly && y <= rect.hy) {
                    best.pop();
                    best.push(Point{ (int8_t)(b.rankid[i] & 0xFF), rank, x, y });
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

    while ((!best.empty()) && best.top().rank == std::numeric_limits<int32_t>::max()) {
        best.pop();
    }

    uint32_t result = (uint32_t)best.size();
    for (int i = ((int)result) - 1; i >= 0; --i) {
        out_points[i] = best.top();
        best.pop();
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
