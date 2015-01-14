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

struct rawpoint
{
    rawpoint(const Point& point)
        : rank(point.rank),
        x(reinterpret_cast<const int32_t&>(point.x)),
        y(reinterpret_cast<const int32_t&>(point.y)),
        id(point.id)
    {

    }

    Point get() const {
        return Point{ id, rank, reinterpret_cast<const float&>(x), reinterpret_cast<const float&>(x) };
    }

    int32_t rank;
    int32_t x;
    int32_t y;
    int8_t id;
};

struct block {
    float xs[4];
    float ys[4];
    uint32_t rankid[4];
    uint32_t children[4];
};

struct SearchContext {
    std::vector<Point> points;
    std::vector<block> blocks;
};

enum quadrant : int {
    lxly = 0,
    lxhy = 1,
    hxly = 2,
    hxhy = 3
};


uint32_t enblock(SearchContext& sc, Point* begin, Point* end) {
    auto count = std::min((int)std::distance(begin, end), 4);
    if (count == 0) {
        return 0;
    }

    uint32_t result = (uint32_t)sc.blocks.size();
    sc.blocks.push_back(block{});
    block& b = sc.blocks.back();
    b.xs[0] = std::numeric_limits<float>::max();
    b.xs[1] = std::numeric_limits<float>::max();
    b.xs[2] = std::numeric_limits<float>::max();
    b.xs[3] = std::numeric_limits<float>::max();

    for (int i = 0; i < count; ++i) {
        Point& p = *begin++;

        b.xs[i] = p.x;
        b.ys[i] = p.y;
        b.rankid[i] = (((uint32_t)p.rank) << 8) | (((uint32_t)p.id) & 0xFF);
    }

    if (count == 4) {
        float sepx = b.xs[3];
        float sepy = b.ys[3];

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

    assert(sizeof(block) == 64);
    auto sc = new SearchContext();

    auto count = std::distance(points_begin, points_end);
    sc->points.resize(count);
    std::copy(points_begin, points_end, sc->points.begin());
    std::sort(sc->points.begin(), sc->points.end(), [](const Point& a, const Point& b) {
        return a.rank < b.rank;
    });

    int bindex = enblock(*sc, &sc->points.data()[0], &sc->points.data()[count]);
    assert(bindex == 0);
    assert(sc->blocks.size() >= (size_t)(count / 4));

    

    std::sort(sc->points.begin(), sc->points.end(), [](const Point& a, const Point& b) {
        return a.rank < b.rank;
    });


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
    if (sc->blocks.empty()) {
        return 0;
    }
    int scanned = 0;
    std::stack<uint32_t> remaining;
    auto comp = [](const Point& a, const Point& b) {
        return a.rank < b.rank;
    };

    std::priority_queue<Point, std::vector<Point>, decltype(comp)> best;
    Point bad;
    bad.rank = std::numeric_limits<int32_t>::max();
    for (int i = 0; i < count; ++i) {
        best.push(bad);
    }
 
    remaining.push(0);
    while (!remaining.empty()) {
        block& b = sc->blocks[remaining.top()];
        remaining.pop();
        ++scanned;
        bool seen_better = false;
        for (int i = 0; i < 4; ++i) {
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
            float x = b.xs[3];
            float y = b.ys[3];
            bool islx = rect.lx < x;
            bool ishx = rect.hx >= x;
            bool isly = rect.ly < y;
            bool ishy = rect.hy >= y;

            if (islx && isly && b.children[lxly]) {
                remaining.push(b.children[lxly]);
            }

            if (islx && ishy && b.children[lxhy]) {
                remaining.push(b.children[lxhy]);
            }

            if (ishx && isly && b.children[hxly]) {
                remaining.push(b.children[hxly]);
            }

            if (ishx && ishy && b.children[hxhy]) {
                remaining.push(b.children[hxhy]);
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
