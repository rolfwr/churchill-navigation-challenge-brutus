#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include "../../challenge/point_search.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include <limits>
#include <cmath>
#include <Windows.h>
#include <cassert>

#define NDUMP 1

//#define RAWFLOAT 1

#ifdef RAWFLOAT
typedef uint32_t float_internal;
static inline float_internal toint(float value) {
    uint32_t bits = reinterpret_cast<const uint32_t&>(value);
    return (float_internal)(bits ^ ((bits >> 31) * 0x7fffffff));
}

static inline float tofloat(float_internal value) {
    uint32_t bits = (uint32_t)value;
    uint32_t ubits = (bits ^ ((bits >> 31) * 0x7ffffff));
    float result = reinterpret_cast<const float&>(ubits);
    return result;
}


#else

typedef float float_internal;
static inline float_internal toint(float value) {
    return value;
}

static inline float tofloat(float_internal value) {
    return value;
}

#endif

struct block {
    uint32_t flags;
    float_internal divs[7];
    uint32_t children[8];
};

struct rawpoint
{
    void copyfrom(const Point& point) {
        rankid = (((uint32_t) point.rank) << 8) | (((uint32_t)point.id) & 0xFF);
        x = toint(point.x);
        y = toint(point.y);
        //id = point.id;
    }

    Point get() const {
        return Point{ rankid & 0xFF, rankid >> 8, tofloat(x), tofloat(y) };
    }


    uint32_t rankid;
    float_internal x;
    float_internal y;
};

struct rankset {
    rawpoint* begin;
    rawpoint* end;
};

struct SearchContext {
    std::vector<rawpoint> points;
    std::vector<rankset> sets;
    std::vector<block> blocks;
};



extern "C" {

__declspec(dllexport) SearchContext* __stdcall create(const Point* points_begin, const Point* points_end) {
    block bl = {};
    assert(sizeof(bl) == 64);

    auto sc = new SearchContext();

    auto count = std::distance(points_begin, points_end);
    
    int stride = ((int)count) / 20;

#ifdef DUMP
    for (int i = 0; i < count; i += stride) {
        std::cout << "[" << i << "] id=" << ((uint32_t)(uint8_t)points_begin[i].id) << ", rank=" << points_begin[i].rank << ", x=" << points_begin[i].x << ", y=" << points_begin[i].y << "\n";
    }
#endif

    sc->points.resize(count);
    auto outitr = sc->points.begin();
    for (auto itr = points_begin; itr != points_end; ++itr) {
        outitr->copyfrom(*itr);
        ++outitr;
    }

    //   0...3 | null | 4..6 | null | 7..8
    //         3      3      6      7    
    const int segments = 8;
    
    sc->blocks.push_back(block{});
    int blockindex = 1;
    block& newblock = sc->blocks.back();
    

    auto first = &sc->points.data()[0];
    int max = (int)(count * segments);
    float divval = std::numeric_limits<float>::lowest();
    int setindex = 0;
    int j = 0;
    for (int i = 0; i < max; ++j) {
        int s = i / segments;
        i += (int)count;
        int e = i / segments;

        auto firstinset = first + s;
        sc->sets.push_back(rankset{ firstinset, first + e });
        newblock.children[j] = setindex;
        setindex++;

        if (j > 0) {
            newblock.divs[j - 1] = divval;
        }
        if (s != e) {
            divval = (first + (e - 1))->x;
        }
        


    }



    //sc->sets.push_back(rankset{ &sc->points.data()[0], &sc->points.data()[count] });

    for (rankset& rs : sc->sets) {
        std::sort(rs.begin, rs.end, [](const rawpoint& a, const rawpoint& b) {
            return a.rankid < b.rankid;
        });
    }

#ifdef DUMP
    float xmin = std::numeric_limits<float>::max();
    float ymin = std::numeric_limits<float>::max();
    float xmax = std::numeric_limits<float>::lowest();
    float ymax = std::numeric_limits<float>::lowest();

    int histogram[256] = {};

    for (int i = 0; i < count; ++i) {
        const Point& point = points_begin[i];

        // Outlier check.
        if (point.x < -1e+9 || point.x > 1e+9 || point.y < -1e+9 || point.y > 1e+9) {
            continue;
        }

        if (point.x < xmin) {
            xmin = point.x;
        }

        if (point.x > xmax) {
            xmax = point.x;
        }

        if (point.y < ymin) {
            ymin = point.y;
        }

        if (point.y > ymax) {
            ymax = point.y;
        }

        int exp;
        frexp(point.x, &exp);
        int index = exp + 100;
        histogram[index]++;
    }

    float scale = 255.0 / (nextafterf(xmax, std::numeric_limits<float>::max()) - xmin);


    for (int i = 0; i < count; ++i) {
        const Point& point = points_begin[i];

        // Outlier check.
        if (point.x < -1e+9 || point.x > 1e+9 || point.y < -1e+9 || point.y > 1e+9) {
            continue;
        }

        int index = (int)((point.x - xmin) * scale);
        histogram[index]++;
    }

    std::cout << "\n\nRange: x~[" << xmin << "," << xmax << "], y~[" << ymin << "," << ymax << "]\n";

    for (int i = 0; i < 256; ++i) {
        std::cout << "bucket " << i << " : " << histogram[i] << "\n";
    }
#endif
    return sc;
};



__declspec(dllexport) int32_t __stdcall search(SearchContext* sc, const Rect rect, const int32_t count, Point* out_points) {
    int setcount = (int)sc->sets.size();
    std::vector<rawpoint> found(setcount * count);

    float_internal lx = toint(rect.lx);
    float_internal hx = toint(rect.hx);
    float_internal ly = toint(rect.ly);
    float_internal hy = toint(rect.hy);


    block& tree = sc->blocks.front();

    std::vector<rankset> selected;
    float_internal* tbegin = &tree.divs[0];
    float_internal* tend = &tree.divs[7];

    float_internal* low = std::lower_bound(tbegin, tend, lx);
    int si = (int) std::distance(&tree.divs[0], low);
    float_internal* high = std::upper_bound(tbegin, tend, hx);
    int ei = (int) std::distance(tbegin, high);
    si = 0;
    ei = 7;
    auto outptr = found.begin();

    for (int i = si; i <= ei; ++i) {
        int index = tree.children[i];
        if (index == 0) {
            continue;
        }
        rankset& rset = sc->sets[index];
        auto pos = rset.begin;
        auto end = rset.end;
        int remaining = count;
        while (pos != end) {
            if (pos->x >= lx && pos->x <= hx && pos->y >= ly && pos->y <= hy) {
                *outptr++ = *pos;
                if (--remaining == 0) {
                    break;
                }
            }
            ++pos;
        }
    }

    std::sort(found.begin(), outptr, [](const rawpoint& a, const rawpoint& b) {
        return a.rankid < b.rankid;
    });

    auto outitr = out_points;

    auto itr = found.begin();
    auto found_count = std::distance(itr, outptr);
    auto enditr = itr + std::min((long long)count, found_count);
    for (; itr != enditr; ++itr) {
        *outitr++ = itr->get();
    }

#ifdef DUMP
    static int i = 0;
    const int stride = 100;
    if ((++i) % stride == 0) {
        float dx = rect.hx - rect.lx;
        float dy = rect.hy - rect.ly;
        std::cout << result << ", count=" << count << ", lx=" << rect.lx << ", hx=" << rect.hx << ", ly=" << rect.ly << ", hy=" << rect.hy << ", " << dx << "*" << dy << "\n";
    }
#endif
    return (int32_t) std::distance(out_points, outitr);
};

__declspec(dllexport) SearchContext* __stdcall destroy(SearchContext* sc) {
    delete sc;
    return nullptr;
};

}
