#include "../../challenge/point_search.h"
#include <vector>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <cmath>

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


struct SearchContext {
    std::vector<Point> points;
};



extern "C" {

__declspec(dllexport) SearchContext* __stdcall create(const Point* points_begin, const Point* points_end) {


    auto sc = new SearchContext();

    auto count = std::distance(points_begin, points_end);
    
    int stride = count / 20;

#ifdef DUMP
    for (int i = 0; i < count; i += stride) {
        std::cout << "[" << i << "] id=" << ((uint32_t)(uint8_t)points_begin[i].id) << ", rank=" << points_begin[i].rank << ", x=" << points_begin[i].x << ", y=" << points_begin[i].y << "\n";
    }
#endif

    sc->points.resize(count);
    std::copy(points_begin, points_end, sc->points.begin());
    std::sort(sc->points.begin(), sc->points.end(), [](const Point& a, const Point& b) {
        return a.rank < b.rank;
    });

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

#ifdef DUMP
    static int i = 0;
    const int stride = 100;
    if ((++i) % stride == 0) {
        float dx = rect.hx - rect.lx;
        float dy = rect.hy - rect.ly;
        std::cout << result << ", count=" << count << ", lx=" << rect.lx << ", hx=" << rect.hx << ", ly=" << rect.ly << ", hy=" << rect.hy << ", " << dx << "*" << dy << "\n";
    }
#endif
    return result;
};

__declspec(dllexport) SearchContext* __stdcall destroy(SearchContext* sc) {
    delete sc;
    return nullptr;
};

}
