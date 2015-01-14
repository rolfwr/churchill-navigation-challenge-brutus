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



struct rawpoint
{
    void copyfrom(const Point& point) {
        rankid = (((uint32_t) point.rank) << 8) | (((uint32_t)point.id) & 0xFF);
        x = point.x;
        y = point.y;
    }

    Point get() const {
        return Point{ rankid & 0xFF, rankid >> 8, x, y };
    }


    uint32_t rankid;
    float x;
    float y;
};

struct SearchContext {
    int counts[256][256];
    rawpoint points[256][256][256];
    float xscale[256];
    float yscale[256];
    std::vector<rawpoint> extra;
    int all;
};

extern "C" {

__declspec(dllexport) SearchContext* __stdcall create(const Point* points_begin, const Point* points_end) {
    auto sc = new SearchContext{};
    uint32_t all = (int)std::distance(points_begin, points_end);
    sc->all = all;
    int samplecount = all; // std::min((int)all, 256 * 10);
    if (samplecount > 0) {

        std::vector<float> xs(samplecount);
        std::vector<float> ys(samplecount);

        for (int i = 0; i < samplecount; ++i) {
            xs[i] = points_begin[i].x;
            ys[i] = points_begin[i].y;
        }

        std::sort(xs.begin(), xs.end());
        std::sort(ys.begin(), ys.end());

        for (int i = 0; i < 256; ++i) {
            float norm = i / 257.0F + 1.0F / 257.0F;
            int j = (int)(norm * samplecount);
            assert(j >= 0);
            assert(j < samplecount);
            sc->xscale[i] = xs[j];
            sc->yscale[i] = ys[j];
        }
    }

    for (uint32_t i = 0; i < all; ++i) {
        rawpoint rp;
        rp.copyfrom(points_begin[i]);
        auto xd = std::upper_bound(&sc->xscale[0], &sc->xscale[255], rp.x);
        int xi = (int)std::distance(&sc->xscale[0], xd);
        auto yd = std::upper_bound(&sc->yscale[0], &sc->yscale[255], rp.y);
        int yi = (int)std::distance(&sc->yscale[0], yd);
        assert(xi >= 0);
        assert(xi < 256);
        assert(yi >= 0);
        assert(yi < 256);

        int count = sc->counts[xi][yi];
        if (count == 256) {
            sc->extra.push_back(rp);
        }
        else {
            sc->points[xi][yi][count] = rp;
            sc->counts[xi][yi] = count + 1;
        }
    }

    for (int xi = 0; xi < 256; ++xi) {
        for (int yi = 0; yi < 256; ++yi) {
            int count = sc->counts[xi][yi];

            std::sort(&sc->points[xi][yi][0], &sc->points[xi][yi][count],
                [](const rawpoint& a, const rawpoint& b) {
                return a.rankid < b.rankid;
            });

            if (count != 256) {
                // NAN terminate.
                sc->points[xi][yi][count].x = NAN;
            }
        }
    }

    std::cout << "\n\nextra: " << sc->extra.size() << "\n\n";
    return sc;
};



__declspec(dllexport) int32_t __stdcall search(SearchContext* sc, const Rect rect, const int32_t count, Point* out_points) {

    auto xld = std::upper_bound(&sc->xscale[0], &sc->xscale[255], rect.lx);
    int xli = (int)std::distance(&sc->xscale[0], xld);
    auto xhd = std::upper_bound(&sc->xscale[0], &sc->xscale[255], rect.hx);
    int xhi = (int)std::distance(&sc->xscale[0], xhd);
    auto yld = std::upper_bound(&sc->yscale[0], &sc->yscale[255], rect.ly);
    int yli = (int)std::distance(&sc->yscale[0], yld);
    auto yhd = std::upper_bound(&sc->yscale[0], &sc->yscale[255], rect.hy);
    int yhi = (int)std::distance(&sc->yscale[0], yhd);

    std::vector<rawpoint> fetched;

    bool needextra = false;
    for (int xi = xli; xi <= xhi; ++xi) {
        for (int yi = yli; yi <= yhi; ++yi) {
            for (int c = 0; c < count; c++) { 
                if (sc->points[xi][yi][c].x == NAN) {
                    goto nextcell;
                }
                fetched.push_back(sc->points[xi][yi][c]);
            }
            needextra = true;

        nextcell:
            ;
        }
    }

    if (needextra) {
        for (auto p : sc->extra) {
            fetched.push_back(p);
        }
    }

    std::sort(fetched.begin(), fetched.end(),
        [](const rawpoint& a, const rawpoint& b) {
        return a.rankid < b.rankid;
    });

    int got = std::min(count, (int32_t)fetched.size());

    for (int i = 0; i < got; ++i) {
        out_points[i] = fetched[i].get();
    }

    return got;
};

__declspec(dllexport) SearchContext* __stdcall destroy(SearchContext* sc) {
    delete sc;
    return nullptr;
};

}
