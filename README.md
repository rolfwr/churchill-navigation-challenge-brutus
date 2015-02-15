Churchill Navigation programming challenge
==========================================

This was my submission for the
[Churchill Navigation programming challenge](http://churchillnavigation.com/challenge/),
which ended up in fourth place (TwistingBrutus.dll).

This repository represents the complete development history, warts and all,
including both abandoned strategies and failed optimization attempts.

I recommend looking at
[Stefan Dessens solution](https://github.com/sDessens/churchill-challange)
which ended up in third place if you want to see a better algorithmic solution.

However it was interesting seeing how much speed improvements was possible by
gradually performing mostly non-algorithmic optimizations.

![improvements.png](https://bitbucket.org/repo/Rj8jdk/images/3546633172-improvements.png)

The final search algorithm was basically in place already at revision tag B2,
but the implementation got quite a lot faster by applying optimizations such
as:

* Better data balancing during point loading.
* Replacing std::deque and std::priority_queue containers with specialized code.
* Use compiler intrinsics to emit SIMD instructions for processing multiple points in one go.
* Use _mm_prefetch() to preload memory in order to reduce cache misses.
* Specialize for returning exactly 20 points.
* Tuning parameters such as points per block, number of blocks to prefetch.
* Eliminate conditionals in inner loop.