#include <cmath>

#include "noarr/structures_extended.hpp"
#include "noarr/structures/extra/traverser.hpp"

using num_t = float;

namespace {

void kernel_deriche(num_t alpha, auto imgIn, auto imgOut, auto y1, auto y2) {
    num_t k;
    num_t a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, c1, c2;
    k = ((num_t)1.0 - std::exp(-alpha)) * ((num_t)1.0 - std::exp(-alpha)) / ((num_t)1.0 + (num_t)2.0 * alpha * std::exp(-alpha) - std::exp(((num_t)2.0 * alpha)));
    a1 = a5 = k;
    a2 = a6 = k * std::exp(-alpha) * (alpha - (num_t)1.0);
    a3 = a7 = k * std::exp(-alpha) * (alpha + (num_t)1.0);
    a4 = a8 = -k * std::exp(((num_t)(-2.0) * alpha));
    b1 = std::pow((num_t)2.0, -alpha);
    b2 = -std::exp(((num_t)(-2.0) * alpha));
    c1 = c2 = 1;

    noarr::traverser(imgIn, y1)
        .template for_dims<'i'>([=](auto inner) {
            num_t ym1 = 0;
            num_t ym2 = 0;
            num_t xm1 = 0;
            
            inner
                .template for_each<'j'>([=, &ym1, &ym2, &xm1](auto state) {
                    y1[state] = a1 * imgIn[state] + a2 * xm1 + b1 * ym1 + b2 * ym2;
                    xm1 = imgIn[state];
                    ym2 = ym1;
                    ym1 = y1[state];
                });
        });

    noarr::traverser(imgIn, y2)
        .template for_dims<'i'>([=](auto inner) {
            num_t yp1 = 0;
            num_t yp2 = 0;
            num_t xp1 = 0;
            num_t xp2 = 0;
            
            inner
                .order(noarr::reverse<'j'>())
                .template for_each<'j'>([=, &yp1, &yp2, &xp1, &xp2](auto state) {
                    y2[state] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
                    xp2 = xp1;
                    xp1 = imgIn[state];
                    yp2 = yp1;
                    yp1 = y2[state];
                });
        });


    noarr::traverser(y1, y2, imgOut)
        .for_each([=](auto state) {
            imgOut[state] = c1 * (y1[state] + y2[state]);
        });
    
    noarr::traverser(imgOut, y1)
        .template for_dims<'j'>([=](auto inner) {
            num_t tm1 = 0;
            num_t ym1 = 0;
            num_t ym2 = 0;
            
            inner
                .template for_each<'i'>([=, &tm1, &ym1, &ym2](auto state) {
                    y1[state] = a5 * imgOut[state] + a6 * tm1 + b1 * ym1 + b2 * ym2;
                    tm1 = imgOut[state];
                    ym2 = ym1;
                    ym1 = y1[state];
                });
        });

    noarr::traverser(imgOut, y2)
        .template for_dims<'j'>([=](auto inner) {
            num_t tp1 = 0;
            num_t tp2 = 0;
            num_t yp1 = 0;
            num_t yp2 = 0;
            
            inner
                .order(noarr::reverse<'i'>())
                .template for_each<'i'>([=, &tp1, &tp2, &yp1, &yp2](auto state) {
                    y2[state] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
                    tp2 = tp1;
                    tp1 = imgOut[state];
                    yp2 = yp1;
                    yp1 = y2[state];
                });
        });

    noarr::traverser(y1, y2, imgOut)
        .for_each([=](auto state) {
            imgOut[state] = c2 * (y1[state] + y2[state]);
        });
}

} // namespace
