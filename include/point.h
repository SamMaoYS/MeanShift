//
// Created by sam on 2020-08-05.
//

#ifndef SEGMENTATION_POINT_H
#define SEGMENTATION_POINT_H


template<typename T>
class Point {
public:
    Point() = default;
    ~Point() = default;

    virtual Point<T> operator+(const Point<T> &p) const {}
    virtual Point<T> &operator+=(const Point<T> &p) {}
    virtual Point<T> operator+(T s) const {}
    virtual Point<T> &operator+=(T s) {}

    virtual Point<T> operator-(const Point<T> &p) const {}
    virtual Point<T> &operator-=(const Point<T> &p) {}
    virtual Point<T> operator-(T s) const {}
    virtual Point<T> &operator-=(T s) {}

    virtual Point<T> operator*(const Point<T> &p) const {}
    virtual Point<T> &operator*=(const Point<T> &p) {}
    virtual Point<T> operator*(T s) const {}
    virtual Point<T> &operator*=(T s) {}

    virtual Point<T> operator/(T s) const {}
    virtual Point<T> &operator/=(T s) {}

    virtual bool operator==(const Point<T> &p) const {}

    virtual T squaredNorm() const {}
    virtual T norm() const {}
    virtual T dot(const Point<T> &p) const {}
};


#endif //SEGMENTATION_POINT_H
