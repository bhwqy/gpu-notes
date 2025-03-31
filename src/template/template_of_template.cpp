#include <iostream>
#include <vector>

template<typename T, template<typename, typename...> class Container>
class ContainerWrapper {
private:
    Container<T> container;

public:
    void add(const T& value) {
        container.push_back(value);
    }

    void print() const {
        for (const auto& elem : container) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }
};

template<typename T>
struct Inner {
    void operator()() const {
        std::cout << "Inner with T = " << typeid(T).name() << std::endl;
    }
};

template<template<typename> class InnerTemplate>
struct Middle {
    InnerTemplate<int> innerInt;
    InnerTemplate<double> innerDouble;

    void operator()() const {
        innerInt();
        innerDouble();
    }
};

template<template<template<typename> class> class MiddleTemplate>
struct Outer {
    MiddleTemplate<Inner> middle;

    void operator()() const {
        middle();
    }
};

int main() {
    ContainerWrapper<int, std::vector> vecWrapper;
    vecWrapper.add(10);
    vecWrapper.add(20);
    vecWrapper.print();

    Outer<Middle>()();
    return 0;
}

