#include <iostream>
#include <vector>
#include <type_traits>

// 1. SFINAE
template <typename T>
typename std::enable_if<std::is_integral<T>::value, void>::type
print(T value) {
    std::cout << "value is " << value << "\n";
}

// 2. specialized template
template <>
void print(uint32_t value) {
    std::cout << "specialized template print value " << value << "\n";
}

// 3. template meta programming
template <int N>
struct Factorial {
    static const int value = N * Factorial<N - 1>::value;
};

template <>
struct Factorial<0> {
    static const int value = 1;
};

// 4. variadic templates
void printN() {
    std::cout << "\n";
}

template <typename T, typename... Args>
void printN(T first, Args... args) {
    std::cout << first << " ";
    printN(args...);
}

// 5. using auto decltype
template <typename T>
auto get_value(T t) -> decltype(t + 0) {
    return t + 0;
}

// 6. concepts

// 7. CRTP
template<typename Derived>
class Base {
public:
    void interface() {
        static_cast<Derived*>(this)->implementation();
    }
};

class Derived : public Base<Derived> {
public:
    void implementation() {
        std::cout << "Implementation in Derived" << std::endl;
    }
};

// 8. type traits
template<typename T>
void check_type_trait(T t) {
    if constexpr (std::is_pointer<T>::value) {
        std::cout << "It's a pointer." << std::endl;
    }
    else {
        std::cout << "It's not a pointer." << std::endl;
    }
}

// 9. perfect forwarding
// advantages over function overload
// 减少代码冗余、支持可变参数、提高代码通用性和避免不必要的复制移动方面
template<typename T>
void wrapper(T&& arg) {
    //actualFunction(std::forward<T>(arg)); // lvalue
    //actualFunction(std::forward<T>(arg));
    //actualFunction(std::move<T>(arg)); // rvalue
}

// 10. template of template
template<template<typename> class Container, typename T>
class MyClass {
    Container<T> container;
};

// 11. Policy-based Design
template<typename Policy>
class MyClass : public Policy {
public:
    void operation() {
        this->invoke();
    }
};

// 12. Mixin Inheritance
template<typename Base>
class LoggingMixin : public Base {
public:
    void logOperation() {
        std::cout << "Logging before operation." << std::endl;
        Base::operation();
    }
};

// 13. alias
template <typename T>
using Vec = std::vector<T>;

// 14. assert

int main() {
    print(3);
    print(3u);
    printN(1, 1.5, "hello");
    std::cout << get_value(42) << "\n";
    check_type_trait(10);
    return 0;
}
