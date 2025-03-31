// CRTP demo
// 1. Upside Down inheritance
// 2. polymorphism without virtual functions
#include <iostream>

template <typename Derived>
class Base {
public:
    void interface() {
        static_cast<Derived*>(this)->implementation();
    }

    static void static_interface() {
        Derived::static_implementation();
    }
};

class Derived : public Base<Derived> {
public:
    void implementation() {
        std::cout << "implementation" << std::endl;
    }

    static void static_implementation() {
        std::cout << "static implementation" << std::endl;
    }
};

int main() {
    Derived d;
    d.interface();
    Base<Derived>::static_interface();
    return 0;
}

