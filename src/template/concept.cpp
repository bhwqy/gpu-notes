#include <iostream>
#include <concepts>
#include <string>

template<typename T>
concept Printable = requires(std::ostream & os, T obj) {
    { os << obj } -> std::same_as<std::ostream&>;
};

template<typename T>
concept Integral = std::is_integral_v<T>;

template<Integral T> requires Printable<T>
void printIntegral(T num) {
    std::cout << num << std::endl;
}

int main() {
    int m = 10;
    printIntegral(m);
    return 0;
}
