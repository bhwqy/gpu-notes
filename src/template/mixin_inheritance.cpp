#include <iostream>

// 基础类
class Base {
public:
    void baseFunction() const {
        std::cout << "Base function" << std::endl;
    }
};

// Mixin类1
template<typename T>
class Mixin1 : public T {
public:
    void mixin1Function() const {
        std::cout << "Mixin1 function" << std::endl;
    }
};

// Mixin类2
template<typename T>
class Mixin2 : public T {
public:
    void mixin2Function() const {
        std::cout << "Mixin2 function" << std::endl;
    }
};

// 使用Mixin类来扩展Base类
using MixedClass = Mixin2<Mixin1<Base>>;

int main() {
    MixedClass obj;

    // 调用基类的方法
    obj.baseFunction();

    // 调用Mixin1的方法
    obj.mixin1Function();

    // 调用Mixin2的方法
    obj.mixin2Function();

    return 0;
}
