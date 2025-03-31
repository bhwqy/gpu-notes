#include <iostream>
#include <vector>
#include <algorithm>

struct AscendingOrder {
    static void sort(std::vector<int>& v) {
        std::sort(v.begin(), v.end());
    }
};

struct DescendingOrder {
    static void sort(std::vector<int>& v) {
        std::sort(v.begin(), v.end(), std::greater<int>());
    }
};

struct SimplePrint {
    static void print(const std::vector<int>& v) {
        for(auto& elem : v) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }
};

template<typename SortPolicy, typename PrintPolicy>
class Container {
    std::vector<int> elements;

public:
    void add(int element) {
        elements.push_back(element);
    }

    void sortElements() {
        SortPolicy::sort(elements);
    }

    void printElements() const {
        PrintPolicy::print(elements);
    }
};

int main() {
    Container<AscendingOrder, SimplePrint> ascendingContainer;
    ascendingContainer.add(5);
    ascendingContainer.add(3);
    ascendingContainer.add(8);
    ascendingContainer.sortElements();
    ascendingContainer.printElements();

    Container<DescendingOrder, SimplePrint> descendingContainer;
    descendingContainer.add(5);
    descendingContainer.add(3);
    descendingContainer.add(8);
    descendingContainer.sortElements();
    descendingContainer.printElements();

    return 0;
}
