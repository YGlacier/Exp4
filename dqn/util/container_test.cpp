#include "util/container.hpp"
#include "gtest/gtest.h"

namespace util {
namespace container {

TEST(Container, VectorToOptionalArray) {
  constexpr size_t N = 4;
  const auto optional_array =
      VectorToOptionalArray<N>(std::vector<int>{ 1, 2 });
  ASSERT_TRUE(optional_array[0]);
  ASSERT_TRUE(optional_array[1]);
  ASSERT_FALSE(optional_array[2]);
  ASSERT_FALSE(optional_array[3]);
  ASSERT_EQ(*optional_array[0], 1);
  ASSERT_EQ(*optional_array[1], 2);
}

TEST(Container, OptionalArrayToVector) {
  constexpr auto N = 4;
  std::array<boost::optional<int>, N> optional_array;
  optional_array[0] = 1;
  optional_array[2] = 2;
  const auto v = OptionalArrayToVector(optional_array);
  ASSERT_EQ(v.size(), 2);
  ASSERT_EQ(v[0], 1);
  ASSERT_EQ(v[1], 2);
}

TEST(Container, AsArray) {
  const std::vector<int> v = {1, 2};
  const auto a = AsArray<2>(v);
  ASSERT_EQ(a.size(), v.size());
  ASSERT_EQ(a[0], 1);
  ASSERT_EQ(a[1], 2);
}

TEST(Container, AsVector) {
  const std::array<int, 2> a = {1, 2};
  const auto v = AsVector(a);
  ASSERT_EQ(v.size(), a.size());
  ASSERT_EQ(v[0], 1);
  ASSERT_EQ(v[1], 2);
}

}
}
