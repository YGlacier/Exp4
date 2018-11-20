#ifndef UTIL_CONTAINER_HPP_
#define UTIL_CONTAINER_HPP_

#include <algorithm>
#include <array>
#include <unordered_map>
#include <vector>
#include <boost/optional.hpp>
#include <boost/range.hpp>

namespace util {
namespace container {

template <size_t N, typename T>
std::array<boost::optional<T>, N> inline VectorToOptionalArray(
    const std::vector<T> &v) {
  assert(v.size() <= N);
  std::array<boost::optional<T>, N> optional_array;
  std::copy(v.begin(), v.end(), optional_array.begin());
  return optional_array;
}

template <typename T, size_t N>
std::array<T, N> inline VectorToArray(const std::vector<T> &v) {
  assert(v.size() == N);
  std::array<T, N> a;
  std::copy(v.begin(), v.end(), a.begin());
  return a;
}

template <typename T, size_t N>
std::vector<T> inline OptionalArrayToVector(
    const std::array<boost::optional<T>, N> &a) {
  std::vector<T> v;
  v.reserve(N);
  for (const auto &item : a) {
    if (item) {
      v.push_back(*item);
    }
  }
  return v;
}

template <typename T, size_t N>
std::array<bool, N> inline OptionalArrayToBoolArray(
    const std::array<boost::optional<T>, N> &a) {
  std::array<bool, N> bool_array;
  std::transform(
      a.begin(), a.end(), bool_array.begin(),
      [](const boost::optional<T> &op) { return static_cast<bool>(op); });
  return bool_array;
}

template <typename T1, typename T2>
inline bool SecondComparator(const std::pair<T1, T2> &a,
                             const std::pair<T1, T2> &b) {
  return a.second < b.second;
}

template <typename T, typename R>
inline std::function<boost::optional<R>(const boost::optional<T> &)>
Optionalize(const std::function<R(const T &)> &func) {
  return [&](const boost::optional<T> &op) {
    if (op) {
      return boost::optional<R>(func(op.get()));
    } else {
      return boost::optional<R>(boost::none);
    }
  };
}

template <typename Key, typename T, typename Hash, typename Pred,
          typename Alloc>
inline Key
KeyOfMaxValue(const std::unordered_map<Key, T, Hash, Pred, Alloc> &map) {
  return std::max_element(map.begin(), map.end(),
                          SecondComparator<Key, T>)->first;
}

template <size_t N, typename IteratorRange>
inline std::array<typename IteratorRange::value_type, N>
AsArray(const IteratorRange &range) {
  assert(range.size() <= N);
  const auto begin = boost::begin(range);
  const auto end = boost::end(range);
  std::array<typename IteratorRange::value_type, N> array;
  std::copy(begin, end, array.begin());
  return array;
}

template <typename IteratorRange>
inline std::vector<typename IteratorRange::value_type>
AsVector(const IteratorRange &range) {
  const auto begin = boost::begin(range);
  const auto end = boost::end(range);
  std::vector<typename IteratorRange::value_type> v;
  v.reserve(range.size());
  std::copy(begin, end, std::back_inserter(v));
  return v;
}

template <typename IteratorRange>
inline std::unordered_map<
    typename IteratorRange::value_type::first_type,
    std::vector<typename IteratorRange::value_type::second_type> >
AsVectorMap(const IteratorRange &range) {
  using ValueType = typename IteratorRange::value_type;
  using FirstType = typename ValueType::first_type;
  using SecondType = typename ValueType::second_type;
  std::unordered_map<FirstType, std::vector<SecondType> > m;
  for (const auto &p : range) {
    m[p.first].push_back(p.second);
  }
  return m;
}
template <typename IteratorRange>
inline std::unordered_map<typename IteratorRange::value_type::first_type,
                          typename IteratorRange::value_type::second_type>
AsMap(const IteratorRange &range) {
  using ValueType = typename IteratorRange::value_type;
  using FirstType = typename ValueType::first_type;
  using SecondType = typename ValueType::second_type;
  return std::unordered_map<FirstType, SecondType>(boost::begin(range),
                                                   boost::end(range));
}
}
}

#endif /* UTIL_CONTAINER_HPP_ */
