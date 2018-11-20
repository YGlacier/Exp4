#ifndef UTIL_RANDOM_HPP_
#define UTIL_RANDOM_HPP_

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <deque>
#include <random>
#include <unordered_map>
#include <vector>

namespace util { namespace random {

inline std::mt19937& RandomEngine() {
  static std::mt19937 random_engine;
  return random_engine;
}

inline void SetSeed(const unsigned int seed) {
  std::srand(seed);
  RandomEngine().seed(seed);
}

inline int RandomInt(const int min, const int max) {
  assert(min <= max);
  std::uniform_int_distribution<int> dist(min, max);
  return dist(RandomEngine());
}

inline double RandomDouble(const double min, const double max) {
  assert(min <= max);
  std::uniform_real_distribution<double> dist(min, max);
  return dist(RandomEngine());
}

template <class T>
inline const T& RandomSelect(const std::vector<T>& v) {
  assert(!v.empty());
  if (v.size() == 1) {
    return v.front();
  }
  return v[RandomInt(0, v.size() - 1)];
}

template <class T>
inline const T& RandomSelect(const std::deque<T>& v) {
  assert(!v.empty());
  if (v.size() == 1) {
    return v.front();
  }
  return v[RandomInt(0, v.size() - 1)];
}

template <class K, class V>
inline const K& RandomSelectKey(const std::unordered_map<K, V>& m) {
  assert(!m.empty());
  if (m.size() == 1) {
    return m.begin()->first;
  }
  return std::next(m.begin(), RandomInt(0, m.size() - 1))->first;
}

template <class Iterator>
inline void RandomShuffle(const Iterator& begin, const Iterator& end) {
  std::shuffle(begin, end, RandomEngine());
}

}}


#endif /* UTIL_RANDOM_HPP_ */
