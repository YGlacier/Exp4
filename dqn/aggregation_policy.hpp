#ifndef AGGREGATION_POLICY_HPP
#define AGGREGATION_POLICY_HPP

#include <stdexcept>
#include <unordered_map>
#include <boost/range.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/adaptors.hpp>
#include <boost/range/numeric.hpp>

namespace dqn {

enum class AggregationPolicy { Max, Min, Mean };

inline AggregationPolicy ParseAggregationPolicy(const std::string &s) {
  static const std::unordered_map<std::string, AggregationPolicy> m = {
    { "max", AggregationPolicy::Max },
    { "min", AggregationPolicy::Min },
    { "mean", AggregationPolicy::Mean },
  };
  if (!m.count(s)) {
    throw std::runtime_error(s + " cannot be parsed as AggregationPolicy.");
  } else {
    return m.at(s);
  }
}

template <typename Rng>
inline double Aggregate(const Rng &rng, const AggregationPolicy policy) {
  switch (policy) {
  case AggregationPolicy::Max:
    return *boost::max_element(boost::adaptors::values(rng));
  case AggregationPolicy::Min:
    return *boost::min_element(boost::adaptors::values(rng));
  case AggregationPolicy::Mean:
    return boost::accumulate(boost::adaptors::values(rng), 0.0) / rng.size();
  default:
    throw std::runtime_error("Not supported aggregation policy.");
  }
}
// template <typename Key, typename Value>
// inline Value Aggregate(const std::unordered_map<Key, Value> &eval,
//                       const AggregationPolicy policy) {
//  switch (policy) {
//  case AggregationPolicy::None:
//    throw std::runtime_error(

//        "Do not make evaluation if aggregation policy is None.");
//  case AggregationPolicy::Max:
//    return *boost::max_element(boost::adaptors::values(eval));
//  case AggregationPolicy::Min:
//    return *boost::min_element(boost::adaptors::values(eval));
//  case AggregationPolicy::Mean:
//    return boost::accumulate(boost::adaptors::values(eval),
//                             static_cast<Value>(0)) /
//           eval.size();
//  default:
//    throw std::runtime_error("Not supported aggregation policy.");
//  }
//}
}

#endif // AGGREGATION_POLICY_HPP
