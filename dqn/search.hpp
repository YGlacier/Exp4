#ifndef SEARCH_HPP
#define SEARCH_HPP

#include <iostream>
#include <vector>
#include <unordered_map>
#include <functional>
#include <boost/range.hpp>
#include <boost/range/irange.hpp>
#include "aggregation_policy.hpp"
#include "util/container.hpp"
#include "prettyprint.hpp"

namespace dqn {

namespace A = boost::adaptors;

using ActionPlan = std::vector<size_t>;
using ActionEvaluation = std::unordered_map<size_t, double>;
using BatchPlanEvaluator =
    std::function<std::vector<double>(const std::vector<ActionPlan> &)>;
using ActionPlanAndValue = std::pair<ActionPlan, double>;

inline std::vector<double>
EvaluateActionPlans(const std::vector<ActionPlan> &plans,
                    const BatchPlanEvaluator &batch_plan_evaluator,
                    const size_t batch_size) {
  std::vector<double> result;
  result.reserve(plans.size());
  for (auto offset = 0; offset < plans.size(); offset += batch_size) {
    const auto size = std::min(plans.size() - offset, batch_size);
    assert(size > 0);
    const auto plan_batch = std::vector<ActionPlan>(
        plans.begin() + offset, plans.begin() + offset + size);
    assert(plan_batch.size() > 0);
    const auto batch_eval = batch_plan_evaluator(plan_batch);
    assert(batch_eval.size() == size);
    boost::copy(batch_eval, std::back_inserter(result));
  }
  return result;
}

template <typename T>
inline std::vector<std::vector<T> >
GetProduct(const std::vector<std::vector<T> > &a,
           const std::vector<std::vector<T> > &b) {
  std::vector<std::vector<T> > result;
  for (const auto &a_elem : a) {
    for (const auto &b_elem : b) {
      std::vector<T> tmp;
      tmp.reserve(a_elem.size() + b_elem.size());
      boost::copy(a_elem, std::back_inserter(tmp));
      boost::copy(b_elem, std::back_inserter(tmp));
      result.push_back(tmp);
    }
  }
  return result;
}

template <typename T>
inline std::vector<std::vector<T> >
GetProduct(const std::vector<std::vector<T> > &a, const size_t count) {
  assert(count > 0);
  if (count == 1) {
    return a;
  } else if (count == 2) {
    return GetProduct(a, a);
  } else {
    return GetProduct(a, GetProduct(a, count - 1));
  }
}

inline double FullForwardSearchForSubplans(
    const std::vector<ActionPlanAndValue> &plan_and_values,
    const AggregationPolicy policy) {
  if (plan_and_values.front().first.size() == 1) {
    return Aggregate(plan_and_values, policy);
  } else {
    std::unordered_map<size_t, std::vector<ActionPlanAndValue> > subplans;
    for (const auto &plan_and_value : plan_and_values) {
      assert(plan_and_value.first.size() > 1);
      subplans[plan_and_value.first.front()].emplace_back(
          ActionPlan(plan_and_value.first.begin() + 1,
                     plan_and_value.first.end()),
          plan_and_value.second);
    }
    const auto action_and_values = util::container::AsVector(
        util::container::AsVector(subplans) |
        A::transformed([&](const auto &action_and_subplans) {
          return std::make_pair(
              action_and_subplans.first,
              FullForwardSearchForSubplans(action_and_subplans.second, policy));
        }));
    return Aggregate(action_and_values, policy);
  }
}

inline std::unordered_map<size_t, double>
FullForwardSearch(const std::vector<ActionPlanAndValue> &plan_and_values,
                  const AggregationPolicy policy) {
  if (plan_and_values.front().first.size() == 1) {
    return util::container::AsMap(
        plan_and_values | A::transformed([](const auto &p) {
                            return std::make_pair(p.first.front(), p.second);
                          }));
  } else {
    std::unordered_map<size_t, std::vector<ActionPlanAndValue> > subplans;
    for (const auto &plan_and_value : plan_and_values) {
      assert(plan_and_value.first.size() > 1);
      subplans[plan_and_value.first.front()].emplace_back(
          ActionPlan(plan_and_value.first.begin() + 1,
                     plan_and_value.first.end()),
          plan_and_value.second);
    }
    const auto action_and_plan_values = util::container::AsVectorMap(
        subplans |
        A::transformed([&](const auto &action_and_subplans) {
          return std::make_pair(
              action_and_subplans.first,
              FullForwardSearchForSubplans(action_and_subplans.second, policy));
        }));
    return util::container::AsMap(action_and_plan_values |
                                  A::transformed([](const auto &p) {
                                    return std::make_pair(
                                        p.first, *boost::max_element(p.second));
                                  }));
  }
}

inline std::unordered_map<size_t, double> FullForwardSearch(
    const std::vector<size_t> &action_indices, const size_t depth,
    const BatchPlanEvaluator &batch_plan_evaluator,
    const AggregationPolicy policy, const size_t search_unit,
    const size_t batch_size) {
  assert(depth > 0);
  assert(std::is_sorted(action_indices.begin(), action_indices.end()));
  assert(search_unit > 0);
  const auto plans = GetProduct(
      util::container::AsVector(action_indices |
                                A::transformed([&](const auto action_idx) {
                                  return std::vector<size_t>(search_unit,
                                                             action_idx);
                                })),
      depth);
  const auto plan_values =
      EvaluateActionPlans(plans, batch_plan_evaluator, batch_size);
  assert(plans.size() == plan_values.size());
  //  std::cout << "plans:" << plans << std::endl;
  //  std::cout << "plan_values:" << plan_values << std::endl;
  const auto plan_and_values = util::container::AsVector(
      boost::irange(static_cast<size_t>(0), plans.size()) |
      A::transformed([&](const auto i) {
        return std::make_pair(plans.at(i), plan_values.at(i));
      }));
  return FullForwardSearch(plan_and_values, policy);
}
}

#endif // SEARCH_HPP
