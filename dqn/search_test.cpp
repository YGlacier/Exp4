#include "gtest/gtest.h"
#include "search.hpp"

namespace dqn {

TEST(Search, FullForwardSearchForSubplans1ply) {
  const std::vector<ActionPlanAndValue> plan_and_values = { { { 1 }, 0.0 },
                                                            { { 2 }, 1.0 },
                                                            { { 3 }, 0.5 } };
  ASSERT_FLOAT_EQ(
      FullForwardSearchForSubplans(plan_and_values, AggregationPolicy::Max),
      1.0);
  ASSERT_FLOAT_EQ(
      FullForwardSearchForSubplans(plan_and_values, AggregationPolicy::Min),
      0.0);
  ASSERT_FLOAT_EQ(
      FullForwardSearchForSubplans(plan_and_values, AggregationPolicy::Mean),
      0.5);
}
TEST(Search, FullForwardSearchForSubplans2ply) {
  const std::vector<ActionPlanAndValue> plan_and_values = {
    { { 1, 1 }, 0.0 }, { { 1, 2 }, 1.0 }, { { 2, 2 }, 1.0 }, { { 2, 2 }, 2.0 }
  };
  ASSERT_FLOAT_EQ(
      FullForwardSearchForSubplans(plan_and_values, AggregationPolicy::Max),
      2.0);
  ASSERT_FLOAT_EQ(
      FullForwardSearchForSubplans(plan_and_values, AggregationPolicy::Min),
      0.0);
  ASSERT_FLOAT_EQ(
      FullForwardSearchForSubplans(plan_and_values, AggregationPolicy::Mean),
      1.0);
}

TEST(Search, FullForwardSearch2ply) {
  const std::vector<ActionPlanAndValue> plan_and_values = {
    { { 1, 1 }, 1.5 }, { { 1, 2 }, 2.0 }, { { 2, 2 }, 1.0 }, { { 2, 2 }, 3.0 }
  };
  {
    const auto max_result =
        FullForwardSearch(plan_and_values, AggregationPolicy::Max);
    ASSERT_FLOAT_EQ(max_result.at(1), 2.0);
    ASSERT_FLOAT_EQ(max_result.at(2), 3.0);
  }
  {
    const auto min_result =
        FullForwardSearch(plan_and_values, AggregationPolicy::Min);
    ASSERT_FLOAT_EQ(min_result.at(1), 1.5);
    ASSERT_FLOAT_EQ(min_result.at(2), 1.0);
  }
  {
    const auto mean_result =
        FullForwardSearch(plan_and_values, AggregationPolicy::Mean);
    ASSERT_FLOAT_EQ(mean_result.at(1), 1.75);
    ASSERT_FLOAT_EQ(mean_result.at(2), 2.0);
  }
}
}
