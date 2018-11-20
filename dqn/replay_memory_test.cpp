#include "gtest/gtest.h"
#include "replay_memory.hpp"

namespace dqn {

TEST(ReplayMemory, ReplayMemory) {
  auto replay_mem = ReplayMemory<int, int>(4);
  replay_mem.AddTransition({ 1, 1, 0, 2 });
  replay_mem.AddTransition({ 1, 2, 0, 3 });
  replay_mem.AddTransition({ 2, 1, 0, 3 });
  replay_mem.AddTransition({ 2, 2, 0, 4 });
  for (auto sample_count = 2u; sample_count < 32u; ++sample_count) {
    const auto action_and_samples =
        replay_mem.SampleTransitionsOfSameAction(sample_count);
    const auto &samples = action_and_samples.second;
    // Number of samples is correct
    ASSERT_EQ(samples.size(), sample_count);
    for (const auto &sample : samples) {
      // Actions are the same
      ASSERT_EQ(sample.action, action_and_samples.first);
    }
  }
  replay_mem.AddTransition({ 3, 1, 0, 4 });
}
TEST(ReplayMemory, SampleConsecutiveTransitionPairs) {
  auto replay_mem = ReplayMemory<int, int>(4);
  replay_mem.AddTransition({ 1, 1, 0, 2 });
  replay_mem.AddTransition({ 2, 1, 0, 3 });
  replay_mem.AddTransition({ 3, 1, 0, boost::none });
  replay_mem.AddTransition({ 1, 2, 0, 3 });
  replay_mem.AddTransition({ 3, 2, 0, 5 });
  replay_mem.AddTransition({ 5, 2, 0, boost::none });
  for (auto sample_count = 2u; sample_count < 32u; ++sample_count) {
    const auto pairs =
        replay_mem.SampleConsecutiveTransitionPairs(sample_count);
    // Number of samples is correct
    ASSERT_EQ(pairs.size(), sample_count);
    for (const auto &p : pairs) {
      ASSERT_TRUE(p.first.next_state_if_not_terminal);
      ASSERT_EQ(p.first.next_state_if_not_terminal.get(), p.second.state);
    }
  }
}
}
