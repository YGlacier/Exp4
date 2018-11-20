#include "gtest/gtest.h"
#include "dqn.hpp"

DEFINE_double(dqn_lr, 0.2, "");

namespace dqn {

TEST(DQN, GetHiddenActivations) {
  using ALEDQN = DQN<4, 84, 32, 18>;
  ALEDQN dqn(std::vector<size_t>{ 0, 1, 2 }, "dqn_solver.prototxt", 0.95,
             false);
  dqn.Initialize("test");
  auto sample_frame = std::make_shared<ALEDQN::FrameData>();
  sample_frame->fill(1);
  ALEDQN::InputFrames sample_frames{ sample_frame, sample_frame, sample_frame,
                                     sample_frame };
  const auto tester = [&](const std::string &blob_name) {
    const auto hidden_activation =
        dqn.GetHiddenActivation(blob_name, sample_frames);
    const auto end_to_end_eval = dqn.EvaluateActions(sample_frames);
    const auto hidden_activation_eval =
        dqn.EvaluateActionsFromHiddenActivation(blob_name, hidden_activation);
    ASSERT_EQ(end_to_end_eval, hidden_activation_eval);
  };
  tester("ip1");
  tester("conv2");
}
}
