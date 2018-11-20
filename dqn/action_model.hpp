#ifndef ACTION_MODEL_HPP_
#define ACTION_MODEL_HPP_

#include <iostream>
#include "dqn.hpp"
#include "prettyprint.hpp"

namespace dqn {

template <size_t FrameCount, size_t FrameSize, size_t MinibatchSize,
          size_t OutputCount>
class ActionModel {
private:
  static constexpr auto kHiddenActivationCount = 256;
  static const auto kCroppedFrameDataSize = FrameSize * FrameSize;
  static const auto kInputDataSize = kCroppedFrameDataSize * FrameCount;
  static const auto kMinibatchDataSize = kInputDataSize * MinibatchSize;
  using TargetLayerInputData = std::array<float, MinibatchSize *OutputCount>;
  using InputLayerData =
      std::array<float, kHiddenActivationCount *MinibatchSize>;
  using FilterLayerData = InputLayerData;
  using TargetActivationLayerData = InputLayerData;
  using DQN = ::dqn::DQN<FrameCount, FrameSize, MinibatchSize, OutputCount>;

public:
  using FrameData = std::array<uint8_t, kCroppedFrameDataSize>;
  using FrameDataSp = std::shared_ptr<FrameData>;
  using InputFrames = std::array<FrameDataSp, FrameCount>;

public:
  using Transition = StateTransition<InputFrames, size_t>;
  ActionModel(const std::string &model_solver_param,
              const std::string &model_snapshot_prefix)
      : model_solver_param_(model_solver_param) {
    caffe::SolverParameter solver_param;
    caffe::ReadProtoFromTextFileOrDie(model_solver_param_, &solver_param);
    solver_param.set_snapshot_prefix(model_snapshot_prefix +
                                     solver_param.snapshot_prefix());
    model_solver_.reset(caffe::GetSolver<float>(solver_param));
    model_net_ = model_solver_->net();
    input_layer_ = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(
        model_net_->layer_by_name("input_layer"));
    target_activation_layer_ =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(
            model_net_->layer_by_name("target_activation_layer"));
    filter_layer_ = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(
        model_net_->layer_by_name("filter_layer"));
    target_reward_layer_ =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(
            model_net_->layer_by_name("target_reward_layer"));
    target_termination_layer_ =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(
            model_net_->layer_by_name("target_termination_layer"));
    // Initialize dummy input data with 0
    std::fill(dummy_input_data_.begin(), dummy_input_data_.end(), 0.0);
  }
  void UpdateModel(DQN &dqn, const std::vector<Transition> &transitions) {
    assert(transitions.size() == MinibatchSize);
    std::array<boost::optional<InputFrames>, MinibatchSize>
    current_frames_batch;
    std::array<boost::optional<InputFrames>, MinibatchSize> next_frames_batch;
    for (auto i = 0u; i < MinibatchSize; ++i) {
      current_frames_batch[i] = transitions[i].state;
      if (transitions[i].next_state_if_not_terminal) {
        next_frames_batch[i] = *transitions[i].next_state_if_not_terminal;
      }
    }
    const auto input_activations =
        dqn.GetHiddenActivations(current_frames_batch);
    auto target_activations = dqn.GetHiddenActivations(next_frames_batch);
    if (predicts_activation_difference_) {
      for (auto i = 0u; i < MinibatchSize; ++i) {
        if (target_activations[i]) {
          std::transform(
              target_activations[i]->begin(), target_activations[i]->end(),
              input_activations[i]->begin(), target_activations[i]->begin(),
              [](const float a, const float b) { return a - b; });
        }
      }
    }
    PourInputActivationData(input_activations);
    PourTargetActivationDataAndFilterData(target_activations);
    PourTargetRewardData(transitions);
    PourTargetTerminationData(transitions);
    model_solver_->Step(1);
    if (target_activations.front()) {
      const auto output_activation_blob =
          model_net_->blob_by_name("output_activation");
      std::cout << std::vector<float>(output_activation_blob->cpu_data(),
                                      output_activation_blob->cpu_data() + 10)
                << std::endl;
      std::cout << std::vector<float>(target_activations.front()->begin(),
                                      target_activations.front()->begin() + 10)
                << std::endl;
    }
  }

private:
  /**
   * 0: negative (-1)
   * 1: zero
   * 2: positive
   * @param reward
   * @return
   */
  size_t GetRewardClass(const double reward) {
    if (reward < 0) {
      return 0u;
    } else if (reward == 0) {
      return 1u;
    } else {
      return 2u;
    }
  }
  void PourTargetRewardData(const std::vector<Transition> &transitions) {
    static std::array<float, MinibatchSize> target_reward_data;
    for (auto i = 0u; i < MinibatchSize; ++i) {
      target_reward_data[i] = GetRewardClass(transitions[i].reward);
    }
    assert(target_reward_data.size() <= dummy_input_data_.size());
    target_reward_layer_->Reset(target_reward_data.data(),
                                dummy_input_data_.data(), MinibatchSize);
  }
  void PourTargetTerminationData(const std::vector<Transition> &transitions) {
    static std::array<float, MinibatchSize> target_termination_data;
    for (auto i = 0u; i < MinibatchSize; ++i) {
      target_termination_data[i] =
          transitions[i].next_state_if_not_terminal ? 0.0f : 1.0f;
    }
    assert(target_termination_data.size() <= dummy_input_data_.size());
    target_termination_layer_->Reset(target_termination_data.data(),
                                     dummy_input_data_.data(), MinibatchSize);
  }
  void PourInputActivationData(const std::array<
      boost::optional<std::vector<float> >, MinibatchSize> &input_activations) {
    static InputLayerData input_layer_data;
    for (auto i = 0u; i < MinibatchSize; ++i) {
      assert(input_activations[i]);
      std::copy(input_activations[i]->begin(), input_activations[i]->end(),
                input_layer_data.begin() + kHiddenActivationCount * i);
    }
    input_layer_->Reset(input_layer_data.data(), dummy_input_data_.data(),
                        MinibatchSize);
  }
  void PourTargetActivationDataAndFilterData(
      const std::array<boost::optional<std::vector<float> >, MinibatchSize> &
          target_activations) {
    static TargetActivationLayerData target_activation_layer_data;
    static FilterLayerData filter_layer_data;
    for (auto i = 0u; i < MinibatchSize; ++i) {
      if (target_activations[i]) {
        // Non-terminal state
        std::copy(target_activations[i]->begin(), target_activations[i]->end(),
                  target_activation_layer_data.begin() +
                      kHiddenActivationCount * i);
        std::fill(filter_layer_data.begin() + kHiddenActivationCount * i,
                  filter_layer_data.begin() + kHiddenActivationCount * (i + 1),
                  1.0f);
      } else {
        // terminal state
        std::fill(target_activation_layer_data.begin() +
                      kHiddenActivationCount * i,
                  target_activation_layer_data.begin() +
                      kHiddenActivationCount * (i + 1),
                  0.0f);
        std::fill(filter_layer_data.begin() + kHiddenActivationCount * i,
                  filter_layer_data.begin() + kHiddenActivationCount * (i + 1),
                  0.0f);
      }
    }
    target_activation_layer_->Reset(target_activation_layer_data.data(),
                                    dummy_input_data_.data(), MinibatchSize);
    filter_layer_->Reset(filter_layer_data.data(), dummy_input_data_.data(),
                         MinibatchSize);
  }

  std::string model_solver_param_;
  dqn::SolverSp model_solver_;
  dqn::NetSp model_net_;
  MemoryDataLayerSp input_layer_;
  MemoryDataLayerSp target_activation_layer_;
  MemoryDataLayerSp target_reward_layer_;
  MemoryDataLayerSp target_termination_layer_;
  MemoryDataLayerSp filter_layer_;
  TargetLayerInputData dummy_input_data_;
  const bool predicts_activation_difference_ = true;
};
}

#endif /* ACTION_MODEL_HPP_ */
