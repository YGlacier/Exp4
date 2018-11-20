#ifndef DQN_HPP_
#define DQN_HPP_

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <caffe/caffe.hpp>
#include <boost/functional/hash.hpp>
#include <boost/optional.hpp>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <glog/logging.h>
#include "prettyprint.hpp"
#include "replay_memory.hpp"
#include "util/container.hpp"
#include "util/random.hpp"

DECLARE_bool(after_relu);
DECLARE_double(dqn_lr);

namespace dqn {

using SolverSp = std::shared_ptr<caffe::Solver<float> >;
using NetSp = boost::shared_ptr<caffe::Net<float> >;
using BlobSp = boost::shared_ptr<caffe::Blob<float> >;
using MemoryDataLayerSp = boost::shared_ptr<caffe::MemoryDataLayer<float> >;

std::string PrintQValues(const std::vector<float> &q_values,
                         const std::vector<size_t> &actions) {
  assert(!q_values.empty());
  assert(!actions.empty());
  assert(q_values.size() == actions.size());
  std::ostringstream actions_buf;
  std::ostringstream q_values_buf;
  for (auto i = 0; i < q_values.size(); ++i) {
    const auto a_str = std::to_string(actions.at(i));
    const auto q_str = std::to_string(q_values.at(i));
    const auto column_size = std::max(a_str.size(), q_str.size()) + 1;
    actions_buf.width(column_size);
    actions_buf << a_str;
    q_values_buf.width(column_size);
    q_values_buf << q_str;
  }
  actions_buf << std::endl;
  q_values_buf << std::endl;
  return actions_buf.str() + q_values_buf.str();
}

template <typename Dtype>
bool HasBlobSize(const caffe::Blob<Dtype> &blob, const size_t num,
                 const size_t channels, const size_t height,
                 const size_t width) {
  return blob.num() == num && blob.channels() == channels &&
         blob.height() == height && blob.width() == width;
}

template <typename Dtype>
bool HasBlobSize(const caffe::Blob<Dtype> &blob, const size_t num,
                 const size_t size) {
  return blob.num() == num && blob.count() == size * num;
}

/**
 * Deep Q-Network
 */
template <size_t FrameCount, size_t FrameSize, size_t MinibatchSize,
          size_t OutputCount>
class DQN {
private:
  static const auto kCroppedFrameDataSize = FrameSize * FrameSize;
  static const auto kInputDataSize = kCroppedFrameDataSize * FrameCount;
  static const auto kMinibatchDataSize = kInputDataSize * MinibatchSize;
  using TargetLayerInputData = std::array<float, MinibatchSize *OutputCount>;
  using FramesLayerInputData = std::array<float, kMinibatchDataSize>;
  using FilterLayerInputData = std::array<float, MinibatchSize *OutputCount>;

public:
  using FrameData = std::array<uint8_t, kCroppedFrameDataSize>;
  using FrameDataSp = std::shared_ptr<FrameData>;
  using InputFrames = std::array<FrameDataSp, FrameCount>;
  using Activation = std::vector<float>;
  using Transition = StateTransition<InputFrames, size_t>;
  template <typename T> using BatchArray = std::array<T, MinibatchSize>;
  template <typename T> using BatchOpArray = BatchArray<boost::optional<T> >;

public:
  DQN(const std::vector<size_t> &legal_actions, const std::string &solver_param,
      const double gamma, const bool after_relu)
      : legal_actions_(legal_actions), solver_param_(solver_param),
        gamma_(gamma), after_relu_(after_relu), current_iteration_(0),
        max_iteration_(0) {}

  /**
   * Initialize DQN. Must be called before calling any other method.
   */
  void Initialize(const std::string &model_snapshot_prefix) {
    // Initialize net and solver
    caffe::SolverParameter solver_param;
    caffe::ReadProtoFromTextFileOrDie(solver_param_, &solver_param);
    solver_param.set_snapshot_prefix(model_snapshot_prefix +
                                     solver_param.snapshot_prefix());
    solver_param.set_base_lr(FLAGS_dqn_lr);
    max_iteration_ = solver_param.max_iter();
    solver_.reset(caffe::GetSolver<float>(solver_param));
    net_ = solver_->net();

    // Cache pointers to blobs that hold Q values
    q_values_blob_ = net_->blob_by_name("q_values");

    // Initialize dummy input data with 0
    std::fill(dummy_input_data_.begin(), dummy_input_data_.end(), 0.0);

    // Cache pointers to input layers
    frames_input_layer_ =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(
            net_->layer_by_name("frames_input_layer"));
    assert(frames_input_layer_);
    assert(HasBlobSize(*net_->blob_by_name("frames"), MinibatchSize, FrameCount,
                       FrameSize, FrameSize));
    target_input_layer_ =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(
            net_->layer_by_name("target_input_layer"));
    assert(target_input_layer_);
    assert(HasBlobSize(*net_->blob_by_name("target"), MinibatchSize,
                       OutputCount, 1, 1));
    filter_input_layer_ =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(
            net_->layer_by_name("filter_input_layer"));
    assert(filter_input_layer_);
    assert(HasBlobSize(*net_->blob_by_name("filter"), MinibatchSize,
                       OutputCount, 1, 1));
  }

  /**
   * Load a trained model from a file.
   */
  void LoadTrainedModel(const std::string &model_bin) {
    net_->CopyTrainedLayersFrom(model_bin);
    // Make weights and biases for unused outputs all zeros
    const auto weight_begin =
        net_->layer_by_name("ip2_layer")->blobs().front()->mutable_cpu_data();
    const auto bias_begin =
        net_->layer_by_name("ip2_layer")->blobs().at(1)->mutable_cpu_data();
    const std::unordered_set<size_t> legal_action_set(legal_actions_.begin(),
                                                      legal_actions_.end());
    for (size_t a = 0; a < OutputCount; ++a) {
      if (!legal_action_set.count(a)) {
        std::fill(weight_begin + 256 * a, weight_begin + 256 * (a + 1), 0.0);
        std::fill(bias_begin + a, bias_begin + a + 1, 0.0);
      }
    }
  }

  /**
   * Select an action by epsilon-greedy.
   */
  size_t SelectAction(const InputFrames &input_data, const double epsilon) {
    assert(epsilon >= 0.0 && epsilon <= 1.0);
    if (util::random::RandomDouble(0.0, 1.0) < epsilon) {
      return util::random::RandomSelect(legal_actions_);
    } else {
      return SelectActionGreedily(input_data).first;
    }
  }

  // Tuple of input, action and target Q-value
  using TrainingSample = std::tuple<InputFrames, size_t, float>;

  void Update(const std::array<TrainingSample, MinibatchSize> &samples) {
    std::cout << "DQN iteration: " << current_iteration_ << std::endl;
    FramesLayerInputData frames_input;
    TargetLayerInputData target_input;
    FilterLayerInputData filter_input;
    std::fill(target_input.begin(), target_input.end(), 0.0f);
    std::fill(filter_input.begin(), filter_input.end(), 0.0f);
    for (auto i = 0; i < MinibatchSize; ++i) {
      const auto action = std::get<1>(samples.at(i));
      assert(action >= 0 && action < OutputCount);
      const auto target_q = std::get<2>(samples.at(i));
      assert(!std::isnan(target_q));
      target_input.at(i *OutputCount + action) = target_q;
      filter_input.at(i *OutputCount + action) = 1;
      VLOG(1) << "filter:" << action << " target:" << target_q;
      for (auto j = 0; j < FrameCount; ++j) {
        const auto &frame_data = std::get<0>(samples.at(i)).at(j);
        std::copy(frame_data->begin(), frame_data->end(),
                  frames_input.begin() + i * kInputDataSize +
                      j * kCroppedFrameDataSize);
      }
    }
    InputDataIntoLayers(frames_input, target_input, filter_input);
    solver_->Step(1);
    // Log the first parameter of each hidden layer
    VLOG(1) << "conv1:"
            << net_->layer_by_name("conv1_layer")->blobs().front()->data_at(
                   1, 0, 0, 0);
    VLOG(1) << "conv2:"
            << net_->layer_by_name("conv2_layer")->blobs().front()->data_at(
                   1, 0, 0, 0);
    VLOG(1) << "ip1:"
            << net_->layer_by_name("ip1_layer")->blobs().front()->data_at(1, 0,
                                                                          0, 0);
    VLOG(1) << "ip2:"
            << net_->layer_by_name("ip2_layer")->blobs().front()->data_at(1, 0,
                                                                          0, 0);
    ++current_iteration_;
    assert(solver_->iter() == current_iteration_);
  }

  std::array<TrainingSample, MinibatchSize> GenerateTrainingSamples(
      const std::array<Transition, MinibatchSize> &transitions) {
    // Array of s'
    std::array<boost::optional<InputFrames>, MinibatchSize>
    target_last_frames_batch;
    std::transform(transitions.begin(), transitions.end(),
                   target_last_frames_batch.begin(),
                   [](const Transition &transition) {
      return transition.next_state_if_not_terminal;
    });
    // Compute Q(s',a)
    const auto actions_and_values =
        SelectActionGreedily(target_last_frames_batch);
    // Compute max_a Q(s',a)
    std::array<TrainingSample, MinibatchSize> samples;
    std::transform(transitions.begin(), transitions.end(),
                   actions_and_values.begin(), samples.begin(),
                   [&](const Transition &transition,
                       const boost::optional<std::pair<size_t, float> > &
                           action_and_value) {
      const auto action = transition.action;
      assert(static_cast<int>(action) < OutputCount);
      const auto reward = transition.reward;
      assert(reward >= -1.0 && reward <= 1.0);
      const auto target = transition.next_state_if_not_terminal
                              ? reward + gamma_ * action_and_value->second
                              : reward;
      assert(!std::isnan(target));
      return TrainingSample{ transition.state, action, target };
    });
    return samples;
  }

  /**
   * Update DQN using one minibatch
   */
  void Update(
      const std::vector<StateTransition<InputFrames, size_t> > &transitions) {
    Update(GenerateTrainingSamples(
        util::container::VectorToArray<Transition, MinibatchSize>(
            transitions)));
  }

  std::vector<float> GetHiddenActivation(const std::string &blob_name,
                                         const InputFrames &last_frames) {
    std::array<boost::optional<InputFrames>, MinibatchSize> last_frames_batch;
    last_frames_batch.front() = last_frames;
    const auto &batch_result =
        GetHiddenActivations(blob_name, last_frames_batch);
    assert(!batch_result.empty());
    assert(batch_result.front());
    return *batch_result.front();
  }

  std::vector<Activation>
  GetHiddenActivations(const std::string &blob_name,
                       std::vector<InputFrames> input_frames_batch) {
    return util::container::OptionalArrayToVector(GetHiddenActivations(
        blob_name, util::container::VectorToOptionalArray<MinibatchSize>(
                       input_frames_batch)));
  }

  std::array<boost::optional<std::vector<float> >, MinibatchSize>
  GetHiddenActivations(const std::string &blob_name,
                       const BatchOpArray<InputFrames> &last_frames_batch) {
    PourInputFramesIntoLayers(last_frames_batch);
    ForwardToLayerRightBeforeBlob(blob_name);
    std::array<boost::optional<std::vector<float> >, MinibatchSize> results;
    const auto hidden_activation_blob = net_->blob_by_name(blob_name);
    const auto hidden_activation_count =
        hidden_activation_blob->count() / hidden_activation_blob->num();
    assert(HasBlobSize(*hidden_activation_blob, MinibatchSize,
                       hidden_activation_count));
    for (auto i = 0u; i < MinibatchSize; ++i) {
      if (last_frames_batch.at(i)) {
        results.at(i) = std::vector<float>(
            hidden_activation_blob->cpu_data() + hidden_activation_count * i,
            hidden_activation_blob->cpu_data() +
                hidden_activation_count * (i + 1));
      }
    }
    return results;
  }

  std::unordered_map<size_t, float> EvaluateActionsFromHiddenActivation(
      const std::string &blob_name,
      const std::vector<float> &hidden_activation) {
    return EvaluateActionsFromHiddenActivation(
               blob_name, std::vector<std::vector<float> >{ hidden_activation })
        .front();
  }

  void ForwardToLayerRightBeforeBlob(const std::string &blob_name) {
    static std::unordered_map<std::string, std::string>
    blob_name_to_layer_right_before_it = { { "ip1", "ip1_layer" },
                                           { "conv2", "conv2_layer" } };
    assert(blob_name_to_layer_right_before_it.count(blob_name));
    const auto layer_name_it =
        std::find(net_->layer_names().begin(), net_->layer_names().end(),
                  blob_name_to_layer_right_before_it.at(blob_name));
    assert(layer_name_it != net_->layer_names().end());
    const auto layer_idx =
        std::distance(net_->layer_names().begin(), layer_name_it) +
        (after_relu_ ? 1 : 0);
    net_->ForwardTo(layer_idx);
  }

  void ForwardFromLayerRightAfterBlob(const std::string &blob_name) {
    static std::unordered_map<std::string, std::string>
    blob_name_to_layer_right_after_it = { { "ip1", "ip1_relu_layer" },
                                          { "conv2", "conv2_relu_layer" } };
    assert(blob_name_to_layer_right_after_it.count(blob_name));
    const auto layer_name_it =
        std::find(net_->layer_names().begin(), net_->layer_names().end(),
                  blob_name_to_layer_right_after_it.at(blob_name));
    assert(layer_name_it != net_->layer_names().end());
    const auto layer_idx =
        std::distance(net_->layer_names().begin(), layer_name_it) +
        (after_relu_ ? 1 : 0);
    net_->ForwardFrom(layer_idx);
  }

  std::array<boost::optional<std::unordered_map<size_t, float> >, MinibatchSize>
  EvaluateActionsFromHiddenActivation(
      const std::string &blob_name,
      const std::array<boost::optional<Activation>, MinibatchSize> &
          hidden_activation_batch) {
    auto hidden_activation_blob = net_->blob_by_name(blob_name);
    auto dest = hidden_activation_blob->data()->mutable_cpu_data();
    for (auto i = 0u; i < hidden_activation_batch.size(); ++i) {
      if (hidden_activation_batch.at(i)) {
        const auto &hidden_activation = *hidden_activation_batch.at(i);
        std::copy(hidden_activation.begin(), hidden_activation.end(),
                  static_cast<float *>(dest) + hidden_activation.size() * i);
      }
    }
    ForwardFromLayerRightAfterBlob(blob_name);
    return EvaluateActionsAfterForwarding(
        util::container::OptionalArrayToBoolArray(hidden_activation_batch));
  }

  std::vector<std::unordered_map<size_t, float> >
  EvaluateActionsFromHiddenActivation(
      const std::string &blob_name_,
      const std::vector<std::vector<float> > &hidden_activation_batch) {
    return util::container::OptionalArrayToVector(
        EvaluateActionsFromHiddenActivation(
            blob_name_, util::container::VectorToOptionalArray<MinibatchSize>(
                            hidden_activation_batch)));
  }

  std::unordered_map<size_t, float>
  EvaluateActions(const InputFrames &last_frames) {
    return EvaluateActions(std::vector<InputFrames>{ last_frames }).front();
  }

  BatchOpArray<std::unordered_map<size_t, float> >
  EvaluateActions(const BatchOpArray<InputFrames> &last_frames_batch) {
    ForwardFrames(last_frames_batch);
    return EvaluateActionsAfterForwarding(
        util::container::OptionalArrayToBoolArray(last_frames_batch));
  }

  std::vector<std::unordered_map<size_t, float> >
  EvaluateActions(const std::vector<InputFrames> &last_frames_batch) {
    return util::container::OptionalArrayToVector(
        EvaluateActions(util::container::VectorToOptionalArray<MinibatchSize>(
            last_frames_batch)));
  }

  std::pair<size_t, float> SelectActionGreedily(const InputFrames &input_data) {
    return SelectActionGreedily(std::vector<InputFrames>{ { input_data } })
        .front();
  }

  size_t current_iteration() const {
    // Solver::iter is undefined until Solver::Presolve() is called.
    // So this class holds current iteration itself.
    return current_iteration_;
  }

  size_t max_iteration() const { return max_iteration_; }
  const NetSp& net() const { return net_; }

private:
  void PourInputFramesIntoLayers(
      const BatchOpArray<InputFrames> &input_frames_batch) {
    static std::array<float, kMinibatchDataSize> frames_input;
    for (auto i = 0u; i < MinibatchSize; ++i) {
      if (input_frames_batch.at(i)) {
        for (auto j = 0; j < FrameCount; ++j) {
          const auto &frame_data = input_frames_batch.at(i)->at(j);
          std::copy(frame_data->begin(), frame_data->end(),
                    frames_input.begin() + i * kInputDataSize +
                        j * kCroppedFrameDataSize);
        }
      }
    }
    InputDataIntoLayers(frames_input, dummy_input_data_, dummy_input_data_);
  }

  void ForwardFrames(const std::array<boost::optional<InputFrames>,
                                      MinibatchSize> &last_frames_batch) {
    PourInputFramesIntoLayers(last_frames_batch);
    net_->ForwardPrefilled(nullptr);
  }

  //  void ForwardFrames(const std::vector<InputFrames> &last_frames_batch) {
  //    assert(!last_frames_batch.empty());
  //    std::array<boost::optional<InputFrames>, MinibatchSize> tmp;
  //    for (auto i = 0u; i < last_frames_batch.size(); ++i) {
  //      tmp.at(i) = last_frames_batch.at(i);
  //    }
  //    ForwardFrames(tmp);
  //  }

  std::vector<std::pair<size_t, float> >
  SelectActionGreedily(const std::vector<InputFrames> &last_frames_batch) {
    return util::container::OptionalArrayToVector(SelectActionGreedily(
        util::container::VectorToOptionalArray<MinibatchSize>(
            last_frames_batch)));
  }

  BatchOpArray<std::pair<size_t, float> >
  SelectActionGreedily(const BatchOpArray<InputFrames> &last_frames_batch) {
    const auto evaluations = EvaluateActions(last_frames_batch);
    BatchOpArray<std::pair<size_t, float> > results;

    std::transform(
        evaluations.begin(), evaluations.end(), results.begin(),
        util::container::Optionalize<std::unordered_map<size_t, float>,
                                     std::pair<size_t, float> >([](
            const std::unordered_map<size_t, float> &eval) {
          return *std::max_element(
                     eval.begin(), eval.end(),
                     util::container::SecondComparator<size_t, float>);
        }));
    return results;
  }

  void InputDataIntoLayers(const FramesLayerInputData &frames_input,
                           const TargetLayerInputData &target_input,
                           const FilterLayerInputData &filter_input) {
    assert(frames_input_layer_);
    assert(target_input_layer_);
    assert(filter_input_layer_);
    frames_input_layer_->Reset(const_cast<float *>(frames_input.data()),
                               dummy_input_data_.data(), MinibatchSize);
    target_input_layer_->Reset(const_cast<float *>(target_input.data()),
                               dummy_input_data_.data(), MinibatchSize);
    filter_input_layer_->Reset(const_cast<float *>(filter_input.data()),
                               dummy_input_data_.data(), MinibatchSize);
  }

  std::array<boost::optional<std::unordered_map<size_t, float> >, MinibatchSize>
  EvaluateActionsAfterForwarding(
      const std::array<bool, MinibatchSize> &evaluate_or_not) {
    std::array<boost::optional<std::unordered_map<size_t, float> >,
               MinibatchSize> results;
    for (auto i = 0u; i < MinibatchSize; ++i) {
      if (evaluate_or_not.at(i)) {
        // Get the Q values from the net
        const auto action_evaluator = [&](size_t action) {
          const auto q =
              q_values_blob_->data_at(i, static_cast<int>(action), 0, 0);
          assert(!std::isnan(q));
          return std::make_pair(action, q);
        };
        std::unordered_map<size_t, float> q_values;
        std::transform(legal_actions_.begin(), legal_actions_.end(),
                       std::inserter(q_values, q_values.end()),
                       action_evaluator);
        results.at(i) = q_values;
      }
    }
    return results;
  }

  std::vector<std::unordered_map<size_t, float> >
  EvaluateActionsAfterForwarding(const size_t batch_size) {
    std::array<bool, MinibatchSize> evaluate_or_not;
    std::fill(evaluate_or_not.begin(), evaluate_or_not.begin() + batch_size,
              true);
    std::fill(evaluate_or_not.begin() + batch_size, evaluate_or_not.end(),
              false);
    const auto v = util::container::OptionalArrayToVector(
        EvaluateActionsAfterForwarding(evaluate_or_not));
    assert(v.size() == batch_size);
    return v;
  }

  const std::vector<size_t> legal_actions_;
  const std::string solver_param_;
  const double gamma_;
  SolverSp solver_;
  NetSp net_;
  BlobSp q_values_blob_;
  MemoryDataLayerSp frames_input_layer_;
  MemoryDataLayerSp target_input_layer_;
  MemoryDataLayerSp filter_input_layer_;
  TargetLayerInputData dummy_input_data_;
  size_t current_iteration_;
  size_t max_iteration_;
  const bool after_relu_;
};
}

#endif /* DQN_HPP_ */
