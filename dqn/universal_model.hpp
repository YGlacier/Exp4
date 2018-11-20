#ifndef UNIVERSAL_MODEL_HPP_
#define UNIVERSAL_MODEL_HPP_

#include <iostream>
#include <boost/range/combine.hpp>
#include <boost/range/join.hpp>
#include <boost/range/adaptors.hpp>
#include "dqn.hpp"
#include "prettyprint.hpp"

namespace dqn {

namespace A = boost::adaptors;

inline std::vector<float>
GetRelativeEffect(const std::vector<float> &current_activation,
                  const std::vector<float> &next_activation) {
  assert(current_activation.size() == next_activation.size());
  return util::container::AsVector(
      boost::combine(current_activation, next_activation) |
      A::transformed([](const auto &t) {
        return static_cast<float>(t.get<1>() - t.get<0>());
      }));
}

inline boost::optional<std::vector<float> >
GetRelativeEffect(const std::vector<float> &current_activation,
                  const boost::optional<std::vector<float> > &next_activation) {
  return next_activation ? boost::make_optional(GetRelativeEffect(
                               current_activation, next_activation.get()))
                         : boost::none;
}

template <size_t FrameCount, size_t FrameSize, size_t MinibatchSize,
          size_t OutputCount>
class UniversalModel {
private:
  static const auto kCroppedFrameDataSize = FrameSize * FrameSize;
  static constexpr auto kRewardClassCount = 3;
  static constexpr auto kTerminationClassCount = 2;
  using DQN = ::dqn::DQN<FrameCount, FrameSize, MinibatchSize, OutputCount>;
  template <typename T> using BatchArray = std::array<T, MinibatchSize>;
  template <typename T>
  using BatchOpArray = std::array<boost::optional<T>, MinibatchSize>;

public:
  using FrameData = std::array<uint8_t, kCroppedFrameDataSize>;
  using FrameDataSp = std::shared_ptr<FrameData>;
  using InputFrames = std::array<FrameDataSp, FrameCount>;
  using Activation = std::vector<float>;
  using ActivationAndAction = std::pair<Activation, size_t>;
  using RewardAndNextActivation =
      std::pair<double, boost::optional<Activation> >;

public:
  using Transition = StateTransition<InputFrames, size_t>;
  UniversalModel(const DQN &dqn, const std::string &activation_blob_name,
                 const std::string &solver_proto_file,
                 const std::string &model_snapshot_prefix)
      : model_solver_param_(solver_proto_file),
        activation_blob_name_(activation_blob_name) {
    caffe::SolverParameter solver_param;
    caffe::ReadProtoFromTextFileOrDie(model_solver_param_, &solver_param);
    solver_param.set_snapshot_prefix(model_snapshot_prefix +
                                     solver_param.snapshot_prefix());
    model_solver_.reset(caffe::GetSolver<float>(solver_param));
    model_net_ = model_solver_->net();
    input_layer_ = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(
        model_net_->layer_by_name("input_layer"));
    filter_layer_ = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(
        model_net_->layer_by_name("filter_layer"));
    target_activation_layer_ =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(
            model_net_->layer_by_name("target_activation_layer"));
    target_reward_layer_ =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(
            model_net_->layer_by_name("target_reward_layer"));
    target_termination_layer_ =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(
            model_net_->layer_by_name("target_termination_layer"));
    output_activation_blob_ = model_net_->blob_by_name("output_activation");
    output_reward_blob_ = model_net_->blob_by_name("output_reward");
    output_termination_blob_ = model_net_->blob_by_name("output_termination");
    assert(output_activation_blob_);
    assert(output_reward_blob_);
    assert(output_termination_blob_);
    // blob size
    activation_data_size_ =
        output_activation_blob_->count() / output_activation_blob_->num();
    // Initialize dummy input data with 0
    dummy_input_data_.assign(activation_data_size_ * MinibatchSize, 0.0f);
    if (model_net_->has_layer("q_loss_layer")) {
      std::cout << "q_loss_layer found" << std::endl;
      // Copy weights from DQN
      const auto &source_blobs = dqn.net()->layer_by_name("ip2_layer")->blobs();
      const auto &weight_blob = *source_blobs.front();
      const auto &bias_blob = *source_blobs.at(1);
      // weights
      model_net_->layer_by_name("output_q_layer")->blobs().front()->CopyFrom(
          weight_blob);
      model_net_->layer_by_name("target_q_layer")->blobs().front()->CopyFrom(
          weight_blob);
      // biases
      model_net_->layer_by_name("output_q_layer")->blobs().at(1)->CopyFrom(
          bias_blob);
      model_net_->layer_by_name("target_q_layer")->blobs().at(1)->CopyFrom(
          bias_blob);
    }
  }

  /**
   * Load a trained model from a file.
   */
  void LoadTrainedModel(const std::string &model_bin) {
    model_net_->CopyTrainedLayersFrom(model_bin);
  }

  using TrainingSample =
      std::pair<ActivationAndAction, RewardAndNextActivation>;

  void UpdateModel(const BatchArray<TrainingSample> &sample_batch) {
    // [?(activation,action)]
    const auto input_batch = util::container::AsArray<MinibatchSize>(
        sample_batch | A::transformed([](const auto &p) {
                         return boost::make_optional(p.first);
                       }));
    PourInputData(input_batch);
    // [(reward,?activation)]
    const auto target_batch =
        util::container::AsArray<MinibatchSize>(sample_batch | A::map_values);
    PourTargetData(target_batch);
    model_solver_->Step(1);
  }

  void UpdateModel(
      DQN &dqn, const std::vector<Transition> &single_transitions,
      const std::vector<std::pair<Transition, Transition> > &transition_pairs,
      const bool log) {
    assert(single_transitions.size() + transition_pairs.size() ==
           MinibatchSize);
    // A:[(s_t,a_t,r_t,s_t+1)] -> [h_t]
    // B:[((s_t-1,a_t-1,r_t-1,s_t),(s_t,a_t,r_t,s_t+1)] -> [h_t-1]
    const auto real_input_activations_and_prev_next_activations =
        dqn.GetHiddenActivations(
            activation_blob_name_,
            util::container::AsVector(
                boost::range::join(single_transitions,
                                   transition_pairs | A::map_keys) |
                A::transformed([](const auto &t) { return t.state; })));
    // A:[h_t]
    const auto real_input_activations =
        real_input_activations_and_prev_next_activations |
        A::sliced(static_cast<size_t>(0), single_transitions.size());
    // B:[h_t-1]
    const auto prev_next_activations =
        real_input_activations_and_prev_next_activations |
        A::sliced(single_transitions.size(), MinibatchSize);
    // B:[h_t-1], [a_t-1] -> [(h_t-1,a_t-1)]
    const auto prev_activation_action_pairs = util::container::AsVector(
        boost::combine(
            prev_next_activations,
            transition_pairs | A::map_keys |
                A::transformed([](const auto &t) { return t.action; })) |
        A::transformed([](const auto &t) {
          return std::make_pair(t.get<0>(), t.get<1>());
        }));
    // B:[(h_t-1,a_t-1)] -> [h_t]
    const auto hallucinated_input_activations =
        transition_pairs.empty()
            ? std::vector<Activation>()
            : util::container::AsVector(
                  PredictNextActivations(prev_activation_action_pairs, true) |
                  A::transformed([](const auto &p) { return p.second.get(); }));
    assert(real_input_activations.size() +
               hallucinated_input_activations.size() ==
           MinibatchSize);
    const auto input_activations = util::container::AsVector(boost::range::join(
        real_input_activations, hallucinated_input_activations));
    const auto transitions = util::container::AsVector(boost::range::join(
        single_transitions, transition_pairs | A::map_values));
    const auto next_frames_batch = util::container::AsArray<MinibatchSize>(
        transitions | A::transformed([](const auto &t) {
                        return t.next_state_if_not_terminal;
                      }));
    const auto raw_target_activations =
        dqn.GetHiddenActivations(activation_blob_name_, next_frames_batch);
    // [?h_t+1]
    const auto &target_activations =
        predicts_activation_difference_
            ? util::container::AsArray<MinibatchSize>(
                  boost::combine(input_activations, raw_target_activations) |
                  A::transformed([](const auto &t) {
                    return GetRelativeEffect(t.get<0>(), t.get<1>());
                  }))
            : raw_target_activations;
    const auto sample_batch = util::container::AsArray<MinibatchSize>(
        boost::combine(transitions,
                       boost::combine(input_activations, target_activations)) |
        A::transformed([](const auto &t) {
          const auto &transition = t.get<0>();
          const auto &input_activation = t.get<1>().get<0>();
          const auto &target_activation_or_none = t.get<1>().get<1>();
          return std::make_pair(
              std::make_pair(input_activation, transition.action),
              std::make_pair(transition.reward, target_activation_or_none));
        }));
    UpdateModel(sample_batch);
    if (log) {
      const auto output_reward_blob = model_net_->blob_by_name("output_reward");
      const auto output_reward = std::vector<double>(
          output_reward_blob->cpu_data(), output_reward_blob->cpu_data() + 3);
      const auto output_termination_blob =
          model_net_->blob_by_name("output_termination");
      const auto output_termination =
          std::vector<double>(output_termination_blob->cpu_data(),
                              output_termination_blob->cpu_data() + 2);
      std::cout << "reward:" << transitions.front().reward << output_reward
                << std::endl;
      std::cout << "termination:"
                << !transitions.front().next_state_if_not_terminal
                << output_termination << std::endl;
      if (target_activations.front()) {
        const auto &input = input_activations.front();
        const auto raw_output = std::vector<float>(
            output_activation_blob_->cpu_data(),
            output_activation_blob_->cpu_data() + activation_data_size_);
        const auto &output =
            predicts_activation_difference_
                ? GetNextActivationFromRelativeEffect(raw_output, input)
                : raw_output;
        const auto &target = predicts_activation_difference_
                                 ? GetNextActivationFromRelativeEffect(
                                       target_activations.front().get(), input)
                                 : target_activations.front().get();
        const auto sample_size = 10;
        assert(sample_size <= activation_data_size_);
        std::cout << "input_activation:"
                  << std::vector<float>(input.begin(),
                                        input.begin() + sample_size)
                  << std::endl;
        std::cout << "output_activation:"
                  << std::vector<float>(output.begin(),
                                        output.begin() + sample_size)
                  << std::endl;
        std::cout << "target_activation:"
                  << std::vector<float>(target.begin(),
                                        target.begin() + sample_size)
                  << std::endl;
        const auto input_eval = dqn.EvaluateActions(transitions.front().state);
        const auto output_eval = dqn.EvaluateActionsFromHiddenActivation(
            activation_blob_name_, output);
        const auto target_eval = dqn.EvaluateActionsFromHiddenActivation(
            activation_blob_name_, target);
        //      const auto target2_eval =
        // dqn.EvaluateActions(*transitions.front().next_state_if_not_terminal);
        std::cout << "input_eval:" << input_eval << std::endl;
        std::cout << "output_eval:" << output_eval << std::endl;
        std::cout << "target_eval:" << target_eval << std::endl;
        //      std::cout << target2_eval << std::endl;
      }
      //    std::cout << model_net_->blob_by_name("filtered_output_activation")
      //                     ->data_at(0, 0, 0, 0) << std::endl;
      //    std::cout << model_net_->blob_by_name("output_reward")->data_at(0,
      // 0,
      // 0, 0)
      //              << std::endl;
      //    std::cout << model_net_->blob_by_name("output_termination")
      //                     ->data_at(0, 0, 0, 0) << std::endl;
      //    const auto target_activation =
      //        model_net_->blob_by_name("target_activation");
      //    //    std::cout << std::vector<float>(target_activation->cpu_data(),
      //    // target_activation->cpu_data() + 256) << std::endl;
      //    std::cout << target_activation->data_at(0, 0, 0, 0) << std::endl;
      //    std::cout << model_net_->blob_by_name("target_reward")->data_at(0,
      // 0,
      // 0, 0)
      //              << std::endl;
      //    std::cout << model_net_->blob_by_name("target_termination")
      //                     ->data_at(0, 0, 0, 0) << std::endl;
    }
  }

  void UpdateModel(DQN &dqn, const std::vector<Transition> &transitions,
                   const bool log) {
    UpdateModel(dqn, transitions,
                std::vector<std::pair<Transition, Transition> >(), log);
  }

  Activation GetNextActivationFromRelativeEffect(
      const Activation &relative_effect,
      const Activation &current_activation) const {
    assert(relative_effect.size() == current_activation.size());
    return util::container::AsVector(
        boost::combine(relative_effect, current_activation) |
        boost::adaptors::transformed([](const auto &t) {
          return t.get<0>() + t.get<1>();
        }));
  }

  BatchOpArray<RewardAndNextActivation>
  PredictNextActivations(const BatchOpArray<std::pair<Activation, size_t> > &
                             activation_action_pairs,
                         const bool force_to_predict_next_activation = false) {
    PourInputData(activation_action_pairs);
    PourDummyTargetActivationDataAndFilterData();
    PourDummyTargetRewardDataAndTargetTerminationData();
    model_net_->ForwardPrefilled(nullptr);
    BatchOpArray<RewardAndNextActivation> next_activations;
    for (auto i = 0; i < activation_action_pairs.size(); ++i) {
      if (activation_action_pairs.at(i)) {
        const auto terminal = GetOutputTerminal(i);
        const auto reward = GetOutputReward(i);
        const auto next_activation_or_none =
            terminal && !force_to_predict_next_activation
                ? boost::none
                : boost::make_optional(
                      predicts_activation_difference_
                          ? GetNextActivationFromRelativeEffect(
                                GetOutputActivation(i),
                                activation_action_pairs.at(i)->first)
                          : GetOutputActivation(i));
        next_activations.at(i) =
            std::make_pair(reward, next_activation_or_none);
      }
    }
    return next_activations;
  }
  std::vector<RewardAndNextActivation> PredictNextActivations(
      const std::vector<ActivationAndAction> &activation_action_pairs,
      const bool force_to_predict_next_activation = false) {
    assert(activation_action_pairs.size() > 0 &&
           activation_action_pairs.size() <= MinibatchSize);
    return util::container::OptionalArrayToVector(PredictNextActivations(
        util::container::VectorToOptionalArray<MinibatchSize>(
            activation_action_pairs),
        force_to_predict_next_activation));
  }
  std::vector<RewardAndNextActivation>
  PredictNextActivations(const std::vector<float> &activation,
                         const std::vector<size_t> &action_indices) {
    assert(activation.size() == activation_data_size_);
    assert(action_indices.size() > 0 && action_indices.size() <= MinibatchSize);
    std::vector<std::pair<std::vector<float>, size_t> > activation_action_pairs;
    activation_action_pairs.reserve(action_indices.size());
    std::transform(action_indices.begin(), action_indices.end(),
                   std::back_inserter(activation_action_pairs),
                   [&](const size_t action_idx) {
      return std::make_pair(activation, action_idx);
    });
    return PredictNextActivations(activation_action_pairs);
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

  void PourDummyTargetRewardDataAndTargetTerminationData() {
    target_reward_layer_->Reset(dummy_input_data_.data(),
                                dummy_input_data_.data(), MinibatchSize);
    target_termination_layer_->Reset(dummy_input_data_.data(),
                                     dummy_input_data_.data(), MinibatchSize);
  }

  void PourTargetRewardData(const BatchArray<double> &reward_batch) {
    static std::array<float, MinibatchSize> target_reward_data;
    for (auto i = 0u; i < MinibatchSize; ++i) {
      target_reward_data.at(i) = GetRewardClass(reward_batch.at(i));
    }
    assert(target_reward_data.size() <= dummy_input_data_.size());
    target_reward_layer_->Reset(target_reward_data.data(),
                                dummy_input_data_.data(), MinibatchSize);
  }

  void PourTargetTerminationData(const BatchArray<bool> &termination_batch) {
    static std::array<float, MinibatchSize> target_termination_data;
    for (auto i = 0u; i < MinibatchSize; ++i) {
      target_termination_data.at(i) = termination_batch.at(i) ? 1.0f : 0.0f;
    }
    assert(target_termination_data.size() <= dummy_input_data_.size());
    target_termination_layer_->Reset(target_termination_data.data(),
                                     dummy_input_data_.data(), MinibatchSize);
  }

  void PourInputData(const BatchOpArray<
      std::pair<std::vector<float>, size_t> > &activation_action_pairs) {
    static std::vector<float> input_layer_data(
        (activation_data_size_ + OutputCount) * MinibatchSize, 0.0f);
    for (auto i = 0u; i < activation_action_pairs.size(); ++i) {
      if (activation_action_pairs.at(i)) {
        const auto activation_begin = input_layer_data.begin() +
                                      (activation_data_size_ + OutputCount) * i;
        const auto &activation = activation_action_pairs.at(i)->first;
        assert(activation.size() == activation_data_size_);
        std::copy(activation.begin(), activation.end(), activation_begin);
        const auto action = activation_action_pairs.at(i)->second;
        const auto action_begin = activation_begin + activation_data_size_;
        for (auto j = 0u; j < OutputCount; ++j) {
          *(action_begin + j) = j == action ? 1 : 0;
        }
      }
    }
    input_layer_->Reset(input_layer_data.data(), dummy_input_data_.data(),
                        MinibatchSize);
  }

  void PourInputData(
      const std::vector<ActivationAndAction> &activation_action_pairs) {
    assert(activation_action_pairs.size() <= MinibatchSize);
    PourInputData(util::container::VectorToOptionalArray<MinibatchSize>(
        activation_action_pairs));
  }
  void PourInputData(const std::vector<float> &input_activation,
                     const size_t action_idx) {
    PourInputData(std::vector<std::pair<std::vector<float>, size_t> >{
      std::make_pair(input_activation, action_idx)
    });
  }
  void PourInputData(const std::array<boost::optional<std::vector<float> >,
                                      MinibatchSize> &input_activations,
                     const std::vector<Transition> &transitions) {
    assert(transitions.size() == MinibatchSize);
    std::vector<std::pair<std::vector<float>, size_t> > activation_action_pairs;
    activation_action_pairs.reserve(MinibatchSize);
    for (auto i = 0u; i < MinibatchSize; ++i) {
      assert(input_activations.at(i));
      activation_action_pairs.emplace_back(input_activations.at(i).get(),
                                           transitions.at(i).action);
    }
    PourInputData(activation_action_pairs);
  }
  void PourDummyTargetActivationDataAndFilterData() {
    filter_layer_->Reset(dummy_input_data_.data(), dummy_input_data_.data(),
                         MinibatchSize);
    target_activation_layer_->Reset(dummy_input_data_.data(),
                                    dummy_input_data_.data(), MinibatchSize);
  }

  void PourTargetData(const BatchArray<RewardAndNextActivation> &target_batch) {
    // [?activation]
    const auto target_activation_batch =
        util::container::AsArray<MinibatchSize>(
            target_batch |
            A::transformed([](const auto &p) { return p.second; }));
    PourTargetActivationDataAndFilterData(target_activation_batch);
    // [double]
    const auto target_reward_batch = util::container::AsArray<MinibatchSize>(
        target_batch | A::transformed([](const auto &p) { return p.first; }));
    PourTargetRewardData(target_reward_batch);
    // [bool]
    const auto target_termination_batch =
        util::container::AsArray<MinibatchSize>(
            target_batch |
            A::transformed([](const auto &p) { return !p.second; }));
    PourTargetTerminationData(target_termination_batch);
  }

  void PourTargetActivationDataAndFilterData(
      const BatchOpArray<Activation> &target_activations) {
    static std::vector<float> target_activation_layer_data(
        activation_data_size_ * MinibatchSize, 0.0f);
    static std::vector<float> filter_layer_data(
        activation_data_size_ * MinibatchSize, 0.0f);
    for (auto i = 0u; i < MinibatchSize; ++i) {
      assert(target_activation_layer_data.size() >=
             activation_data_size_ * (i + 1));
      assert(filter_layer_data.size() >= activation_data_size_ * (i + 1));
      if (target_activations.at(i)) {
        // Non-terminal state
        std::copy(
            target_activations.at(i)->begin(), target_activations.at(i)->end(),
            target_activation_layer_data.begin() + activation_data_size_ * i);
        std::fill(filter_layer_data.begin() + activation_data_size_ * i,
                  filter_layer_data.begin() + activation_data_size_ * (i + 1),
                  1.0f);
      } else {
        // terminal state
        std::fill(target_activation_layer_data.begin() +
                      activation_data_size_ * i,
                  target_activation_layer_data.begin() +
                      activation_data_size_ * (i + 1),
                  0.0f);
        std::fill(filter_layer_data.begin() + activation_data_size_ * i,
                  filter_layer_data.begin() + activation_data_size_ * (i + 1),
                  0.0f);
      }
    }
    target_activation_layer_->Reset(target_activation_layer_data.data(),
                                    dummy_input_data_.data(), MinibatchSize);
    filter_layer_->Reset(filter_layer_data.data(), dummy_input_data_.data(),
                         MinibatchSize);
  }
  bool GetOutputTerminal(const size_t idx_in_batch) {
    const auto termination_begin = output_termination_blob_->cpu_data() +
                                   kTerminationClassCount * idx_in_batch;
    const auto termination_end = output_termination_blob_->cpu_data() +
                                 kTerminationClassCount * (idx_in_batch + 1);
    const auto termination_idx =
        std::distance(termination_begin,
                      std::max_element(termination_begin, termination_end));
    switch (termination_idx) {
    case 0:
      return false;
    case 1:
      return true;
    default:
      throw std::runtime_error("");
    }
  }
  double GetOutputReward(const size_t idx_in_batch) {
    const auto reward_begin =
        output_reward_blob_->cpu_data() + kRewardClassCount * idx_in_batch;
    const auto reward_end = output_reward_blob_->cpu_data() +
                            kRewardClassCount * (idx_in_batch + 1);
    const auto reward_idx =
        std::distance(reward_begin, std::max_element(reward_begin, reward_end));
    switch (reward_idx) {
    case 0:
      return -1.0;
    case 1:
      return 0.0;
    case 2:
      return 1.0;
    default:
      throw std::runtime_error("");
    }
  }
  Activation GetOutputActivation(const size_t idx_in_batch) const {
    return std::vector<float>(output_activation_blob_->cpu_data() +
                                  activation_data_size_ * idx_in_batch,
                              output_activation_blob_->cpu_data() +
                                  activation_data_size_ * (idx_in_batch + 1));
  }

  std::string model_solver_param_;
  dqn::SolverSp model_solver_;
  dqn::NetSp model_net_;
  MemoryDataLayerSp input_layer_;
  MemoryDataLayerSp filter_layer_;
  MemoryDataLayerSp target_activation_layer_;
  MemoryDataLayerSp target_reward_layer_;
  MemoryDataLayerSp target_termination_layer_;
  std::vector<float> dummy_input_data_;
  const bool predicts_activation_difference_ = true;
  BlobSp output_activation_blob_;
  BlobSp output_reward_blob_;
  BlobSp output_termination_blob_;
  const std::string activation_blob_name_;
  size_t activation_data_size_;
};
}

#endif /* UNIVERSAL_MODEL_HPP_ */
