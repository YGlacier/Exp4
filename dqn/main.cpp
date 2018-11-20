#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <sstream>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/range.hpp>
#include <boost/range/numeric.hpp>
#include <boost/range/irange.hpp>
#include <boost/range/combine.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/adaptors.hpp>
#include <ale_interface.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include "prettyprint.hpp"
#include "ale.hpp"
#include "util/container.hpp"
#include "util/random.hpp"
#include "aggregation_policy.hpp"
#include "search.hpp"

#ifdef CPU_ONLY
DEFINE_bool(gpu, false, "Use GPU to brew Caffe");
#else
DEFINE_bool(gpu, true, "Use GPU to brew Caffe");
#endif

DEFINE_bool(gui, false, "Open a GUI window");

DEFINE_string(rom, "", "Atari 2600 ROM to play e.g. breakout.bin");
DEFINE_string(solver, "dqn_solver.prototxt",
              "Solver parameter file (*.prototxt)");
DEFINE_int32(memory, 500000, "Capacity of replay memory");
DEFINE_int32(explore, 1000000,
             "Number of iterations needed for epsilon to reach 0.1");
DEFINE_double(gamma, 0.95, "Discount factor of future rewards (0,1]");
DEFINE_int32(skip_frame, 3, "Number of frames skipped");
DEFINE_bool(show_frame, false, "Show the current frame in CUI");
DEFINE_string(dqn_bin, "", "DQN *.caffemodel file to load");
DEFINE_string(universal_model_bin, "",
              "UniversalModel *.caffemodel file to load");
DEFINE_bool(evaluate, false, "Evaluate performance of trained DQN");
DEFINE_double(evaluate_with_epsilon, 0.05,
              "Epsilon value to be used in evaluation mode");
DEFINE_int32(repeat_games, 1, "Number of games played in evaluation mode");
DEFINE_int32(model_search_depth, 0, "Depth of forward search using models");
DEFINE_int32(ale_search_depth, 0, "Depth of search using ALE interface");
DEFINE_int32(search_unit, 1,
             "Number of repetition of the same action in forward search");
DEFINE_int32(seed, 0, "Seed of the random generator");
DEFINE_int32(max_iter, 2000000, "Max iteration of training");
DEFINE_string(activation_blob, "ip1",
              "Blob name of DQN used as hidden activation in UniversalModel");
DEFINE_string(universal_model_solver, "",
              "*.prototxt file for UniversalModel solver");
DEFINE_string(mode, "", "train_dqn/train_model/eval_dqn/eval_model/dyna");
DEFINE_string(aggregation_policy, "max",
              "Aggregation policy in forward search: none, max, min, mean");
DEFINE_bool(leaf_evaluation, true, "Evaluate leaves of search trees by DQN");
DEFINE_int32(hallucination, 0, "Number of hallucinated samples in a minibatch");
DEFINE_bool(fix, true, "Use a fixed version of screen preprocessing");
DEFINE_int32(warmup_by_dqn, 0,
             "Number of DQN-based transitions to fill replay memory by");
DEFINE_double(dqn_mix_rate, 0.0, "x of x*Q_DQN(s,a) + (1-x)*Q_MDL(s,a)");
DEFINE_bool(after_relu, false, "Use activations after ReLU for model learning");
DEFINE_double(dqn_lr, 0.2, "Learning rate for training DQN");

namespace A = boost::adaptors;

namespace {

std::string GetCurrentDatetimeString() {
  // std::put_time is not available in gcc until gcc5, so use strftime instead.
  const auto t = std::time(nullptr);
  char buf[(4 + 2 + 2) + (2 + 2 + 2) + 1]; // yyyyMMddHHmmss and null
  std::strftime(buf, sizeof(buf), "%Y%m%d%H%M%S", std::localtime(&t));
  return std::string(buf);
  // const auto now = std::chrono::system_clock::now();
  // const auto in_time_t = std::chrono::system_clock::to_time_t(now);
  // std::ostringstream ss;
  // ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d%H%M%S");
  // return ss.str();
}

double CalculateEpsilon(const int iter) {
  if (iter < FLAGS_explore) {
    return 1.0 - 0.9 * (static_cast<double>(iter) / FLAGS_explore);
  } else {
    return 0.1;
  }
}

std::string GetSnapshotPrefix() {
  const auto dir_name =
      GetCurrentDatetimeString() + "_" +
      boost::algorithm::replace_all_copy(FLAGS_rom, ".bin", "");
  boost::filesystem::create_directory(dir_name);
  std::cout << "dir_name:" << dir_name << std::endl;
  return dir_name + "/";
}
}

namespace dqn {

namespace ale {

using ActionSelector = std::function<Action(const DQN::InputFrames &)>;
using TransitionUser = std::function<void(const Transition &)>;

Action SelectActionEpsilonGreedily(DQN &dqn, const double epsilon,
                                   const DQN::InputFrames &input_frames) {
  const auto eval = dqn.EvaluateActions(input_frames);
  std::cout << "epsilon:" << epsilon << std::endl;
  std::cout << PrintActionEvaluation(eval) << std::endl;
  if (util::random::RandomDouble(0, 1) < epsilon) {
    return static_cast<Action>(util::random::RandomSelectKey(eval));
  } else {
    return static_cast<Action>(util::container::KeyOfMaxValue(eval));
  }
};

reward_t RepeatActionAndObserveScore(ALEInterface &ale, const Action action) {
  reward_t immediate_score = 0;
  for (auto i = 0; i < FLAGS_skip_frame + 1 && !ale.game_over(); ++i) {
    immediate_score += ale.act(action);
  }
  return immediate_score;
}

DQN::InputFrames GetNextInputFrames(const DQN::InputFrames &current_frames,
                                    const ALEScreen &next_screen) {
  return DQN::InputFrames{ *(current_frames.end() - 3),
                           *(current_frames.end() - 2),
                           *(current_frames.end() - 1),
                           PreprocessScreen(next_screen) };
}

double NormalizeReward(const reward_t raw_reward) {
  // Rewards for DQN are normalized as follows:
  // 1 for any positive score, -1 for any negative score, otherwise 0
  return raw_reward == 0 ? 0.0 : raw_reward / std::abs(raw_reward);
}

double PlayOneEpisode(ALEInterface &ale, const ActionSelector &action_selector,
                      const TransitionUser &transition_user) {
  assert(!ale.game_over());
  std::deque<DQN::FrameDataSp> past_frames;
  auto total_score = 0.0;
  for (auto frame = 0; !ale.game_over(); ++frame) {
    std::cout << "frame: " << frame << std::endl;
    const auto current_frame = PreprocessScreen(ale.getScreen());
    if (FLAGS_show_frame) {
      std::cout << DrawFrame(*current_frame) << std::endl;
    }
    assert(past_frames.size() <= dqn::ale::kInputFrameCount);
    if (past_frames.size() == dqn::ale::kInputFrameCount) {
      past_frames.pop_front();
    }
    past_frames.push_back(current_frame);
    if (past_frames.size() < dqn::ale::kInputFrameCount) {
      // If there are not past frames enough for DQN input, just select NOOP
      total_score += RepeatActionAndObserveScore(ale, PLAYER_A_NOOP);
    } else {
      assert(past_frames.size() == dqn::ale::kInputFrameCount);
      DQN::InputFrames input_frames;
      std::copy(past_frames.begin(), past_frames.end(), input_frames.begin());
      const auto action = action_selector(input_frames);
      const auto immediate_score =
          RepeatActionAndObserveScore(ale, static_cast<Action>(action));
      total_score += immediate_score;
      const auto reward = NormalizeReward(immediate_score);
      const auto next_state_if_not_terminal =
          ale.game_over() ? boost::none
                          : boost::make_optional(GetNextInputFrames(
                                input_frames, ale.getScreen()));
      // Add the current transition to replay memory
      const auto transition = Transition{ input_frames, action, reward,
                                          next_state_if_not_terminal };
      transition_user(transition);
    }
  }
  ale.reset_game();
  return total_score;
}

std::vector<size_t> GetLegalActionIndices(ALEInterface &ale) {
  // [Action] -> [size_t]
  return util::container::AsVector(
      ale.getMinimalActionSet() |
      A::transformed([](const auto &a) { return static_cast<size_t>(a); }));
}

std::pair<size_t, double> FullForwardSearchUsingTrainedModel(
    DQN &dqn, UniversalModel &model,
    const std::vector<size_t> &legal_action_indices,
    const DQN::InputFrames &input_frames, const size_t depth,
    const AggregationPolicy policy) {
  if (depth == 0) {
    return dqn.SelectActionGreedily(input_frames);
  }
  const auto current_activation =
      dqn.GetHiddenActivation(FLAGS_activation_blob, input_frames);
  const auto batch_plan_evaluator = [&](const std::vector<ActionPlan> &plans) {
    assert(!plans.empty() && plans.size() <= kMinibatchSize);
    const auto plan_batch =
        util::container::VectorToOptionalArray<kMinibatchSize>(plans);
    const auto activation_action_pairs =
        util::container::AsArray<kMinibatchSize>(
            plans | A::transformed([&](const auto &plan) {
                      return boost::make_optional(
                          std::make_pair(current_activation, plan.front()));
                    }));
    const auto reward_and_next_activation_or_none =
        model.PredictNextActivations(activation_action_pairs);
    auto rewards =
        util::container::AsVector(reward_and_next_activation_or_none |
                                  A::transformed([](const auto &p) {
                                    return p ? FLAGS_gamma * p->first : 0.0;
                                  }));
    assert(rewards.size() == kMinibatchSize);
    // [?(reward,?activation)] -> [?activation]
    auto next_activations_or_none = util::container::AsArray<kMinibatchSize>(
        reward_and_next_activation_or_none | A::transformed([](const auto &p) {
                                               return p ? p->second
                                                        : boost::none;
                                             }));
    for (auto i = 1; i < plans.front().size(); ++i) {
      // [plan], [?activation] -> [?(activation,action)]
      const auto activation_and_action_pairs =
          util::container::AsArray<kMinibatchSize>(
              boost::combine(plan_batch, next_activations_or_none) |
              A::transformed([&](const auto &t) {
                const auto &plan_or_none = t.get<0>();
                const auto &next_activation_or_none = t.get<1>();
                return plan_or_none && next_activation_or_none
                           ? boost::make_optional(
                                 std::make_pair(next_activation_or_none.get(),
                                                plan_or_none.get().at(i)))
                           : boost::none;
              }));
      // [?(activation,action)] -> [?(reward,?activation)] in batch
      const auto reward_and_next_activation_or_none_batch =
          model.PredictNextActivations(activation_and_action_pairs);
      // Accumulate reward for each plan
      for (const auto idx_in_batch :
           boost::irange(static_cast<size_t>(0), kMinibatchSize)) {
        if (reward_and_next_activation_or_none_batch.at(idx_in_batch)) {
          rewards.at(idx_in_batch) +=
              std::pow(FLAGS_gamma, i + 1) *
              reward_and_next_activation_or_none_batch.at(idx_in_batch)->first;
        }
      }
      // [?(reward,?activation)] -> [?activation]
      next_activations_or_none = util::container::AsArray<kMinibatchSize>(
          reward_and_next_activation_or_none_batch |
          A::transformed([](const auto &p) {
            return p ? p->second : boost::none;
          }));
    }
    //    std::cout << "rewards:" << rewards << std::endl;
    if (!FLAGS_leaf_evaluation) {
      return util::container::AsVector(
          rewards | A::sliced(static_cast<size_t>(0), plans.size()));
    } else {
      // [?activation] -> [?eval] in batch
      const auto evals = dqn.EvaluateActionsFromHiddenActivation(
          FLAGS_activation_blob, next_activations_or_none);
      // [?eval] -> [?value]
      return util::container::AsVector(
          boost::irange(static_cast<size_t>(0), plans.size()) |
          A::transformed([&](const auto i) {
            //            if (evals.at(i)) {
            //              std::cout << "eval" <<
            // ActionIndicesToString(plans.at(i))
            //                        << evals.at(i).get() << std::endl;
            //            }
            return rewards.at(i) +
                   (evals.at(i) ? Aggregate(evals.at(i).get(), policy) : 0.0);
          }));
    }
  };
  auto search_eval =
      FullForwardSearch(legal_action_indices, depth, batch_plan_evaluator,
                        policy, FLAGS_search_unit, kMinibatchSize);
  if (FLAGS_dqn_mix_rate > 0.0) {
    //    std::cout << "mix" << std::endl;
    const auto dqn_eval = dqn.EvaluateActions(input_frames);
    std::cout << "search_eval:" << search_eval << std::endl;
    std::cout << "dqn_eval:" << dqn_eval << std::endl;
    for (const auto &a : legal_action_indices) {
      search_eval[a] = dqn_eval.at(a) * FLAGS_dqn_mix_rate +
                       search_eval[a] * (1.0 - FLAGS_dqn_mix_rate);
    }
    //    std::cout << "mixed_eval:" << search_eval << std::endl;
  }
  return *boost::max_element(search_eval,
                             util::container::SecondComparator<size_t, double>);
}

std::pair<size_t, double>
FullForwardSearchUsingTrueModel(ALEInterface &ale, DQN &dqn,
                                const std::vector<size_t> &legal_action_indices,
                                const DQN::InputFrames &input_frames,
                                const size_t depth,
                                const AggregationPolicy policy) {
  assert(!ale.game_over());
  if (depth == 0) {
    return dqn.SelectActionGreedily(input_frames);
  }
  const auto current_state = ale.cloneState();
  const auto batch_plan_evaluator = [&](const std::vector<ActionPlan> &plans) {
    const auto reward_and_next_frames_batch = util::container::AsVector(
        plans | A::transformed([&](const auto &plan) {
                  auto reward = 0.0;
                  auto next_input_frames = input_frames;
                  for (const auto action_idx : plan) {
                    reward += NormalizeReward(RepeatActionAndObserveScore(
                        ale, static_cast<Action>(action_idx)));
                    if (ale.game_over()) {
                      break;
                    }
                    next_input_frames =
                        GetNextInputFrames(next_input_frames, ale.getScreen());
                  }
                  const auto result = std::make_pair(
                      reward, ale.game_over()
                                  ? boost::none
                                  : boost::make_optional(next_input_frames));
                  ale.restoreState(current_state);
                  return result;
                }));
    if (!FLAGS_leaf_evaluation) {
      return util::container::AsVector(
          reward_and_next_frames_batch | A::map_keys |
          A::sliced(static_cast<size_t>(0), plans.size()));
    } else {
      const auto next_frames_batch = util::container::AsArray<kMinibatchSize>(
          reward_and_next_frames_batch |
          A::transformed([](const auto &p) { return p.second; }));
      const auto evals = dqn.EvaluateActions(next_frames_batch);
      return util::container::AsVector(
          boost::irange(static_cast<size_t>(0), plans.size()) |
          A::transformed([&](const auto i) {
            if (evals.at(i)) {
              std::cout << "eval" << ActionIndicesToString(plans.at(i))
                        << evals.at(i).get() << std::endl;
            }
            return reward_and_next_frames_batch.at(i).first +
                   (evals.at(i) ? Aggregate(evals.at(i).get(), policy) : 0.0);
          }));
    }
  };
  auto search_eval =
      FullForwardSearch(legal_action_indices, depth, batch_plan_evaluator,
                        policy, FLAGS_search_unit, kMinibatchSize);
  if (FLAGS_dqn_mix_rate > 0.0) {
    //    std::cout << "mix" << std::endl;
    const auto dqn_eval = dqn.EvaluateActions(input_frames);
    //    std::cout << "search_eval:" << search_eval << std::endl;
    //    std::cout << "dqn_eval:" << dqn_eval << std::endl;
    for (const auto &a : legal_action_indices) {
      search_eval[a] = dqn_eval.at(a) * FLAGS_dqn_mix_rate +
                       search_eval[a] * (1.0 - FLAGS_dqn_mix_rate);
    }
    //    std::cout << "mixed_eval:" << search_eval << std::endl;
  } else if (std::all_of(search_eval.begin(), search_eval.end(),
                         [&](const auto &p) {
               return p.second == search_eval.begin()->second;
             })) {
    // The values of all the plan are completely the same
    std::cout << "Using backup action selector" << std::endl;
    return dqn.SelectActionGreedily(input_frames);
  }
  return *boost::max_element(search_eval,
                             util::container::SecondComparator<size_t, double>);
}

void TrainModelWithSampledBatch(DQN &dqn, UniversalModel &model,
                                ReplayMemory &replay_memory, const bool log) {
  assert(FLAGS_hallucination >= 0 && FLAGS_hallucination <= kMinibatchSize);
  const auto hallucination =
      replay_memory.size() >= 2 ? FLAGS_hallucination : 0;
  const auto &single_transitions =
      replay_memory.SampleTransitions(kMinibatchSize - hallucination);
  const auto &transition_pairs =
      replay_memory.SampleConsecutiveTransitionPairs(hallucination);
  model.UpdateModel(dqn, single_transitions, transition_pairs, log);
}

ActionSelector EpsilonGreedily(const double epsilon,
                               const std::vector<size_t> &action_indices,
                               const ActionSelector &greedy_selector) {
  return [=](const DQN::InputFrames &input_frames) {
    if (util::random::RandomDouble(0, 1) < epsilon) {
      return static_cast<Action>(util::random::RandomSelect(action_indices));
    } else {
      return greedy_selector(input_frames);
    }
  };
}

void TrainUniversalModelWithGivenDQN(ALEInterface &ale) {
  const auto &legal_action_indices = GetLegalActionIndices(ale);
  dqn::ale::DQN dqn(legal_action_indices, FLAGS_solver, FLAGS_gamma,
                    FLAGS_after_relu);
  const auto snapshot_prefix = GetSnapshotPrefix();
  dqn.Initialize(snapshot_prefix);
  assert(!FLAGS_dqn_bin.empty());
  std::cout << "Loading " << FLAGS_dqn_bin << std::endl;
  dqn.LoadTrainedModel(FLAGS_dqn_bin);
  dqn::ale::ReplayMemory replay_memory(FLAGS_memory);
  assert(!FLAGS_activation_blob.empty());
  assert(!FLAGS_universal_model_solver.empty());
  auto universal_model =
      UniversalModel(dqn, FLAGS_activation_blob, FLAGS_universal_model_solver,
                     snapshot_prefix);
  if (!FLAGS_universal_model_bin.empty()) {
    universal_model.LoadTrainedModel(FLAGS_universal_model_bin);
  }
  std::cout << "Hallucination: " << FLAGS_hallucination << std::endl;
  std::cout << "policy:" << FLAGS_aggregation_policy << std::endl;
  const auto agg_policy = ParseAggregationPolicy(FLAGS_aggregation_policy);
  auto iter = 0;
  const auto action_selector =
      EpsilonGreedily(FLAGS_evaluate_with_epsilon, legal_action_indices,
                      [&](const DQN::InputFrames &input_frames) {
        if (iter < FLAGS_warmup_by_dqn) {
          std::cout << "warmup_by_dqn:" << iter << "/" << FLAGS_warmup_by_dqn
                    << std::endl;
          if (iter == FLAGS_warmup_by_dqn - 1) {
            std::cout << "warmup done" << std::endl;
          }
          return static_cast<Action>(
              dqn.SelectActionGreedily(input_frames).first);
        }
        return static_cast<Action>(
            FullForwardSearchUsingTrainedModel(
                dqn, universal_model, legal_action_indices, input_frames,
                FLAGS_model_search_depth, agg_policy).first);
      });
  const auto transition_user = [&](const Transition &transition) {
    std::cout << "iteration: " << iter << std::endl;
    ++iter;
    // Add the current transition to replay memory
    replay_memory.AddTransition(transition);
    const auto log = iter % 10 == 0;
    TrainModelWithSampledBatch(dqn, universal_model, replay_memory, log);
  };
  for (auto episode = 0; iter < FLAGS_max_iter; episode++) {
    std::cout << "episode: " << episode << std::endl;
    const auto score = PlayOneEpisode(ale, action_selector, transition_user);
    std::cout << "episode score:" << score << std::endl;
  }
  std::cout << "Finished " << iter << "/" << FLAGS_max_iter << " iterations."
            << std::endl;
}

void EvaluatePerformance(ALEInterface &ale, const size_t repetitions,
                         const ActionSelector &action_selector,
                         const TransitionUser &transition_user) {
  std::vector<double> scores;
  for (auto i = 0; i < repetitions; ++i) {
    std::cout << "game: " << i << std::endl;
    const auto score =
        dqn::ale::PlayOneEpisode(ale, action_selector, transition_user);
    std::cout << "score: " << score << std::endl;
    scores.push_back(score);
    std::cout << "scores so far: " << scores << std::endl;
    const auto max = *std::max_element(scores.begin(), scores.end());
    const auto min = *std::min_element(scores.begin(), scores.end());
    const auto sum = std::accumulate(scores.begin(), scores.end(), 0.0);
    const auto mean = sum / scores.size();
    std::cout << "max score:" << max << std::endl;
    std::cout << "min score:" << min << std::endl;
    std::cout << "mean score:" << mean << std::endl;
  }
  std::cout << "all the scores: " << scores << std::endl;
}

void EvaluateTrainedModel(ALEInterface &ale) {
  const auto snapshot_prefix = GetSnapshotPrefix();
  const auto legal_action_indices = GetLegalActionIndices(ale);
  DQN dqn(legal_action_indices, FLAGS_solver, FLAGS_gamma, FLAGS_after_relu);
  dqn.Initialize(snapshot_prefix);
  assert(!FLAGS_dqn_bin.empty());
  std::cout << "Loading " << FLAGS_dqn_bin << std::endl;
  dqn.LoadTrainedModel(FLAGS_dqn_bin);
  UniversalModel model(dqn, FLAGS_activation_blob, FLAGS_universal_model_solver,
                       snapshot_prefix);
  std::cout << "Loading " << FLAGS_universal_model_bin << std::endl;
  model.LoadTrainedModel(FLAGS_universal_model_bin);
  std::cout << "policy:" << FLAGS_aggregation_policy << std::endl;
  const auto agg_policy = ParseAggregationPolicy(FLAGS_aggregation_policy);
  const auto action_selector =
      EpsilonGreedily(FLAGS_evaluate_with_epsilon, legal_action_indices,
                      [&](const DQN::InputFrames &input_frames) {
        return static_cast<Action>(FullForwardSearchUsingTrainedModel(
                                       dqn, model, legal_action_indices,
                                       input_frames, FLAGS_model_search_depth,
                                       agg_policy).first);
      });
  const auto transition_user = [&](const Transition &transition) {};
  EvaluatePerformance(ale, FLAGS_repeat_games, action_selector,
                      transition_user);
}

void EvaluateTrainedDQN(ALEInterface &ale) {
  const auto legal_action_indices = GetLegalActionIndices(ale);
  DQN dqn(legal_action_indices, FLAGS_solver, FLAGS_gamma, FLAGS_after_relu);
  dqn.Initialize(GetSnapshotPrefix());
  assert(!FLAGS_dqn_bin.empty());
  std::cout << "Loading " << FLAGS_dqn_bin << std::endl;
  dqn.LoadTrainedModel(FLAGS_dqn_bin);
  std::cout << "policy:" << FLAGS_aggregation_policy << std::endl;
  const auto agg_policy = ParseAggregationPolicy(FLAGS_aggregation_policy);
  const auto action_selector =
      EpsilonGreedily(FLAGS_evaluate_with_epsilon, legal_action_indices,
                      [&](const DQN::InputFrames &input_frames) {
        if (FLAGS_ale_search_depth > 0) {
          return static_cast<Action>(FullForwardSearchUsingTrueModel(
                                         ale, dqn, legal_action_indices,
                                         input_frames, FLAGS_ale_search_depth,
                                         agg_policy).first);
        } else {
          return static_cast<Action>(
              dqn.SelectActionGreedily(input_frames).first);
        }
      });
  const auto transition_user = [&](const Transition &transition) {};
  EvaluatePerformance(ale, FLAGS_repeat_games, action_selector,
                      transition_user);
}

std::array<DQN::TrainingSample, kMinibatchSize>
GenerateSimulatedTrainingSample(DQN &dqn, UniversalModel &model,
                                const std::vector<Transition> &transitions) {
  assert(transitions.size() == kMinibatchSize);
  // [(s,a,r,s?)] -> [s]
  const auto current_state_batch = util::container::AsArray<kMinibatchSize>(
      transitions | A::transformed([](const Transition &transition) {
                      return boost::make_optional(transition.state);
                    }));
  // [s] -> [h] batch
  const auto current_activation_batch =
      dqn.GetHiddenActivations(FLAGS_activation_blob, current_state_batch);
  // [h], [(s,a,r,s?)] -> [(h, a)]
  const auto current_activation_and_action_batch = util::container::AsVector(
      boost::combine(current_activation_batch, transitions) |
      A::transformed([](const auto &t) {
        return std::make_pair(boost::get<0>(t).get(), boost::get<1>(t).action);
      }));
  assert(current_activation_and_action_batch.size() == kMinibatchSize);
  // [(h, a)] -> [(r', h'?)] batch (' == predicted)
  const auto predicted_reward_and_next_activation_batch =
      model.PredictNextActivations(current_activation_and_action_batch);
  assert(predicted_reward_and_next_activation_batch.size() == kMinibatchSize);
  // [(r', h'?)] -> [h'?]
  const auto predicted_next_activation_batch =
      util::container::AsArray<kMinibatchSize>(
          predicted_reward_and_next_activation_batch |
          A::transformed([](const auto &reward_and_next_activation) {
            return reward_and_next_activation.second;
          }));
  // [h'?] -> [e'?] batch
  const auto evalations_based_on_predicted_next_activation_batch =
      dqn.EvaluateActionsFromHiddenActivation(FLAGS_activation_blob,
                                              predicted_next_activation_batch);
  // [(r', h'?)], [e'?] -> [Q']
  const auto target_q_range =
      boost::combine(predicted_reward_and_next_activation_batch,
                     evalations_based_on_predicted_next_activation_batch) |
      A::transformed([](const auto &t) {
        const auto predicted_r = boost::get<0>(t).first;
        const auto predicted_next_q =
            boost::get<1>(t)
                ? boost::max_element(
                      boost::get<1>(t).get(),
                      util::container::SecondComparator<size_t, double>)->second
                : 0.0;
        return predicted_r + predicted_next_q;
      });
  // [(s,a,r,s?)], [Q'] -> [(s,a,Q')]
  const auto samples = util::container::AsArray<kMinibatchSize>(
      boost::combine(transitions, target_q_range) |
      A::transformed([](const auto &t) {
        const auto &transition = boost::get<0>(t);
        const auto target_q = boost::get<1>(t);
        return DQN::TrainingSample{ transition.state, transition.action,
                                    target_q };
      }));
  return samples;
}

std::string PrintDQNTrainingSamples(
    const std::array<DQN::TrainingSample, kMinibatchSize> &samples) {
  std::ostringstream os;
  os << util::container::AsVector(
            samples |
            A::transformed([](const auto &sample) {
              return std::make_pair(ActionIndexToString(std::get<1>(sample)),
                                    std::get<2>(sample));
            }));
  return os.str();
}

void TrainDQNWithSampledBatch(DQN &dqn, ReplayMemory &replay_memory,
                              const bool log) {
  const auto &transitions_for_direct_training =
      util::container::VectorToArray<Transition, kMinibatchSize>(
          replay_memory.SampleTransitions(kMinibatchSize));
  const auto samples_based_on_real_experiences =
      dqn.GenerateTrainingSamples(transitions_for_direct_training);
  if (log) {
    std::cout << "real:" << PrintDQNTrainingSamples(
                                samples_based_on_real_experiences) << std::endl;
  }
  dqn.Update(samples_based_on_real_experiences);
}

void TrainDQNWithPlanningOnSampledBatch(DQN &dqn, UniversalModel &model,
                                        ReplayMemory &replay_memory,
                                        const bool log) {
  const auto &transitions_for_simulated_training =
      replay_memory.SampleTransitions(kMinibatchSize);
  const auto samples_based_on_simulated_experiences =
      GenerateSimulatedTrainingSample(dqn, model,
                                      transitions_for_simulated_training);
  if (log) {
    std::cout << "simulated:"
              << PrintDQNTrainingSamples(samples_based_on_simulated_experiences)
              << std::endl;
  }
  dqn.Update(samples_based_on_simulated_experiences);
}

void PlayEpisodesByEpsilonGreedyDQN(ALEInterface &ale, DQN &dqn,
                                    const TransitionUser &transition_user) {
  const auto action_selector_for_eval =
      std::bind(SelectActionEpsilonGreedily, dqn, 0.05, std::placeholders::_1);
  const auto transition_user_for_eval = [](const Transition &transition) {};
  auto iter = 0;
  for (auto episode = 0; iter < FLAGS_max_iter; episode++) {
    const auto epsilon = CalculateEpsilon(dqn.current_iteration());
    std::cout << "episode: " << episode << std::endl;
    PlayOneEpisode(ale,
                   [&](const auto &input) {
                     ++iter;
                     return SelectActionEpsilonGreedily(dqn, epsilon, input);
                   },
                   transition_user);
    //        std::bind(SelectActionEpsilonGreedily, dqn, epsilon,
    //                                  std::placeholders::_1)},
    if (episode > 0 && episode % 10 == 0) {
      // After every 10 episodes, evaluate the current strength
      const auto eval_score = PlayOneEpisode(ale, action_selector_for_eval,
                                             transition_user_for_eval);
      std::cout << "evaluation score: " << eval_score << std::endl;
    }
  }
  std::cout << "Finished " << iter << "/" << FLAGS_max_iter << " iterations."
            << std::endl;
}

void TrainDQN(ALEInterface &ale, DQN &dqn) {
  dqn::ale::ReplayMemory replay_memory(FLAGS_memory);
  const auto transition_user = [&](const Transition &transition) {
    // Add the current transition to replay memory
    replay_memory.AddTransition(transition);
    const auto log = replay_memory.next_transition_id() % 10 == 0;
    // Training DQN based on real experiences
    TrainDQNWithSampledBatch(dqn, replay_memory, log);
  };
  PlayEpisodesByEpsilonGreedyDQN(ale, dqn, transition_user);
}

void TrainDQN(ALEInterface &ale) {
  dqn::ale::DQN dqn(GetLegalActionIndices(ale), FLAGS_solver, FLAGS_gamma,
                    FLAGS_after_relu);
  dqn.Initialize(GetSnapshotPrefix());
  TrainDQN(ale, dqn);
}

void TrainDQNAndModelWithDynaQ(ALEInterface &ale, DQN &dqn,
                               UniversalModel &model) {
  dqn::ale::ReplayMemory replay_memory(FLAGS_memory);
  const auto transition_user = [&](const Transition &transition) {
    // Add the current transition to replay memory
    replay_memory.AddTransition(transition);
    const auto log = replay_memory.next_transition_id() % 10 == 0;
    // Training DQN based on real experiences
    TrainDQNWithSampledBatch(dqn, replay_memory, log);
    // Training model based on real experiences
    TrainModelWithSampledBatch(dqn, model, replay_memory, log);
    // Training DQN based on simulated experiences
    TrainDQNWithPlanningOnSampledBatch(dqn, model, replay_memory, log);
  };
  PlayEpisodesByEpsilonGreedyDQN(ale, dqn, transition_user);
}

void TrainDQNAndModelWithDynaQ(ALEInterface &ale) {
  assert(!FLAGS_universal_model_solver.empty());
  const auto &legal_action_indices = GetLegalActionIndices(ale);
  dqn::ale::DQN dqn(legal_action_indices, FLAGS_solver, FLAGS_gamma,
                    FLAGS_after_relu);
  const auto snapshot_prefix = GetSnapshotPrefix();
  dqn.Initialize(snapshot_prefix);
  if (!FLAGS_dqn_bin.empty()) {
    std::cout << "Loading DQN from a file " << FLAGS_dqn_bin << std::endl;
    dqn.LoadTrainedModel(FLAGS_dqn_bin);
  }
  UniversalModel model(dqn, FLAGS_activation_blob, FLAGS_universal_model_solver,
                       snapshot_prefix);
  if (!FLAGS_universal_model_bin.empty()) {
    std::cout << "Loading Model from a file " << FLAGS_universal_model_bin
              << std::endl;
    model.LoadTrainedModel(FLAGS_universal_model_bin);
  }
  TrainDQNAndModelWithDynaQ(ale, dqn, model);
}
}
}

int main(int argc, char **argv) {
  for (auto i = 0; i < argc; ++i) {
    std::cout << argv[i] << ' ';
  }
  std::cout << std::endl;
  using namespace dqn::ale;
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  google::LogToStderr();

#ifdef CPU_ONLY
  if (FLAGS_gpu) {
    std::cerr << "Please rebuild with CPU_ONLY turned off." << std::endl;
    std::abort();
  }
#endif
#ifndef __USE_SDL
  if (FLAGS_gui) {
    std::cerr << "Please rebuild with USE_SDL turned on." << std::endl;
    std::abort();
  }
#endif

  if (FLAGS_gpu) {
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
  } else {
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
  }

  ALEInterface ale(FLAGS_gui);

  // Load the ROM file
  ale.loadROM(FLAGS_rom);

  const auto mode_to_func =
      std::unordered_map<std::string, std::function<void()> >{
        { "train_dqn", [&]() { TrainDQN(ale); } },
        { "train_model", [&]() { TrainUniversalModelWithGivenDQN(ale); } },
        { "eval_dqn", [&]() { EvaluateTrainedDQN(ale); } },
        { "eval_model", [&]() { EvaluateTrainedModel(ale); } },
        { "dyna", [&]() { TrainDQNAndModelWithDynaQ(ale); } },
        { "",
          []() { std::cerr << "Please specify --mode option." << std::endl; } }
      };
  if (mode_to_func.count(FLAGS_mode)) {
    mode_to_func.at(FLAGS_mode)();
  } else {
    std::cerr << FLAGS_mode << " is not a valid mode." << std::endl;
  }
}
