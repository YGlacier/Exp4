#ifndef REPLAY_MEMORY_HPP_
#define REPLAY_MEMORY_HPP_

#include <boost/serialization/strong_typedef.hpp>
#include <boost/optional.hpp>
#include "util/random.hpp"

namespace dqn {

// Transition tuple: (s_t, a_t, r_t, s_t+1)
template <typename State, typename Action> struct StateTransition {
  State state;
  Action action;
  double reward;
  boost::optional<State> next_state_if_not_terminal;
};

BOOST_STRONG_TYPEDEF(size_t, TransitionId);

template <typename State, typename Action> class ReplayMemory {
public:
  using Transition = StateTransition<State, Action>;
  ReplayMemory(const size_t capacity)
      : capacity_(capacity), next_transition_id_(0) {
    assert(capacity_ > 0);
  }
  void AddTransition(const Transition &transition) {
    if (whole_transitions_.size() == capacity_) {
      PopOldestTransition();
    }
    assert(whole_transitions_.size() < capacity_);
    whole_transitions_.push_back(transition);
    action_transition_ids_[transition.action].push_back(next_transition_id_);
    next_transition_id_ = next_transition_id_.t + 1;
  }
  std::vector<Transition> SampleTransitions(const size_t n) const {
    assert(!whole_transitions_.empty());
    std::vector<Transition> result;
    result.reserve(n);
    while (result.size() < n) {
      result.push_back(util::random::RandomSelect(whole_transitions_));
    }
    return result;
  }
  std::vector<std::pair<Transition, Transition> >
  SampleConsecutiveTransitionPairs(const size_t n) const {
    assert(whole_transitions_.size() >= 2 || n == 0);
    std::vector<std::pair<Transition, Transition> > result;
    result.reserve(n);
    while (result.size() < n) {
      const auto idx_first =
          util::random::RandomInt(0, whole_transitions_.size() - 2);
      const auto &transition_first = whole_transitions_.at(idx_first);
      if (transition_first.next_state_if_not_terminal) {
        const auto &transition_second = whole_transitions_.at(idx_first + 1);
        result.emplace_back(transition_first, transition_second);
      }
    }
    return result;
  }
  std::pair<Action, std::vector<Transition> >
  SampleTransitionsOfSameAction(const size_t n) const {
    assert(!whole_transitions_.empty());
    std::vector<Transition> result;
    result.reserve(n);
    result.push_back(util::random::RandomSelect(whole_transitions_));
    const auto action = result.front().action;
    while (result.size() < n) {
      const auto transition_id =
          util::random::RandomSelect(action_transition_ids_.at(action));
      result.push_back(
          whole_transitions_.at(TransitionIdToIndex(transition_id)));
    }
    return std::make_pair(action, result);
  }
  size_t size() const { return whole_transitions_.size(); }
  TransitionId next_transition_id() const { return next_transition_id_; }

private:
  void PopOldestTransition() {
    const auto &popped = whole_transitions_.front();
    const auto action = popped.action;
    assert(!action_transition_ids_.at(action).empty());
    assert(popped.state ==
           whole_transitions_.at(TransitionIdToIndex(
                                     action_transition_ids_.at(action).front()))
               .state);
    assert(popped.action ==
           whole_transitions_.at(TransitionIdToIndex(
                                     action_transition_ids_.at(action).front()))
               .action);
    action_transition_ids_.at(action).pop_front();
    whole_transitions_.pop_front();
  }
  size_t TransitionIdToIndex(const TransitionId &id) const {
    return id.t - (next_transition_id_ - whole_transitions_.size());
  }
  const size_t capacity_;
  TransitionId next_transition_id_;
  std::deque<Transition> whole_transitions_;
  std::unordered_map<Action, std::deque<TransitionId> > action_transition_ids_;
};
}

#endif /* REPLAY_MEMORY_HPP_ */
