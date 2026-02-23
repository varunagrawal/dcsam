/**
 *
 * @file DiscretePriorFactor.h
 * @brief Discrete prior factor
 * @author Kevin Doherty, kdoherty@mit.edu
 * Copyright 2021 The Ambitious Folks of the MRG
 */

#pragma once

#include <gtsam/discrete/DiscreteFactor.h>
#include <gtsam/discrete/DiscreteKey.h>

#include <vector>

namespace dcsam {

/**
 * @brief Implementation of a discrete prior factor
 *
 * This factor specifies a prior distribution over a discrete variable. The user
 * provides a discrete key `dk` consisting of a key (e.g. `gtsam::Symbol`) and
 * the cardinality of the discrete variable. The vector `probs` specifies a
 * distribution over the possible values that could be taken by the variable.
 *
 * For example, for a variable `d1` with 2 possible values, p(d1 = i) = probs[i]
 * with i being 0 or 1. The length of the vector `probs` therefore must be equal
 * to the cardinality of the discrete variable.
 */
class DiscretePriorFactor : public gtsam::DiscreteFactor {
 protected:
  gtsam::DiscreteKey dk_;
  std::vector<double> probs_;

 public:
  using Base = gtsam::DiscreteFactor;

  DiscretePriorFactor() = default;

  DiscretePriorFactor(const gtsam::DiscreteKey& dk,
                      const std::vector<double> probs)
      : dk_(dk), probs_(probs) {
    // Ensure that length of probs is equal to the cardinality of the discrete
    // variable (for gtsam::DiscreteKey dk, dk.second is the cardinality).
    assert(probs.size() == dk.second);

    // For gtsam::DiscreteKey dk, dk.first is the Key for this variable.
    keys_.push_back(dk.first);
  }

  bool equals(const DiscreteFactor& other, double tol = 1e-9) const override {
    if (!dynamic_cast<const DiscretePriorFactor*>(&other)) return false;
    const DiscretePriorFactor& f(
        static_cast<const DiscretePriorFactor&>(other));
    if (probs_.size() != f.probs_.size() || (dk_ != f.dk_)) {
      return false;
    } else {
      for (size_t i = 0; i < probs_.size(); i++)
        if (abs(probs_[i] - f.probs_[i]) > tol) return false;
      return true;
    }
  }

  DiscretePriorFactor& operator=(const DiscretePriorFactor& rhs) {
    Base::operator=(rhs);
    dk_ = rhs.dk_;
    probs_ = rhs.probs_;
    return *this;
  }

  gtsam::DecisionTreeFactor toDecisionTreeFactor() const override {
    gtsam::DecisionTreeFactor converted(dk_, probs_);
    return converted;
  }

  gtsam::DecisionTreeFactor operator*(
      const gtsam::DecisionTreeFactor& f) const override {
    return toDecisionTreeFactor() * f;
  }

  double operator()(const DiscreteValues& values) const {
    size_t assignment = values.at(dk_.first);
    return probs_[assignment];
  }

  virtual double evaluate(
      const gtsam::Assignment<gtsam::Key>& values) const override {
    return this->operator()(DiscreteValues(values));
  }

  virtual DiscreteFactor::shared_ptr operator*(double s) const override {
    return toDecisionTreeFactor() * s;
  }

  virtual DiscreteFactor::shared_ptr multiply(
      const DiscreteFactor::shared_ptr& df) const override {
    return df->multiply(
        std::make_shared<gtsam::DecisionTreeFactor>(toDecisionTreeFactor()));
  }

  /// divide by DiscreteFactor::shared_ptr f (safely)
  virtual DiscreteFactor::shared_ptr operator/(
      const DiscreteFactor::shared_ptr& df) const override {
    throw std::logic_error("Not implemented: operator/.");
  }

  /// Create new factor by summing all values with the same separator values
  DiscreteFactor::shared_ptr sum(size_t nrFrontals) const override {
    return toDecisionTreeFactor().sum(nrFrontals);
  }

  /// Create new factor by summing all values with the same separator values
  DiscreteFactor::shared_ptr sum(const gtsam::Ordering& keys) const override {
    return toDecisionTreeFactor().sum(keys);
  }

  /// Find the maximum value in the factor.
  double max() const override { return toDecisionTreeFactor().max(); };

  /// Create new factor by maximizing over all values with the same separator.
  DiscreteFactor::shared_ptr max(size_t nrFrontals) const override {
    return toDecisionTreeFactor().max(nrFrontals);
  }

  /// Create new factor by maximizing over all values with the same separator.
  DiscreteFactor::shared_ptr max(const gtsam::Ordering& keys) const override {
    return toDecisionTreeFactor().max(keys);
  }

  virtual uint64_t nrValues() const override { return probs_.size(); }

  /// Restrict the factor to the given assignment.
  virtual DiscreteFactor::shared_ptr restrict(
      const DiscreteValues& assignment) const override {
    return toDecisionTreeFactor().restrict(assignment);
  }

  std::string markdown(const gtsam::KeyFormatter& keyFormatter,
                       const Names& names) const override {
    return toDecisionTreeFactor().markdown(keyFormatter, names);
  }

  std::string html(const gtsam::KeyFormatter& keyFormatter,
                   const Names& names) const override {
    return toDecisionTreeFactor().markdown(keyFormatter, names);
  }
};

}  // namespace dcsam
