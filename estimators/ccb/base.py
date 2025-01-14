""" Interface for implementation of conditional contextual bandits estimators """

from abc import ABC, abstractmethod
from typing import List


class Estimator(ABC):
	""" Interface for implementation of conditional contextual bandits estimators """

	@abstractmethod
	def add_example(self, p_log: List, r: List, p_pred: List, count: float) -> None:
		""" 
		Args:
			p_log: List of probability of the logging policy
			r: List of reward for choosing an action in the given context
			p_pred: List of predicted probability of making decision
			count: weight
		"""
		...

	@abstractmethod
	def get(self) -> List[float]:
		""" Calculates the selected estimator

		Returns:
			The estimator value
		"""
		...


class Interval(ABC):
	""" Interface for implementation of conditional contextual bandits estimators interval """

	@abstractmethod
	def add_example(self, p_log: List[float], r: List[float], p_pred: List[float], count: float) -> None:
		""" 
		Args:
			p_log: List of probability of the logging policy
			r: List of reward for choosing an action in the given context
			p_pred: List of predicted probability of making decision
			count: weight
		"""
		...

	@abstractmethod
	def get(self, alpha: float) -> List[List[float]]:
		""" Calculates the CI
		Args:
			alpha: alpha value
		Returns:
			Returns the confidence interval as list[float]
		"""
		...
