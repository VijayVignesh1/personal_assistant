
from deepeval.metrics import BaseConversationalMetric
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    check_conversational_test_case_params,
    construct_verbose_logs,
)
from deepeval.test_case import ConversationalTestCase, SingleTurnParams


class ContainsMetric(BaseConversationalMetric):
    _required_params: list[SingleTurnParams] = [  # noqa: RUF012
        SingleTurnParams.INPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ]
    def __init__(
        self,
        contains: str,
        threshold: float = 1.0,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        self.contains = contains
        self.threshold = threshold
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        self.score = 0.0

    def measure(
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        check_conversational_test_case_params(
            test_case,
            [],
            self
        )

        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            actual = test_case.turns[-1].content.strip()

            if self.contains in actual:
                self.score = self.precision = self.recall = self.f1 = 1.0
                self.reason = (
                    "The actual output contains the expected keyword."
                )
            else:
                self.score = self.precision = self.recall = self.f1 = 0.0
                self.reason = "The actual output does not contain the expected keyword."

            self.success = self.is_successful()

            if self.verbose_mode:
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Score: {self.score:.2f}",
                        f"Reason: {self.reason}",
                        f"Model Output: {actual}",
                    ],
                )

            return self.score


    async def a_measure(
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        return self.measure(
            test_case,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        )

    @property
    def __name__(self) -> str:
        return "ContainsMetric"