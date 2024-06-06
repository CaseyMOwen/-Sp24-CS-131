#
# Behavior Tree framework for A1 Behavior trees assignment.
# CS 131 - Artificial Intelligence
#
# Version 2.0.0 - copyright (c) 2023 Santini Fabrizio. All rights reserved.
#

from bt_library.blackboard import Blackboard
from bt_library.common import ResultEnum
from bt_library.decorator import Decorator
from bt_library.tree_node import TreeNode


class UntilFails(Decorator):
    """
    Specific implementation of the until fails decorator.
    """

    def __init__(self, child: TreeNode):
        """
        Default constructor.

        :param child: Child associated to the decorator
        """
        super().__init__(child)

        # self.__time = time

    def run(self, blackboard: Blackboard) -> ResultEnum:
        """
        Execute the behavior of the node.

        :param blackboard: Blackboard with the current state of the problem
        :return: The result of the execution
        """
        # Evaluate the child
        self.print_message('Running until fails')
        result_child = self.child.run(blackboard)

        # If the child failed, terminate immediately
        if result_child == ResultEnum.FAILED:
            return self.report_succeeded(blackboard)

        return self.report_running(blackboard)
