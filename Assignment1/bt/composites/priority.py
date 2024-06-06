#
# Behavior Tree framework for A1 Behavior trees assignment.
# CS 131 - Artificial Intelligence
#
# Version 2.0.0 - copyright (c) 2023 Santini Fabrizio. All rights reserved.
#

import bt_library as btl


class Priority(btl.Composite):
    """
    Specific implementation of the priority composite.
    """

    def __init__(self, children: btl.NodeListType, priorities: list[int]):
        """
        Default constructor.

        :param children: List of children for this node
        :param priorities: List of integers from 0 to the number of children showing priority order for the children, lowest with highest priority
        """
        super().__init__(children)
        self.priorities = priorities
        if len(self.priorities) != len(self.children):
            raise ValueError(f'Priority Error: priorities length of {len(self.priorities)} does not match number of children {len(self.children)}')
        elif set(range(len(self.children))) != set(self.priorities):
            raise ValueError(f'Priority Error: priorities must be a list of integers from 0 to the number of children')
        
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        """
        Execute the behavior of the node.

        :param blackboard: Blackboard with the current state of the problem
        :return: The result of the execution
        """
        # Priority always starts at first priority again, regardless of running child
        for child_priority in range(0, len(self.children)):
            child_position = self.priorities.index(child_priority)
            child = self.children[child_position]

            result_child = child.run(blackboard)
            if result_child == btl.ResultEnum.SUCCEEDED:
                return self.report_succeeded(blackboard)

            if result_child == btl.ResultEnum.RUNNING:
                return self.report_running(blackboard)

        return self.report_failed(blackboard, 0)
