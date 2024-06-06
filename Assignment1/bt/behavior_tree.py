#
# Behavior Tree framework for A1 Behavior trees assignment.
# CS 131 - Artificial Intelligence
#
# version 2.0.0 - copyright (c) 2023 Santini Fabrizio. All rights reserved.
#

import bt_library as btl

import bt as bt

# Instantiate the tree according to the assignment.
# Building as subtrees to make more readable

battery_check = bt.Sequence(
    [
        bt.conditions.BatteryLessThan30(),
        bt.tasks.FindHome(),
        bt.tasks.GoHome(),
        bt.tasks.Dock()
    ]
)

spot_cleaning = bt.Sequence(
    [
        bt.conditions.SpotCleaning(),
        btl.Timer(20, bt.tasks.CleanSpot()),
        bt.tasks.DoneSpot()
    ]
)

general_cleaning_actions = bt.Priority(
    [
        bt.composites.Sequence(
            [
                bt.conditions.DustySpot(),
                btl.Timer(35, bt.tasks.CleanSpot()),
                bt.tasks.AlwaysFail()
            ]
        ),
        bt.decorators.UntilFails(bt.tasks.CleanFloor())
    ],
    [0,1]
)

general_cleaning = bt.Sequence(
    [
        bt.conditions.GeneralCleaning(),
        bt.composites.Sequence(
            [
                general_cleaning_actions,
                bt.tasks.DoneGeneral()
            ]
        )
    ]
)

tree_root = bt.Priority(
    [
        battery_check,
        bt.composites.Selection(
            [
                spot_cleaning,
                general_cleaning
            ]
        ),
        bt.tasks.DoNothing()
    ],
    [0,1,2]
)