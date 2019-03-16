#! /usr/bin/python3

# Allow the three system generators to run in parallel.  Each system's
# response is an HTML string.
#
# To simplyfy Docker configuration, this code supports the following
# environment variables for locating model and vocabulary files:
#
# CWC_KEYWORD_MODEL_SYSTEM_3
# CWC_KEYWORD_VOCAB_SYSTEM_3
# CWC_STORY_MODEL_SYSTEM_3
# CWC_STORY_VOCAB_SYSTEM_3
# CWC_SCORERS_CONFIG_SYSTEM_3
#
# System 3 is the same as System 2 with discriminators applied.

import system2

class System3_Generator(system2.System2_Generator):
    def __init__(self, system_id, apply_disc=True):
        super().__init__(system_id, apply_disc=apply_disc)
