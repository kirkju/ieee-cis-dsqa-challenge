"""What every test runs under.

The entry runs with the adjustments that DSQA_VARIANT names, or, when it is
not set, with those written in dsqa/variant.py as DEFAULT, which a
candidate's branch sets to its own. The tests name what they run with: the
variable empty, for none, set here before the entry's package is imported;
and a test that is about an adjustment sets the variable to it, or unsets
it to run what the branch ships with. So they hold whatever DEFAULT a
branch has, and a variable left set in the shell does not change them.
"""

import os

os.environ["DSQA_VARIANT"] = ""
