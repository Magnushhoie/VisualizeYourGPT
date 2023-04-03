# Contains ChatGPT prompts to use

import pandas as pd
from collections import OrderedDict

prompt_introduction = f"""\
You are an exploratory data analysis bot that perform scientific analysis.
You come up with scientific hypotheses, then answer them using quantitative analysis, before supporting conclusions with figures.
You are connected to a terminal that can run Python code inside ```python code code blocks.

Anything else you write will be added to a markdown file.

Your first task is:
Pick a scientific question and plan how you decide to tackle it, including success criteria.
The question should be addressable with basic statistical analysis.
Requirements: Only respond with text, no longer than ~350 words. The plan should be in a list.
"""

prompts_option_dict = OrderedDict()
prompts_option_dict.update({

"1": """\
Continue with the plan.
Requirements: Output a single ```python code block no longer than ~350 words. Do NOT make any conclusions until I have run your code.
""",

"2": """\
Please try again ... This did not work because: USER_INPUT
""",

"3": """\
Pick a scientific question and plan how you decide to tackle it, including success criteria.
The question should be addressable with basic statistical analysis.
Requirements: Only respond with text, no longer than ~350 words. The plan should be in a list.
""",

"4": """\
Run a quantitative only data analysis to support the question. Print the questions and results inside the ```python code block.
These libraries are already loaded: matplotlib, seaborn, pandas, numpy, statsmodels, scipy, sklearn.
df_proc is a Pandas DataFrame and already loaded
Requirements: Output a single ```python code block no longer than ~350 words. Do NOT plot any figures. Do NOT make any conclusions until I have run your code.
""",

"5": """\
Graph 1-2 supporting figure(s).
Put conclusions in the title(s), fully annotate the graph and axes, and add any releavant statistical annotation.
These libraries are already loaded: matplotlib, seaborn, pandas, numpy, statsmodels, scipy, sklearn.
df_proc is a Pandas DataFrame and already loaded
Requirements: Output a single ```python code block no longer than ~350 words. Do NOT make any conclusions until I have run your code.
""",

"6": """\
Summarize the main findings so far in a list, and evaluate whether you were successful.
Requirements: Output a single list in markdown format, then add a conclusion at the end
""",

"9": "Quit",
})

code_for_script = """\
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels
import scipy
import scipy.stats as stats
import sklearn

# Print everything
pd.option_context(
    'display.max_rows', 40,
    'display.max_columns', 40,
    'display.precision', 3,
    )

df_proc = pd.read_csv("data/data.csv")
"""

responses_simulated_dict = {
"text_and_code1": f"""\
Here is what I plan to do:
1. Do this
2. Do that
3. Do the other thing

```python
def sum_numbers(numbers):
    return np.sum(numbers)

print("Sum:", sum_numbers([1, 2, 3, 4, 5]))
```
""",

"text_and_code2": f"""\
As you can see, the dataset has 100 rows and 4 columns. The columns are: 'x', 'y', 'z', and 'w'. The first 5 rows are: blah blah blah

```python
def mean_numbers(numbers):
    return np.mean(numbers)

print("Mean:", mean_numbers([1, 2, 3, 4, 5]))
```
""",

"code1": f"""\
```python
def sum_numbers(numbers):
    return sum(numbers)

print(sum_numbers([1, 2, 3, 4, 5]))
```
""",

"text1": f"""\
My plan is to:
1. Do this
2. Do that
3. Do the other thing
""", 

"text2": f"""\
- This dataset has 100 rows and 4 columns.
- The columns are: 'x', 'y', 'z', and 'w'.
- The first 5 rows are: blah blah blah
""",
}