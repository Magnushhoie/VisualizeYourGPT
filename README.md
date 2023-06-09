<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<center><h1>VisualizeYourGPT</h1></center>

<p align="center">
  <img alt="VisualizeYourGPT"  src="https://github.com/Magnushhoie/VisualizeYourGPT/blob/main/images/gpt.jpg?raw=true" width="250"/
</p>


What is VisualizeYourGPT?
- Connect [ChatGPT](https://chat.openai.com/chat) (GPT-3.5 or GPT-4.0) to a Python terminal and your dataset
- Let it reason about new hypotheses, execute code and summarize findings in a [Jupyter notebook](https://github.com/Magnushhoie/VisualizeYourGPT/blob/main/output/notebook.ipynb)
- It's free: Just login to ChatGPT through your browser, no API key is needed
- Nb: No code will run without confirmation, unless automatic mode is set

```bash
# Download and install
git clone https://github.com/Magnushhoie/VisualizeYourGPT/
cd VisualizeYourGPT

# Install requirements
pip install -r requirements.txt

# Setup ChatGPT, and login through the Playwright browser
chatgpt install

# Run on example dataset
python gpt.py --csv_file data/data.csv

# Nb: You can edit the choice of prompts in your_prompts.py
```

## Requirements

- [chatGPT-wrapper](https://github.com/mmabrouk/chatgpt-wrapper)
- [Python 3.8+](https://www.python.org/downloads/)

## Example workflow:

1. Introduce context and give dataset description (automatic)
2. **Option 6**: "6: Run a quantitative only data analysis to support the question."
3. Let GPT output code, run it and return output, then ask to summarize findings
3. **Option 7**: "7: Graph 1-2 supporting figure(s)."
4. Let GPT output code, run it and return output, then ask to summarize findings
5. **Option 5**: "5: Pick a scientific question and plan how you decide to tackle it"
6. Repeat steps 2-5.

## Visualizing graphs in the terminal (MacOS)

```bash
# Install itermplot https://github.com/daleroberts/itermplot
pip install itermplot==0.5

# Set these settings in your terminal before running gpt.py (recommend NOT putting in .bashrc)
# Use itermplot for Matplotlib plotting
MPLBACKEND="module://itermplot"
# Dark theme
ITERMPLOT=rv
```

## Documentation

```
USAGE:     python gpt.py --csv_file <FILE> --model <MODEL>

    Examples:

    # Run VisualizeYourGPT on pre-processed dataset 'data.csv'
    python gpt.py --csv_file data/data.csv

    # Output notebook, insights and script stored in output/

Connect dataset and executable Python script to ChatGPT

optional arguments:
  -h, --help           show this help message and exit
  --csv_file CSV_FILE  Input, pre-processed dataset (CSV file)
  --model MODEL        ChatGPT model to use (legacy-free, legacy-paid, gpt4)
  --simulate           Simulate ChatGPT responses
  -v, --verbose        Verbose mode
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Magnushhoie/VisualizeYourGPT.svg?style=for-the-badge
[contributors-url]: https://github.com/Magnushhoie/VisualizeYourGPT/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Magnushhoie/VisualizeYourGPT.svg?style=for-the-badge
[forks-url]: https://github.com/Magnushhoie/VisualizeYourGPT/network/members
[stars-shield]: https://img.shields.io/github/stars/Magnushhoie/VisualizeYourGPT.svg?style=for-the-badge
[stars-url]: https://github.com/Magnushhoie/VisualizeYourGPT/stargazers
[issues-shield]: https://img.shields.io/github/issues/Magnushhoie/VisualizeYourGPT.svg?style=for-the-badge
[issues-url]: https://github.com/Magnushhoie/VisualizeYourGPT/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/Magnushhoie/VisualizeYourGPT/blob/master/LICENSE.txt
