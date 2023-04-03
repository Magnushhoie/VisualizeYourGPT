<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<center><h1>VisualizeYourGPT</h1></center>


What is VisualizeYourGPT?
- Connect [ChatGPT](https://chat.openai.com/chat) (GPT-3.5 or GPT-4.0) to a Python terminal and your dataset
- Let it reason about new hypotheses, execute code and summarize findings in a [Jupyter notebook](https://github.com/Magnushhoie/VisualizeYourGPT/blob/main/output/notebook.ipynb)
- It's free: Just login to ChatGPT through your browser, no API key is needed
- Nb: No code will run without confirmation

## Installation and usage

```bash
# Download and install
git clone https://github.com/Magnushhoie/VisualizeYourGPT/
cd VisualizeYourGPT

# Install requirements
pip -r requirements.txt

# (Make sure you can login to https://chat.openai.com/chat)
# Run on example dataset
python gpt.py
```

## Requirements

- [chatGPT-wrapper](https://github.com/mmabrouk/chatgpt-wrapper)
- [Python 3.8+](https://www.python.org/downloads/)

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
