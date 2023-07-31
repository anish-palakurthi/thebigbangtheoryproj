# The Big Bang Theory NLP Episode Recommender

## Overview

This project involves using K-means clustering and LDA (Latent Dirichlet Allocation) to train an unsupervised model on the scripts of the (greatest) TV show "The Big Bang Theory." The main objective is to model topics in episodes and facilitate user episode recommendations.

## Table of Contents

- [The Big Bang Theory NLP Episode Recommender](#the-big-bang-theory-nlp-episode-recommender)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Results](#results)
  - [Contributing](#contributing)
  - [License](#license)

## Description

"The Big Bang Theory" is a popular (the best) TV show known for its witty dialogue and humorous relationships. In this project, we leverage K-means clustering to group similar scripts and LDA to extract underlying topics from the episodes. By combining these techniques, we create an unsupervised model that can discover patterns and relationships within the dialogue data, which we can leverage to recommend episodes to a user that they may enjoy based on their input. As someone who has seen all 12 seasons of this show 3 times, I wanted to be able to find my favorite episodes to rewatch without having to restart the series.

## Installation

To use this project, you'll need to have Python installed. Clone this repository and install the required dependencies using the following commands:

```
git clone https://github.com/your_username/thebigbangtheoryproj.git
cd your_project_path
pip install -r requirements.txt
```


## Usage

1. Make sure you have installed all the dependencies as mentioned in the installation section.
2. Run ```kclustering.py``` to perform K-means clustering and ```lda.py``` to conduct LDA analysis on the scripts data,
   1. LDA processing can take significantly longer than K-means clustering but allows for increased recommendation robustness as it includes multiple topical distributions per episode rather than classifying them under cut lines.
3. Explore the discovered topics and cluster assignments to gain insights into the episodes.
4. Use the generated model to recommend episodes to users based on their interests.

## Results

We present the following results from our analysis:

- Identified and visualized distinct clusters of episodes based on dialogue patterns.
- Extracted latent topics, such as "science jokes," "relationship humor," and "geek culture," providing a deeper understanding of the content.
- Successfully generated episode recommendations for users, showcasing the practicality of the unsupervised model.

## Contributing

Contributions to this project are welcome! If you find any issues or have ideas to enhance the model's performance, please feel free to open an issue or submit a pull request. I appreciate your input!
This project can easily be expanded to any other television show, song lyrics, etc. Edit the json files or adjust how data is loaded into the model as you see fit!

## License

This project is licensed under the [MIT License](LICENSE).
