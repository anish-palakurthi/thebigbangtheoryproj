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

"The Big Bang Theory" is a popular TV show known for its witty and humorous dialogue. In this project, we leverage K-means clustering to group similar scripts and LDA to extract underlying topics from the episodes. By combining these techniques, we create an unsupervised model that can discover patterns and relationships within the dialogue data.

## Installation

To use this project, you'll need to have Python installed. Clone this repository and install the required dependencies using the following commands:

```
git clone https://github.com/your_username/thebigbangtheoryproj.git
cd your_project_path
pip install -r requirements.txt
```


## Usage

1. Make sure you have installed all the dependencies as mentioned in the installation section.
2. Run kclustering.py to perform K-means clustering and lda.py to conduct LDA analysis on the scripts data (generally more accurate as LDA allows for multiple topics per episode but requires slightly more training time.)
3. Explore the discovered topics and cluster assignments to gain insights into the episodes.
4. Use the generated model to recommend episodes to users based on their interests.

## Results

We present the following results from our analysis:

- Identified X distinct clusters of episodes based on dialogue patterns.
- Extracted Y latent topics, such as "science jokes," "relationship humor," and "geek culture," providing a deeper understanding of the content.
- Successfully generated episode recommendations for users, showcasing the practicality of the unsupervised model.

## Contributing

Contributions to this project are welcome! If you find any issues or have ideas to enhance the model's performance, please feel free to open an issue or submit a pull request. We appreciate your input!

## License

This project is licensed under the [MIT License](LICENSE).
