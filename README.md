# Lyrics2Chords
### Using Machine Learning To Facilitate Creativity


[Lyrics2Chords](http://thrasher.website) is a web application designed for budding songwriters. Songwriting is a cumbersome process that requires an understanding of how to deliver a message both lyrically and acoustically. A lyrically sad song for instance can be accentuated by sad sounding harmonies. However for those new to songwriting arranging music to support the lyrical sentiments can be a trial and error process. 

Lyrics2Chords addresses this problem by using machine learning to predict the [emotional valence](https://en.wikipedia.org/wiki/Valence_(psychology)) of lyrics inputted by the user to suggest chord progressions
that are able to acoustically voice the appropriate message. 

This project was built over a period of a three weeks at Insight Data Science as part of my data science fellowship project. The data was collected from  APISeeds Lysics, Spotify, and Hook Theory. The backed of the application was written in Python 3.5 using a variety of machine learning and scientific computing packages.

This repository contains saved trained models and all of the code dependencies for the web application. 

The NLP of the lyrical data and construction of the emotional sentiment classifier may be found in [Emotional-Valence-Classifier.ipynb](https://github.com/rkthrasher/Lyrics2Chords/blob/master/Emotional-Valence-Classifier.ipynb).
