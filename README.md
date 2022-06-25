# CORD-19
Repository for completing the CORD-19 Challenge

## Files
Project files include sample documents (in json format) in comm_use_subset/ directory. These files are used for training and testing. <br>

cord_training.py - contains methods used for analyzing data and training the ML classifier to recognize relevant pieces of data from the given documents. 

covid_analysis.py - contains methods for formatting data and preparing it for us with classifier.

covid_model.sav - trained model for determining relevant data

### Results/Progress
The project reads from the scientific papers and articles located in the comm_use_subset/ directory. It then determines snippets from each paper that are relevant for COVID analysis based on keywords. 
After the snippets have been extracted, they are labelled based on what keywords are used and the types of data discussed. 
Labelled snippets are then used to train the ML Classifier to predict relevant snippets and locate articles that are potentially viable in incorporating into further analysis of COVID factors. 

The problem thus far has been accurately labelling the extracted data. In order to do so, one needs to read through the extracted text which is very technical and precise. 
Methods are included for automatically labelling based simply on the number of keyword mentions, but this is not very practical or accurate.

![Not Maintained](https://img.shields.io/badge/Maintenance%20Level-Not%20Maintained-yellow.svg)
