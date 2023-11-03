# Maximum Entropy Learning with Lexical Scales
## Intro
This is an updated implementation of the maximum entropy model of phonological learning that uses lexical scales described by Linzen et al. (2013) and implemented by Hughto et al. (2019).
The software can learn phonolical mappings, including mappings that include lexical exceptions.  

## Using the Software
To use the learner:
* First, you'll need a directory called "Training_Data" with all the training data files.
* Then, you'll need a directory called "Raw_Output" where the learner will write its output files.
* Then you can run scaled_weights_MaxEnt.py
* Command line parameters (all but the last two are obligatory):
	* Learning algorithm? Choose from the set {GD, GD_CLIP, and L-BFGS-B} for "gradient descent", "gradient descent with clipping" (preferred value), and "L-BFGS-B", respectively.
	* Negative constraint weights? Choose either 0 (no negative constraint weights allowed; preferred value) or 1 (negative constraint weights allowed)
	* Negative scales? Choose either 0 (no negative scales allowed; preferred value) or 1 (negative scales allowed)
	* Priors? Either 1 value showing the strenght of the prior on scales and weights (e.g. ".25") or 2 comma-delimited values showing first the constraints' prior strength then the scales' (e.g. ".01,.25")
	* L2 Prior? If this is 1 you'll use an L2 prior and if it's 0 you'll use an L1 prior. L2 can't be used with GD_CLIP and L1 can't be used with GD.
	* Initial weights/scales? Either 1 value showing the initial value for scales and weights (e.g. "1.0") or 2 comma-delimited values showing first the constraints' initial value then the scales' (e.g. "1,10")
	* Random weights? If this is 1, the previous argument is ignored (but does still need a value) and instead, all weights and scales are randomly initialized
	* Learning rate (for GD and GD_CLIP only)? Either 1 value showing the learning rate for scales and weights (e.g. ".05") or 2 comma-delimited values showing first the constraints' learning rate then the scales' (e.g. ".01,.05")
	* Epochs (for GD and GD_CLIP only)? How many full passed through the data will the model make?
	* Language? Arbitrary label used in the input and output files
* Dependencies:
	* numpy
	* sys
	* scipy.optimize
	* datetime
	* re
	* math

## Authors
The orginal version of this code is published [here](https://github.com/chughto/Lexically-Scaled-MaxEnt) and was written by:

* Coral Hughto
* Andrew Lamont
* Brandon Prickett
* Gaja Jarosz

In the years since its initial publication, the software has been further developed with feedback and support from:

* Seoyoung Kim
* Maggie Baird
* Max Nelson
* Cerys Hughes

## References:
* Hughto, Coral; Lamont, Andrew; Prickett, Brandon; and Jarosz, Gaja (2019). "Learning Exceptionality and Variation with Lexically Scaled MaxEnt," *Proceedings of the Society for Computation in Linguistics: Vol. 2*, Article 11. DOI: https://doi.org/10.7275/y68s-kh12 Available at: https://scholarworks.umass.edu/scil/vol2/iss1/11
* Tal Linzen, Sofya Kasyanenko, and Maria Gouskova (2013). "Lexical and phonological variation in Russian prepositions," *Phonology, 30(3)*:453–515.
