# Maximum Entropy Learning with Lexical Scales
## Intro
This is an updated implementation of the maximum entropy model of phonological learning that uses lexical scales described by Linzen et al. (2013) and implemented by Hughto et al. (2019).
The software can learn phonolical mappings, including mappings that include lexical exceptions.  

## Input Format
To run a simulation, the model needs two input files: one showing grammatical information (whose filename should end in "_grammar.txt") and one showing distributional information (whose filename should end in "_dist.txt"). The first line of the distributional file should show how many input-output pairs the model has in its training data in the format "*number* pairs". Every subsequent line should show the following information for each of those pairs:

* The underlying form (surrounded by quotation marks in the first column and bare in the last column)
* The surface form (in the second column, also surrounded by quotation marks)
* The frequency of the pair in your training data (should be a whole number and appear in the third column)
* A numerical label for each input-output pair in the fourth column (e.g., 1 for the first pair, 2 for the second, etc.)

Columns can be separated by any kind of whitespace and there shouldn't be any lines after the final pair is described. 

For the grammar file, you first specify the number of constraints the model will use. The first line should have the format "*number* constraints:". Then you list each constraint in the format: 
"constraint \[*N*\] "*constraintName*" 1 100 *constraintName*". Note that the "1 100" has no meaning here--it's meant to make these files compatible with other models of phonological learning. Also note that the square brackets here *should* appear in the actual file and "N" represents each constraint's number (where constraints are numbered 1, 2, 3... etc.). Next a line is skipped and then the total number of tableaux the model needs are given in the format "*number* tableaus". This number of tableaux should match the number of pairs from the distributional file. Each tableau should then be listed, first with a line specifying its input in the format "input \[*N*\]: "*UR*" *numberOfOutputCandidates*. Where N is tableau's number (and they're numbered 1, 2, 3... etc.), UR is the input to the tableau, and the number of candidates represents every output that you want to be possible, given that input. After the input line, the rest of a tableau's lines should each represent one candidate and that candidate's constraint violations, in the format "candidate \[*N*\]: "*SR*" *violations*. Where N is candidate's number (and they're numbered 1, 2, 3... etc.), SR is the candidate's phonological form, and violations are a space-delimited list of the constraint violations, in the order that the constraints were given at the top of the file. After all tableaux have been specified, the file can end.

Examples of both the distribution and grammar files can be viewed in the Training_Data directory of this repo.

## Output Format
After running the model, it will save a number of files. These files are timestamped, so you don't need to worry about overwriting files from previous simulations. Seven different kinds of output are created:

* A "best epoch" file showing information about the learning update with the lowest loss
* A "generalization curve" file showing how the model generalizes to novel data over the course of learning. Note that "novel" here just means "the same data the model saw in its training data, but without the effect of its lexical scales".
* A "learning curve" file showing the model's accuracy on the training data over the course of learning.
* A "nonce probs" file showing the how the model generalizes to novel data at the end of learning. Note that "novel" here just means "the same data the model saw in its training data, but without the effect of its lexical scales".
* An "output weights" file showing you the model's constraint weights and scale values at the end of learning.
* A "td prob curve" file showing you the probability of each training datum over the course of learning.
* A "td probs" file showing you the probability of each training datum at the end of learning.

Example files can be viewed in the Raw_Output directory of this repo.

## Using the Software
To use the learner:
* First, you'll need a directory called "Training_Data" with all the input files.
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
