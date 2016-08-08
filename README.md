# Handwritten-Digits-Recognition
Using Machine learning algorithms and classification techiques write the code in R programming language to classify the different classes of handwritten digits (0 - 9) and predict the results based on those techniques using validation datasets.

Optical Handwritten Digits Recognition

Files:

optdigits-orig.names	Original data documentation

optdigits-orig.tra	Training set (original format)


optdigits-orig.cv	Cross validation set (original format)

linear5.awk		Dev code getting ready to process "test.in" (linearize)
test.in			Short file of test data
test.out		Results of running linear5.awk on test.in


linear33b.awk		AWK script (from linear5.awk) adapted to *.tra and *.cv
sep.sed			SED script to separate digits


Note:			*.tra and *.cv need to be edited before running through
			linear33b.awk and sep.sed

			linear33b outputs a second copy of the reference category
