# Work Report

## Information

- Name: <ins> DAS, NEHA </ins>
- GitHub: <ins> NehaDas25 </ins>


## Features

- Not Implemented:
  - what features have been implemented

<br><br>

- Implemented:
  - PART 1 - Importing the Data
  - PART 1.3 - Understanding the Iterator
    - Exercise 1 - data_generator
      - Implemented a data_generator() function that takes in Q1, Q2, batch_size, pad=1, shuffle=True.
      - Returns a batch of size batch_size in the following format  ([𝑞11,𝑞12,𝑞13,...],[𝑞21,𝑞22,𝑞23,...]). The tuple consists of two arrays and each array has batch_size questions.
      - The command next(data_generator)returns the next batch. This iterator returns the data in a format that you could directly use in your model when computing the feed-forward of your algorithm. This iterator returns a pair of arrays of questions.
      - Here are some things that needs to written in while loop:
        1. While true loop.
        2. if index >= len_Q1, set the idx to  0.
        3. The generator should return shuffled batches of data. To achieve this without modifying the actual question lists, a list containing the indexes of the questions is created. This list can be shuffled and used to get random batches everytime the index is reset.
        4. Get questions at the `question_indexes[idx]` position in Q1 and Q2 and append elements of 𝑄1 and 𝑄2 to input1 and input2 respectively.
        5. if len(input1) == batch_size, determine max_len as the longest question in input1 and input2. Ceil max_len to a power of 2(for computation purposes) using the following command: max_len = 2**int(np.ceil(np.log2(max_len))).
        6. Pad every question by vocab['<PAD>'] until you get the length max_len.
        7. Used yield to return input1, input2.
        8. Reset input1, input2 to empty arrays at the end (data generator resumes from where it last left).
      - This passed all the test cases.

  - PART 2 - Defining the Siamese Model
  - PART 2.1 - Understanding Siamese Network
    - Exercise 2 - Siamese
      - Implemented the model Siamese() function that takes in vocab_size=41699, d_model=128, mode='train'. 
      - To implement this model, google's trax package has been used.
      - The following packages when constructing the model has been used here are:
        1. *tl.Serial()*: Combinator that applies layers serially.Here,the layers are passed as arguments to Serial, separated by commas like tl.Serial(tl.Embeddings(...), tl.Mean(...), tl.Dense(...), tl.LogSoftmax(...)).
        2. *tl.Embedding(vocab_size, d_feature)*: Initializes the embedding. In this case it is the size of the vocabulary by the dimension of the model.vocab_size is the number of unique words in the given vocabulary.d_feature is the number of elements in the word embedding (some choices for a word embedding size range from 150 to 300, for example).   
        3. *tl.LSTM()*: The LSTM layer. It leverages another Trax layer called LSTMCell. The number of units should be specified and should match the number of elements in the word embedding.tl.LSTM(n_units) Builds an LSTM layer of n_units.
        4. *tl.Mean()*: Computes the mean across a desired axis. Mean uses one tensor axis to form groups of values and replaces each group with the mean value of that group. tl.Mean(axis=1) mean over columns.
        5. *tl.Fn()*: Layer with no weights that applies the function f, which should be specified using a lambda syntax. 𝑥-> This is used for cosine similarity.
        tl.Fn('Normalize', lambda x: normalize(x)) Returns a layer with no weights that applies the function f
        6. tl.parallel(): It is a combinator layer (like Serial) that applies a list of layers in parallel to its inputs.
      - This passed all the unit-test cases as well.

  - PART 2.2 - Hard Negative Mining
    - Exercise 3 - TripletLossFn
      - Implemented the function TripletLossFn() that takes in v1, v2, margin=0.25.
      - Used fastnp.dot to calculate the similarity matrix 𝑣1𝑣2^T of dimension batch_size x batch_size.
      - Calculated new batch size.
      - Used fastnp to grab all postive diagonal entries in scores.
      - Subtracted fastnp.eye(batch_size) out of 1.0 and do element-wise multiplication with scores and store it in negative_zero_on_duplicate.
      - Used fastnp.sum on negative_zero_on_duplicate for axis=1 and divide it by (batch_size - 1).
      - Created a composition of two masks: the first mask to extract the diagonal elements,the second mask to extract elements in the negative_zero_on_duplicate matrix that are larger than the elements in the diagonal. Store it in mask_exclude_positives.
      - Multiply mask_exclude_positives with 2.0 and subtract it out of negative_zero_on_duplicate.
      - Taken the row by row, max of negative_without_positive,i.e.,**closest_negative = negative_without_positive.max(axis = 1)**.
      - Computed fastnp.maximum among 0.0 and A where A = subtract positive from margin and add closest_negative, that is **triplet_loss1 = fastnp.maximum(0.0, margin - positive + closest_negative)**.
      - Computed fastnp.maximum among 0.0 and B where B = subtract positive from margin and add mean_negative, that is **triplet_loss2 = fastnp.maximum(0.0, margin - positive + mean_negative)**.
      - Added the two losses together and take the  fastnp.sum of it.
      - This passed all the unit-test cases as well.

  - PART 3 - Training
  - PART 3.1 - Training the Model 
    - Exercise 4 - train_model
      - Implemented the train_model() to train the neural network that takes Siamese, TripletLoss, train_generator, val_generator, output_dir='model/' as inputs.  
      - Created TrainTask and EvalTask.
      - training task that uses the train data generator defined in the cell 
        1. labeled_data=train_generator
        2. loss_layer=TripletLoss()
        3. optimizer = trax.optimizers.Adam(0.01) 
        4. lr_schedule=trax.lr.warmup_and_rsqrt_decay(400, 0.01)
      - evaluation task that uses the validation data generator defined in the cell
        1. labeled_data=val_generator
        2. metrics=[TripletLoss()]
      - Created the trainer object by calling trax.supervised.training.Loop and passed in the following:
        1. model = Siamese()
        2. train_task
        3. eval_tasks=[eval_task]
        4. output_dir=output_dir
      - This passed all the unit-test cases as well.
    
  - PART 4 - Evaluation
  - PART 4.2 - Classify
    - Exercise 5 - classify
      - Implemented a function called classify() that takes in test_Q1, test_Q2, y, threshold, model, vocab, data_generator=data_generator, batch_size=64.
      - Looped through the incoming data in batch_size chunks
      - Used the data generator to load q1, q2 a batch at a time and set shuffle=False using next().
      - Copied a batch_size chunk of y into y_test
      - Computed v1, v2 using the model
      - for each element of the batch
        - Computed the cos similarity of each pair of entries, `v1[j]`,`v2[j]`
        - determine if `d` > threshold
        - increment accuracy if that result matches the expected results (`y_test[j]`)
      - Computed the final accuracy and return it.
      - This passed all the unit-test cases as well.

  - PART 5 - Testing with your Own Questions
    - Exercise 6 - predict
      - Implemented a function predict() that takes in two questions, the model, and the vocabulary, data_generator(), verbose=False.
      - Tokenized your question using nltk.word_tokenize.
      - Created Q1,Q2 by encoding your questions as a list of numbers using vocab.
      - padded Q1,Q2 with next(data_generator([Q1], [Q2],1,vocab[''])).
      - Used model() to create v1, v2.
      - Computed the cosine similarity (dot product) of v1, v2.
      - Computed res by comparing d to the threshold.
      - This passed all the unit-test cases as well.


  
<br><br>

- Partly implemented:
  - w4_unittest.py was also not implemented as part of assignment to pass all unit-tests for the graded functions().

<br><br>

- Bugs
  - No bugs

<br><br>


## Reflections

- Assignment is quiet knowledgeable. Gives a thorough understanding of the LSTM, Siamese Networks, Question duplication, Hard Negative Mining.


## Output

### output:

<pre>
<br/><br/>
Out[2] -

Number of question pairs:  404351
  id qid1	qid2	          question1	                     question2	                    is_duplicate
0	 0	1 	2	   What is the step by step guide to..	 What is the step by step guide to ...	      0
1  1  3   4    What is the story of Kohinoor...      What would happen if the Indian govern...    0
2  2  5   6    How can I increase the speed of my intern..  How can Internet speed be increas..   0
3  3  7   8    Why am I mentally very lonely?How can ..  Find the remainder when[math]23^{24}..   0
4  4  9   10   Which one dissolve in water quikly sugar.. Which fish would survive in salt water? 0

Out[3] -

Train set: 300000 Test set: 10240

Out[4] -

number of duplicate questions:  111486
indexes of first ten duplicate questions: [5, 7, 11, 12, 13, 15, 16, 18, 20, 29]

Out[5] - 

Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me?
I'm a triple Capricorn (Sun, Moon and ascendant in Capricorn) What does this say about me?
is_duplicate:  1

Out[7] -

TRAINING QUESTIONS:

Question 1:  Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me?
Question 2:  I'm a triple Capricorn (Sun, Moon and ascendant in Capricorn) What does this say about me? 

Question 1:  What would a Trump presidency mean for current international master’s students on an F1 visa?
Question 2:  How will a Trump presidency affect the students presently in US or planning to study in US? 

TESTING QUESTIONS:

Question 1:  How do I prepare for interviews for cse?
Question 2:  What is the best way to prepare for cse? 

is_duplicate = 0 

Out[9] - The length of the vocabulary is:  36268

Out[10] - 

1
2
0

Out[12] -

Train set has reduced to:  111486
Test set length:  10240

Out[14] -

first question in the train set:

Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me? 

encoded version:
[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] 

first question in the test set:

How do I prepare for interviews for cse? 

encoded version:
[32, 38, 4, 107, 65, 1015, 65, 11509, 21] 

Out[15] -

Number of duplicate questions:  111486
The length of the training set is:   89188
The length of the validation set is:  22298

Out[17] -

First questions  :  
 [[  30   87   78  134 2132 1981   28   78  594   21    1    1    1    1
     1    1]
 [  30   55   78 3541 1460   28   56  253   21    1    1    1    1    1
     1    1]] 

Second questions :  
 [[  30  156   78  134 2132 9508   21    1    1    1    1    1    1    1
     1    1]
 [  30  156   78 3541 1460  131   56  253   21    1    1    1    1    1
     1    1]] 

Expected Output:

First questions  :  
 [[  30   87   78  134 2132 1981   28   78  594   21    1    1    1    1
     1    1]
 [  30   55   78 3541 1460   28   56  253   21    1    1    1    1    1
     1    1]] 

Second questions :  
 [[  30  156   78  134 2132 9508   21    1    1    1    1    1    1    1
     1    1]
 [  30  156   78 3541 1460  131   56  253   21    1    1    1    1    1
     1    1]]

Out[18] - All tests passed

Out[20] - 

Parallel_in2_out2[
  Serial[
    Embedding_41699_128
    LSTM_128
    Mean
    Normalize
  ]
  Serial[
    Embedding_41699_128
    LSTM_128
    Mean
    Normalize
  ]
]
Expected output:

Parallel_in2_out2[
  Serial[
    Embedding_41699_128
    LSTM_128
    Mean
    Normalize
  ]
  Serial[
    Embedding_41699_128
    LSTM_128
    Mean
    Normalize
  ]
]

Out[21] - All tests passed

Out[23] - 

Triplet Loss: 0.7035077

Expected Output:

Triplet Loss: ~ 0.70

Out[24] - All tests passed

Out[26] - 

train_Q1.shape  (89188,)
val_Q1.shape    (22298,)

Out[28] - 

Step      1: Total number of trainable weights: 5469056
Step      1: Ran 1 train steps in 2.25 secs
Step      1: train TripletLoss |  127.74435425

Step      1: eval  TripletLoss |  127.74793243

Out[29] -  All tests passed

Out[32] - 

Accuracy 0.74423828125
Expected Result
Accuracy ~0.74

Out[33] - All tests passed

Out[36] - 

Q1  =  [[  443  1145  3159  1169    78 29017    21     1]] 
Q2  =  [[  443  1145    60 15302    28    78  7431    21]]
d   =  0.5364446
res =  False
False
Expected output
If input is:

question1 = "Do they enjoy eating the dessert?"
question2 = "Do they like hiking in the desert?"
Output (d may vary a bit):

Q1  =  [[  443  1145  3159  1169    78 29017    21     1]] 
Q2  =  [[  443  1145    60 15302    28    78  7431    21]]
d   =  0.53644466
res =  False
False

Out[37] - All tests passed


<br/><br/>
</pre>
