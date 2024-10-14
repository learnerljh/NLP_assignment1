
# Experiment Report: Analysis of String Processing Time with Increasing Length

## Objective:

This experiment aims to investigate the relationship between the processing time of a specific operation on strings and their length. We will measure the execution time for "char_count" function while varying the input string's length.

## Methodology:

We conducted this experiment using a Python script that generates random strings with different lengths. The lengths were chosen such that they increased exponentially, starting from 3000 characters up to 300 million characters. For each string length, we measured the execution time taken by a specific function to process the string.

## Results:

The results of our experiment are summarized in the table below. Each row represents the average runtime for processing a string of a particular length.

| String Length | Execution Time (seconds) |
|---------------|--------------------------|
| 3000          | 0.009921                |
| 30000         | 0.019985                |
| 300000        | 0.180171                |
| 3000000       | 1.799481                |
| 30000000      | 17.547169               |
| 300000000     | 176.872086              |


## Conclusion:

In conclusion, our experiment provides evidence that the processing time of a function operating on strings is directly influenced by the length of the input string. The execution time of "char_count" function is nearly proportional to the length of the input.


