# Iris Flowers

## Dataset

**Title: Iris Plants Database**

Updated Sept 21 by C. Blake - Added discrepancy information

**Sources:**

- Creator: R.A. Fisher
- Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
- Date: July, 1988

**Past Usage:**

Publications: too many to mention!!! Here are a few:

1. Fisher,R.A. "The use of multiple measurements in taxonomic problems"
   Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions
   to Mathematical Statistics" (John Wiley, NY, 1950).
   Duda,R.O., & Hart,P.E. (1973) Pattern Classification and Scene Analysis.
   (Q327.D83) John Wiley & Sons. ISBN 0-471-22361-1. See page 218.
2. Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
   Structure and Classification Rule for Recognition in Partially Exposed
   Environments". IEEE Transactions on Pattern Analysis and Machine
   Intelligence, Vol. PAMI-2, No. 1, 67-71.
   -- Results:
   -- very low misclassification rates (0% for the setosa class)
3. Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule". IEEE
   Transactions on Information Theory, May 1972, 431-433.
   -- Results:
   -- very low misclassification rates again
4. See also: 1988 MLC Proceedings, 54-64. Cheeseman et al's AUTOCLASS II
   conceptual clustering system finds 3 classes in the data.

**Relevant Information:**

- This is perhaps the best known database to be found in the pattern
  recognition literature. Fisher's paper is a classic in the field
  and is referenced frequently to this day. (See Duda & Hart, for
  example.) The data set contains 3 classes of 50 instances each,
  where each class refers to a type of iris plant. One class is
  linearly separable from the other 2; the latter are NOT linearly
  separable from each other.
- Predicted attribute: class of iris plant.
- This is an exceedingly simple domain.
- This data differs from the data presented in Fishers article
  (identified by Steve Chadwick, spchadwick@espeedaz.net )
  The 35th sample should be: 4.9,3.1,1.5,0.2,"Iris-setosa"
  where the error is in the fourth feature.
  The 38th sample: 4.9,3.6,1.4,0.1,"Iris-setosa"
  where the errors are in the second and third features.

**Number of Instances:**
150 (50 in each of three classes)

**Number of Attributes:**
4 numeric, predictive attributes and the class

**Attribute Information:**

1. Sepal length in cm
2. Sepal width in cm
3. Petal length in cm
4. Petal width in cm
5. Class:
   - Iris Setosa
   - Iris Versicolour
   - Iris Virginica

**Missing Attribute Values:**
None

**Summary Statistics:**

|              | Min | Max | Mean | SD   | Class Correlation |
| ------------ | --- | --- | ---- | ---- | ----------------- |
| Sepal length | 4.3 | 7.9 | 5.84 | 0.83 | 0.7826            |
| Sepal width  | 2.0 | 4.4 | 3.05 | 0.43 | -0.4194           |
| Petal length | 1.0 | 6.9 | 3.76 | 1.76 | 0.9490 (high!)    |
| Petal width  | 0.1 | 2.5 | 1.20 | 0.76 | 0.9565 (high!)    |

**Class Distribution:**
33.3% for each of 3 classes.
