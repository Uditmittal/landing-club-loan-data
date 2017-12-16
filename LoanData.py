
# coding: utf-8

# In[1]:


#Importing library Pandas and numpy 
#Pandas depends upon and interoperates with NumPy, the Python library for fast numeric array computations
#We can use for Panda DataFrame But for Easy Visualization we use Panda
import pandas as pd
import numpy as np


# In[4]:


#Loading CSV via using Panda. this will create a Panda Dataframe which help us to creating the EDA(Exploratory Data Analysis).
loan = pd.read_csv('/home/udit/Downloads/loan.csv')


# In[3]:


#This command will help for Checking checking the data types
loan.dtypes


# In[4]:


#Print some Lines for checking the values
loan.head()


# <h1>Here we see grade wise count to check which is the highest grade here we see B grade have more people.</h1>

# In[5]:


# Here we take a group by of grade for checking the count of a purticular columns.
loan.groupby('grade').count()


# In[6]:


# Here we take a group by of loan amount for checking the count of a purticular columns. 
loan.groupby('loan_amnt').count()


# <h1>Here we see generally people take 5K to 20K is most of the time</h1>

# In[7]:


# Here we make the graph for cheking the history of a purticular loan amount column.

#Import materplot library for making the graph
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# Figsize is creates an inch-by-inch image, which will be 80-by-80 pixels unless you also give a different dpi argument.
plt.rc("figure", figsize=(8, 5))

#hist is a function in numpy module which helping us to creating the histogram of a column
loan["loan_amnt"].hist()

# Setting the title 
plt.title("distribution of loan amount")


# In[3]:


# Here we take a group by of addr_state for checking the count of a purticular columns. 
loan.groupby("addr_state").count()
# This simply says CA state have more loan holder .


# In[44]:


# Here we take a group by of addr_state for checking the count of a purticular columns and take id column. 
df_group = loan.groupby('addr_state', as_index=False)['id'].count()


# In[11]:


# Now Sort the Values of id column in asecending order
df_group.sort_values(['id'], ascending=[True], inplace=True)
# Resetting index 
df_group = df_group.reset_index(drop=True)


# <h1>Here we see in the graph State CA have more people</h1>

# In[12]:


# Creating Bar chart in Index and id columns
plt.bar(df_group.index,df_group.id/100000)
# Creating X Sticks for this graph
plt.xticks(df_group.index,df_group.addr_state)
N = 3
params = plt.gcf()

#Ploat size is taking in inches
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*N, plSize[1]*N) )
# showing the plot
plt.show()


# <h3>Here we represent state wise income, where we see DC state have more income people</h3>

# In[13]:


# Now we take mean of Address State columns for showing the annual income 
df_group = loan.groupby('addr_state', as_index=False).mean()

# Taking annual income data and apply sorting in ascending order
df_group.sort_values(['annual_inc'], ascending=[True], inplace=True)

#resetting indexing
df_group = df_group.reset_index(drop=True)

#creating Bar chart
plt.bar(df_group.index,df_group.annual_inc/1000)

#Making X sticks for making the group and index
plt.xticks(df_group.index,df_group.addr_state)
N = 3
params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*N, plSize[1]*N) )
# Visuakize the plots
plt.show()

# This help us to finding the loan holder


# <h1>Here we see grade wise interest rate, G grade people may give more interest rate</h1>

# In[45]:


# Now we take mean of grade columns for showing the int_rate
df_group = loan.groupby('grade', as_index=False).mean()

#This will show the intrest rate and order is ascending
df_group.sort_values(['int_rate'], ascending=[True], inplace=True)
df_group = df_group.reset_index(drop=True)
#Ploting the grade Bars
plt.bar(df_group.index,df_group.int_rate,align="center")
plt.xticks(df_group.index,df_group.grade)
N = 3
params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*N, plSize[1]*N) )

#Print Plots 
plt.show()


# In[23]:


#Here we take group by interest rate for counting the numbers
loan.groupby("int_rate").count()


# <h3>Here we see the distribution of Interest rate where we see 10-15% is standerd Interest rate<h3>

# In[8]:


import warnings # ignore warnings 
warnings.filterwarnings("ignore")
import seaborn as sns

# Density Distribution of Interest Rate
sns.set_style("whitegrid")
ax=sns.distplot(loan.int_rate, color="g")
ax.set(xlabel='Interest Rate %', 
       ylabel='Distribution in %',title='Density Plot of Interest Rate')

plt.legend();

#This shows the density plot on behalf of interst rate


# <h3>Here we see grade wise interest rate where we analysis than lower grade have lower interest rate and in above graph we also see B grade member give approx 10-15% interest rate</h3>

# In[19]:


#A plot of sub grade and rate shows that they are also correlated
df_group = loan.groupby('sub_grade', as_index=False).mean()
df_group.sort_values(['int_rate'], ascending=[True], inplace=True)
df_group = df_group.reset_index(drop=True)
plt.bar(df_group.index,df_group.int_rate,align="center")
plt.xticks(df_group.index,df_group.sub_grade)
N = 3
params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*N, plSize[1]*N) )
plt.show()


# <h3>Here we see B grade have more number of people. Grade is decided on behalf of Credit score and income.
#  Here also we represent count on grade</h3>

# In[32]:


sns.countplot(loan['grade'])


#   <h5> Here we see more than 50% people pay loan amount in 3 years</h5>

# In[34]:


sns.countplot(loan['term'])


# <h3>Here We see the current status of loan more than 50% of loan application is in the under processing.</h3>

# In[35]:


loan_status_distribution=pd.DataFrame(loan['loan_status'].value_counts())
loan_status_distribution.reset_index(inplace=True)
loan_status_distribution.columns=['Loan_Status_Category','Number of applicants']
loan_status_distribution


# <h1>Represent Data in the python list format along with unique key</h2>

# In[11]:


pd.DataFrame(zip(range(0,loan.shape[0]),loan.values.tolist()),columns=['uniqueid','column_values'])


# In[24]:


loan['title'].value_counts().head(n=10)


# In[26]:


sns.set_style('whitegrid')
plt.figure(figsize=(10,10))
sns.countplot(x='loan_status',data=loan,palette='cubehelix')
sns.despine(top=True,right=True)
plt.xticks(rotation=90)
plt.show()


# <h1>Fill Median Where Annual Income is NA</h1>

# In[5]:


loan['annual_inc'] = loan['annual_inc'].fillna(loan['annual_inc'].median())


# <h3>After Filling NA with Median </h3>

# In[8]:


# Now we take mean of Address State columns for showing the annual income 
df_group = loan.groupby('addr_state', as_index=False).mean()

# Taking annual income data and apply sorting in ascending order
df_group.sort_values(['annual_inc'], ascending=[True], inplace=True)

#resetting indexing
df_group = df_group.reset_index(drop=True)

#creating Bar chart
plt.bar(df_group.index,df_group.annual_inc/1000)

#Making X sticks for making the group and index
plt.xticks(df_group.index,df_group.addr_state)
N = 3
params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*N, plSize[1]*N) )
# Visuakize the plots
plt.show()

# This help us to finding the loan holder


# In[11]:


#For title, i will impute 'title not given' since there are so few missing.
loan['title'] = np.where(loan['title'].isnull(), 0, loan['title'])

#delinq_2yrs has 29 missing observations and I think we can replace those with zero, giving lendors the benefit of the doubt they wouldn't forget someone deliquent.
loan['delinq_2yrs'] = np.where(loan['delinq_2yrs'].isnull(), 0, loan['delinq_2yrs'])

#inq_last_6mths will be fixed in a similar manner.
loan['inq_last_6mths'] = np.where(loan['inq_last_6mths'].isnull(), 0, loan['inq_last_6mths'])


# <h1>Number of null values in each column </h1>

# In[13]:


null_counts = loan.isnull().sum()
print("Number of null values in each column:\n{}".format(null_counts))


# In[15]:


mapping_dict = {
    "emp_length": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0
 
    },
    "grade":{
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
        "E": 5,
        "F": 6,
        "G": 7
    }
}

filtered_loans = loan.replace(mapping_dict)
filtered_loans[['emp_length','grade']].head()


# In[16]:


nominal_columns = ["home_ownership", "verification_status", "purpose", "term"]
dummy_df = pd.get_dummies(filtered_loans[nominal_columns])
filtered_loans = pd.concat([filtered_loans, dummy_df], axis=1)
filtered_loans = filtered_loans.drop(nominal_columns, axis=1)


# In[17]:


loan.head()


# In[18]:


print("Data types and their frequency\n{}".format(filtered_loans.dtypes.value_counts()))


# <h3>Timestamp Handling</h3>

# In[39]:


loan['earliest_cr_line']

# Creating date on behalf of date
loan['created_date'] = pd.to_datetime(loan['earliest_cr_line'])

loan['created_date'] += pd.to_timedelta(loan.groupby(level=0).cumcount(), unit='H')

loan['month'] = pd.DatetimeIndex(loan['created_date']).month

# Creting Month Here 
loan['month']


# In[5]:


# Importing Pyspark ML Library
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType


# In[6]:


# Create Tokenizer Parameters
tokenizer = Tokenizer(inputCol="desc", outputCol="words")


# In[7]:


# Regex Tokenizer
regexTokenizer = RegexTokenizer(inputCol="desc", outputCol="words", pattern="\\W")


# In[8]:


# Count Tokens For cheking word wise
countTokens = udf(lambda words: len(words), IntegerType())


# In[12]:


# Convert Spark Data frame from Panda

loanslice = loan[:50]


# In[42]:


loansfinal = loanslice['desc']
df1 =loanslice[['desc']]


# In[43]:


from pyspark.sql.types import *
p_schema = StructType([StructField('desc',StringType(),True)])


# In[44]:


df_person = sqlContext.createDataFrame(df1, p_schema)


# <h1>Tokenizer on a purticular column</h1>

# In[46]:


tokenized = tokenizer.transform(df_person)
tokenized.select("desc", "words")    .withColumn("tokens", countTokens(col("words"))).show(truncate=False)


# <h3>Stop Word Removal</h3>

# In[55]:


#Load Stop Word Remover 
from pyspark.ml.feature import StopWordsRemover

remover = StopWordsRemover(inputCol="words", outputCol="filtered")
remover.transform(tokenized).show(truncate=False)


# <h1>Binary Tokenization Example</h1>

# In[56]:



from pyspark.ml.feature import Binarizer

continuousDataFrame = spark.createDataFrame([
    (0, 0.1),
    (1, 0.8),
    (2, 0.2)
], ["id", "feature"])

binarizer = Binarizer(threshold=0.5, inputCol="feature", outputCol="binarized_feature")

binarizedDataFrame = binarizer.transform(continuousDataFrame)

print("Binarizer output with Threshold = %f" % binarizer.getThreshold())
binarizedDataFrame.show()

