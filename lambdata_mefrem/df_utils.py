'''
utility functions for working with DataFrames
'''

import pandas
import numpy as np

# Function to calculate missing values by column

def missing_values_table(df):
        # total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(2)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


# Function to split data, built directly on top of sklearn's library
# Function to split data, built directly on top of sklearn's library
def train_val_test_split(df, **options):
    """Split arrays or matrices into random train, validation, and test
    subsets
    
    Parameters
    ----------
    df : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.
    test_size : float
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.2.
    val_size : float
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.2.
    train_size : float
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test + val
        size.
    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.
        .. versionadded:: 0.16
            If the input is sparse, the output will be a
            ``scipy.sparse.csr_matrix``. Else, output type is the same as the
            input type.
    """
    test_size = options.pop('test_size')
    val_size = options.pop('val_size')
    train_size = options.pop('train_size')
    if test_size + val_size + train_size != 1:
        raise ValueError("Test, val, and train size must be positive and sum to 1")
    
    train, val, test = np.split(df.sample(frac=1),
                                [int(train_size*len(df)),
                                     int(train_size*len(df) + int(val_size*len(df)))])
    print('Shape of train, val, and test dataframes:')
    print(train.shape,val.shape,test.shape)