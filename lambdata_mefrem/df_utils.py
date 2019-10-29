'''
A suite of utilities to help working with DataFrames
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
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
    "There are " + str(mis_val_table_ren_columns.shape[0]) +
    " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns

class DF_Processor:
    """
    Working with pandas DataFrames before analysis.
    """

    def __init__(self, df):
        self.df = df

    def train_val_test_split(self, **options):
        """
        Split arrays or matrices into random train, validation, and test
        subsets

        Parameters
        ----------
        df : sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays, scipy-sparse
            matrices or pandas dataframes.
        train_size : float
            Should be between 0.0 and 1.0 and represent the proportion of the
            dataset to include in the train split. Defaults to 0.6.
        test_size : float
            Should be between 0.0 and 1.0 and represent the proportion of the
            dataset to include in the test split. Defaults to 0.2.
        val_size : float
            Should be between 0.0 and 1.0 and represent the proportion of the
            dataset to include in the test split. Defaults to 0.2.
        Returns train, val, test and a statement indicating each shape.
        """
        test_size = options.pop('test_size')
        if test_size == None:
            test_size = .2
        val_size = options.pop('val_size')
        if val_size == None:
            val_size = .2
        train_size = options.pop('train_size')
        if train_size == None:
            train_size = .6
        if test_size + val_size + train_size != 1:
            raise ValueError("Size floats must be positive and sum to 1")

        train, val, test = np.split(self.df.sample(frac=1),
                                    [int(train_size*len(self.df)),
                                         int(train_size*len(self.df) +
                                         int(val_size*len(self.df)))])
        print('Shape of train, val, and test dataframes:')
        print(train.shape,val.shape,test.shape)
        return train, val, test

    def add_column(self, df, list_to_add):
        """
        Takes a list, converts to a pandas.Series(), and adds as a column
        to DataFrame, with name of list as name of column.

        Parameters
        ----------
        df : an existing DataFrame
        list_to_add : a list of length equal to number of rows of DataFrame.
        """
        if df.shape[0] != len(list_to_add):
            raise ValueError("Length of list must equal number of rows of df")

        new_column = pandas.Series(data=list_to_add)
        df[new_column] = new_column
        return df
