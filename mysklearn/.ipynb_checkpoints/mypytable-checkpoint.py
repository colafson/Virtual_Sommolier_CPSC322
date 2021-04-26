import mysklearn.myutils as myutils
import copy
import csv 
#from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data. There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    # def pretty_print(self):
    #     """Prints the table in a nicely formatted grid structure.
    #     """
    #     print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        height = len(self.column_names)
        width = len(self.data)
        return (width, height)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        column = []
        index = self.column_names.index(col_identifier)
        for row in self.data:
            value = row[index]
            if value == "NA":
                if include_missing_values:
                    column.append(value)
            else:
                column.append(value)
        return column 

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row in range(len(self.data)):
            for col in range(len(self.data[row])):
                try: 
                    self.data[row][col] = float(self.data[row][col])
                except ValueError:
                    pass

    def drop_rows(self, rows_to_drop):
        """Remove rows from the table data.

        Args:
            rows_to_drop(list of list of obj): list of rows to remove from the table data.
        """
        temp = []
        for row in range(len(self.data)):
            if not self.data[row] in rows_to_drop:
                temp.append(self.data[row])
        self.data = temp

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like: table = MyPyTable().load_from_file(fname)
        
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        in_file = open(filename, "r")
        reader = csv.reader(in_file, dialect = 'excel')
        col_names = []
        table = []
        count = 0
        #loop though the csv and fill the table array
        for x in reader:
            if count == 0:
                col_names = x
            else:
                table.append(x)
            count +=1
        in_file.close
        self.column_names = col_names
        self.data = table
        self.convert_to_numeric()
        return self 

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(self.column_names)
            writer.writerows(self.data)

    def find_duplicates(self, key_column_names):
        """Returns a list of duplicates. Rows are identified uniquely baed on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns: 
            list of list of obj: list of duplicate rows found
        
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s). The first instance is not
                considered a duplicate.
        """
        indexes = []
        for x in key_column_names:
            indexes.append(self.column_names.index(x))
        duplicate = []
        occurence = []
        for row in self.data:
            keys = []
            for index in indexes:
                keys.append(row[index])
            if keys in occurence:
                if not row in duplicate:
                    duplicate.append(row)
            else:
                occurence.append(keys)
        return duplicate

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        clean_data = []
        for row in self.data:
            hasMissing = False
            for col in row:
                if col == "NA" or col == "":
                    hasMissing = True
            if not hasMissing:
                clean_data.append(row)
        self.data = clean_data

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        cols = self.get_column(col_name, False)
        average = sum(cols)/len(cols)
        index = self.column_names.index(col_name)
        clean_data = []
        for row in self.data:
            hasMissing = False
            for col in row:
                if col == "NA":
                    hasMissing = True
            if hasMissing:
                row[index] = average
            clean_data.append(row)
        self.data = clean_data

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order is as follows: 
            ["attribute", "min", "max", "mid", "avg", "median"]
        """
        name = ["attribute", "min", "max", "mid", "avg", "median"]
        table = []
        for col in col_names:
            row = []
            values = self.get_column(col,False)
            if not values == []:
                values.sort()
                row.append(col)
                row.append (min(values))
                row.append (max(values))
                mid = (max(values)+min(values))/2
                row.append (mid)
                avg = (sum(values)/len(values))
                row.append (avg)
                if len(values)%2 == 0:
                    median = (values[int((len(values)-1)/2)]+values[int(((len(values)-1)/2)+1)])/2
                else:
                    median = values[int((len(values)-1)/2)]
                row.append(median)
                table.append(row)
        return MyPyTable(name,table)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        table = []
        table2 = []
        joined_table = []
        mainIndexes = []
        otherIndexes = []

        for x in key_column_names:
            mainIndexes.append(self.column_names.index(x))
            otherIndexes.append(other_table.column_names.index(x))

        for row in self.data:
            key = []
            for index in mainIndexes:
                key.append(row[index])
            table.append(key)

        for row in other_table.data:
            key = []
            for index in otherIndexes:
                key.append(row[index])
            table2.append(key)

        header = self.column_names
        otherHeader = other_table.column_names
        for index in range(len(otherHeader)):
            if not index in otherIndexes:
                header.append(otherHeader[index])

        for row in range(len(table)):
            if table[row] in table2:
                info = self.data[row]
                ind = table2.index(table[row])
                for index in range(len(other_table.data[ind])):
                    if not index in otherIndexes:
                        info.append(other_table.data[ind][index])
                joined_table.append(info)     
        return MyPyTable(header, joined_table)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        table = []
        table2 = []
        joined_table = []
        mainIndexes = []
        otherIndexes = []

        for x in key_column_names:
            mainIndexes.append(self.column_names.index(x))
            otherIndexes.append(other_table.column_names.index(x))

        for row in self.data:
            key = []
            for index in mainIndexes:
                key.append(row[index])
            table.append(key)

        for row in other_table.data:
            key = []
            for index in otherIndexes:
                key.append(row[index])
            table2.append(key)

        header = self.column_names
        otherHeader = other_table.column_names
        for index in range(len(otherHeader)):
            if not index in otherIndexes:
                header.append(otherHeader[index])

        remainingKeysMain = []
        for row in range(len(table)):
            if table[row] in table2:
                info = self.data[row]
                ind = table2.index(table[row])
                for index in range(len(other_table.data[ind])):
                    if not index in otherIndexes:
                        info.append(other_table.data[ind][index])
                #del other_table.data[ind]
                #del table2[ind]
                joined_table.append(info)
            else:
                remainingKeysMain.append(self.data[row])

        mainEnd = []
        for x in range(len(otherHeader)):
            if not x in otherIndexes:
                mainEnd.append("NA")
        for row in remainingKeysMain:
            value = row
            value += mainEnd
            joined_table.append(value)


        for row in range(len(table2)):
            if not table2[row] in table:
                values = copy.copy(table2[row])
                end = []
                for x in range(len(self.column_names)-1):
                    if not x in mainIndexes:
                        end.append("NA")
                    else:
                        end.append(values[0])
                        del values[0]      
                for index in range(len(other_table.data[row])):
                    if not index in otherIndexes:
                        end.append(other_table.data[row][index])
                joined_table.append(end)

        return MyPyTable(header, joined_table)