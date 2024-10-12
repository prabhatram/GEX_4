import unittest
from unittest.mock import patch
import pandas as pd
import random
import io
from scipy.stats import linregress, ttest_ind, mannwhitneyu, chi2_contingency
from gex4 import DataAnalysis  

class TestDataAnalysis(unittest.TestCase):

    def setUp(self):
        
        self.analysis = DataAnalysis()
        self.analysis.df = pd.read_csv('test.csv')  
        self.analysis.column_types = self.analysis.list_column_types()

    def random_column(self, data_type, max_categories=None):
        
        available_columns = [col for col, dtype in self.analysis.column_types.items()
                             if dtype == data_type and (max_categories is None or self.analysis.df[col].nunique() <= max_categories)]
        if available_columns:
            return random.choice(available_columns)
        return None

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)  
    def test_perform_regression(self, mock_stdout, mock_input):
        
        
        x_var = self.random_column('interval')
        y_var = self.random_column('interval')

        
        if x_var and y_var:
            
            mock_input.side_effect = [x_var, y_var]

            X = self.analysis.df[x_var].dropna()
            Y = self.analysis.df[y_var].dropna()
            
            min_length = min(len(X), len(Y))
            X = X[:min_length]
            Y = Y[:min_length]
            
            slope, intercept, r_value, p_value, std_err = linregress(X, Y)
            
            self.analysis.perform_regression(x_var, y_var)
            
            output = mock_stdout.getvalue()
            
            
            self.assertIn(f"Slope: {slope:.4f}", output)
            self.assertIn(f"Intercept: {intercept:.4f}", output)
            self.assertIn(f"R-squared: {r_value**2:.4f}", output)
            self.assertIn(f"P-value: {p_value:.15f}", output)
            self.assertIn(f"Standard error: {std_err:.4f}", output)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)  
    def test_t_test_or_mannwhitney(self, mock_stdout, mock_input):

        continuous_var = self.random_column('interval')
        categorical_var = self.random_column('nominal', max_categories=2)

        if continuous_var and categorical_var:
            
            mock_input.side_effect = [continuous_var, categorical_var]

            
            groups = [group[continuous_var].dropna() for name, group in self.analysis.df.groupby(categorical_var)]

            
            normality_test = self.analysis.check_normality(self.analysis.df[continuous_var])

            if normality_test[0] == 'Shapiro-Wilk':
                
                stat, p_value = ttest_ind(*groups)
            else:
                
                stat, p_value = mannwhitneyu(*groups)

            
            self.analysis.t_test_or_mannwhitney(continuous_var, categorical_var)

            
            output = mock_stdout.getvalue()

            
            self.assertIn(f"Statistic = {stat:.4f}", output)
            self.assertIn(f"p-value = {p_value:.15f}", output)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)  
    def test_chi_square_test(self, mock_stdout, mock_input):
        """Test for chi_square_test function."""
        
        categorical_var_1 = self.random_column('nominal')
        categorical_var_2 = self.random_column('nominal')

        if categorical_var_1 and categorical_var_2:
            mock_input.side_effect = [categorical_var_1, categorical_var_2]

            contingency_table = pd.crosstab(self.analysis.df[categorical_var_1], self.analysis.df[categorical_var_2])

            chi2, p, dof, expected = chi2_contingency(contingency_table)

            self.analysis.chi_square_test(categorical_var_1, categorical_var_2)

            output = mock_stdout.getvalue()

            self.assertIn(f"chi2 = {chi2:.4f}", output)
            self.assertIn(f"p-value = {p:.15f}", output)

if __name__ == '__main__':
    unittest.main()
