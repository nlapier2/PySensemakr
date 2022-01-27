"""
Provides the example data for the package.
"""
import pandas as pd
import os

path=os.path.join(os.path.dirname(__file__), 'data/darfur.csv')

def load_darfur():
    """
    Provide the example data of darfur based on a survey among Darfurian refugees in eastern Chad.

    The data is on attitudes of Darfurian refugees in eastern Chad. The main "treatment" variable is *directlyharmed*,
    which indicates that the individual was physically injured during attacks on villages in Darfur,
    largely between 2003 and 2004. The main outcome of interest is *peacefactor*, a measure of pro-peace attitudes.

    Key covariates include *herder_dar* (whether they were a herder in Darfur), *farmer_dar* (whether they were a farmer in Darfur),
    *age*, *female* (indicator for female), and *past_voted* (whether they report having voted in an earlier election,
    prior to the conflict).

    Format
    -------
    A data frame with 1276 rows and 14 columns.

    **wouldvote**

    If elections were held in Darfur in the future, would you vote? (0/1)

    **peacefactor**

    A measure of pro-peace attitudes, from a factor analysis of several questions. Rescaled such that 0 is minimally pro-peace and 1 is maximally pro-peace.

    **peace_formerenemies**

    Would you be willing to make peace with your former enemies? (0/1)

    **peace_jjindiv**

    Would you be willing to make peace with Janjweed individuals who carried out violence? (0/1)

    **peace_jjtribes**

    Would you be willing to make peace with the tribes that were part of the Janjaweed? (0/1)

    **gos_soldier_execute**

    Should Government of Sudan soldiers who perpetrated attacks on civilians be executed? (0/1)

    **directlyharmed**

    A binary variable indicating whether the respondent was personally physically injured during attacks on villages in Darfur largely between 2003-2004. 529 respondents report being personally injured, while 747 do not report being injured.

    **age**

    Age of respondent in whole integer years. Ages in the data range from 18 to 100.

    **farmer_dar**

    The respondent was a farmer in Darfur (0/1). 1,051 respondents were farmers, 225 were not.

    **herder_dar**

    The respondent was a herder in Darfur (0/1). 190 respondents were farmers, 1,086 were not.

    **pastvoted**

    The respondent reported having voted in a previous election before the conflict (0/1). 821 respondents reported having voted in a previous election, 455 reported not having voted in a previous election.

    **hhsize_darfur**

    Household size while in Darfur.

    **village**

    Factor variable indicating village of respondent. 486 unique villages are accounted for in the data.

    **female**

    The respondent identifies as female (0/1). 582 respondents are female-identified, 694 are not.

    Reference
    ------------
    Cinelli, C. and Hazlett, C. (2020), "Making Sense of Sensitivity: Extending Omitted Variable Bias."
    Journal of the Royal Statistical Society, Series B (Statistical Methodology).

    Hazlett, Chad. (2019) "Angry or Weary? How Violence Impacts Attitudes toward Peace among Darfurian Refugees."
    Journal of Conflict Resolution: 0022002719879217.

    Return
    -------
    dafaframe
        a Pandas dataframe containing the Darfur violence data.

    Example
    ------------
    >>> from sensemakr import data
    >>> darfur = data.load_darfur()
    """
    return pd.read_csv(path)
