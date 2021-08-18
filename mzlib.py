import numpy as np
from scipy.stats import chi2_contingency
import pandas as pd


# ===================================================================
# Creates a set so the number of Y=1 equals the number of Y=0.  (I'm expecting too many Y=1.)
# TO DO: print error if Y is not in column set
def get_balanced_set(df_):
    # Valid 'Make' value, NOT in Ntop
    df1_ = df_.loc[ ~df_['Make'].isnull() & (df_['Y']=='0')]

    # Valid 'Make' value, in Ntop.  Down-sampled to have same number of rows as df1
    nrows = df1_.shape[0]
    df2_ = df_.loc[ ~df_['Make'].isnull() & (df_['Y']=='1')].sample(nrows)

    # Concatenate df1 and df2
    df_model_ = pd.concat([df1_, df2_])
    return(df_model_)


# ===================================================================
def get_chi2_stat(df_, col_name_, targ_name_):
    cross = pd.crosstab( index=df_[col_name_], columns=df_[targ_name_] )
    print(cross)

    chi2_stat, p, dof, ex = chi2_contingency(cross)
    # (chi2=test statistic, p=p-value, dof=dof, ex=expected freq.)
    # Note p<0.05  
    # So, there's some correlation here that isn't due to noise.
    print("- - - - - - - - - - - - - - -")
    print("Chi-squared statistic = ", chi2_stat)
    print("p-value = ", p)
    print("No. dof = ", dof)
    print("expected frequency (null hyp.) = ", ex)
    print("- - - - - - - - - - - - - - -")

    
# ===================================================================
def get_test_frame(agency_, color_, code_, list_Agency_, list_Color_, list_Code_, columnsX):

#    print("BEFORE: agency = ", agency_)
#    print("BEFORE: color = ", color_)
#    print("BEFORE: code = ", code_)

    # Agency ----------------------------------------
    agency_new = str(int(float(agency_)))
    if agency_new not in list_Agency_:
        agency_new = "other"
#    print("AFTER: agency_new = ", agency_new)

    # Color ---------------------------------------
    color_new = str(color_)
    if color_new not in list_Color_:
        color_new = "other"
#    print("AFTER: color_new = ", color_new)

    # Violation code -------------------------------
    code_new = str(code_)
    if code_new not in list_Code_:
        code_new = "other"
#    print("AFTER: code_new = ", code_new)
    
    # First create empty dictionary, to be filled in ---------------
    d_empty = {}
    for k in columnsX:
        d_empty[k] = 0

    # Fill in dictionary ------------------
    d_live = d_empty.copy()
    d_live['Agency_new_' + agency_new] = 1
    d_live['Color_new_' + color_new] = 1
    d_live['Code_new_' + code_new] = 1

    df_live = pd.DataFrame(d_live, index=["values"])  #pandas requires an index in this situation
    
    return(df_live)    
    
    
# ===================================================================
def print_prediction( y, Ntop ):
    print("\n")
    if y == '0':
        print("Prediction: NOT in top-{} Makes".format(Ntop))
    elif y == '1':
        print("Prediction: IN top-{} Makes".format(Ntop))
    else:
        print("ERROR in program.  See Mike")
    print("\n")

    
    
