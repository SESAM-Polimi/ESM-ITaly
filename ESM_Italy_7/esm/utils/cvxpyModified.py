# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 18:07:26 2021

@author: Amin
"""
#%%

from tempfile import TemporaryFile
import cvxpy as cp
import pandas as pd
import numpy as np

def cDataFrame(obj
              )-> pd.DataFrame:

    '''
    Description
    =============
    returns a pd.DataFrame of a cvxpy object if is indexed

    '''
    try:
        return pd.DataFrame(data    = obj.value,
                            index   = obj.index,
                            columns = obj.columns
                            )
    except AttributeError:
        raise AssertionError('this function can be used only for indexed cvxpy objects')

class cloccer:

    '''
    Description
    =============
    cloc class is an index-based slicer to be added to the cvxpy methods such
    as Parameter, Variable, Expressions, MultiExpressions and ...

    __getitem__ method works based on a fake pd.DataFrame for index and columns
    that maps the labels to slicers
    '''

    def __init__(self,
                 instance):


        '''
        Parameters
        =============
        instance: any cvxpy object to be indexed
        '''
        self.instance = instance


        # to map the labels to numbers for index
        self.index_id = pd.DataFrame(data    = np.arange(len(self.instance.index)),
                                     index   = self.instance.index,
                                     columns = ['index']
                                     )

        # to map the labels to numbers for columns
        self.columns_id = pd.DataFrame(data    = np.arange(len(self.instance.columns)).T,
                                       index   = self.instance.columns,
                                       columns = ['column'],
                                       ).T

        # a fake dataframe to check the slicer
        self.dataframe = pd.DataFrame(data    = np.zeros((len(self.instance.index),len(self.instance.columns))),
                                      index   = self.instance.index,
                                      columns = self.instance.columns,
                                      )


    def __getitem__(self,key):
        '''
        Description
        =============
        Works exactly as pandas.DataFrame.loc
        '''
        # check if the given key is correct
        check = self.dataframe.loc[key]
        new_index = check.index
        if isinstance(check,pd.DataFrame):
            new_columns = check.columns
        else:
            new_columns = ['0']

        # if it is not a multi-level index, for sure key[0] is index and key[1] is columns
        if isinstance(key,tuple):
            if len(key) == 1:
                index   = self.index_id.loc[key[0],'index']
                columns = slice(None)

            else:

                if isinstance(key[0],slice):
                    index   = key[0]
                    columns = self.columns_id.loc['column',key[1]]
                elif all(isinstance(item,str) for item in key):
                    try:
                        index   = self.index_id.loc[key[0],'index']
                        columns = self.columns_id.loc['column',key[1]]
                    except KeyError:
                        index   = self.index_id.loc[key,'index']
                        columns = slice(None)

                elif any(isinstance(item,list) for item in key):
                    index   = self.index_id.loc[key[0],'index']
                    columns = self.columns_id.loc['column',key[1]]

                else:

                    index   = self.index_id.loc[key[0],'index']
                    columns = self.columns_id.loc['column',key[1]]

        elif isinstance(key,(str,list)):
            index   = self.index_id.loc[key,'index']
            columns = slice(None)

        # else:
        #     print(key,type(key))


        try:
            if len(index) == self.index_id.shape[0]:
                #index = slice(None)
                index = [i for i in index.values]

        except TypeError:
            pass

        try:
            if len(columns) == self.columns_id.shape[1]:
                columns = slice(None)
                #columns = [i for i in columns.values]

        except TypeError:
            pass

        if isinstance(index,(pd.DataFrame,pd.Series)):
            index = [i for i in index.values]
        if isinstance(columns,(pd.DataFrame,pd.Series)):
            columns = [i for i in columns.values]

        try:
            result = self.instance[index,columns]
        except IndexError:
            result = self.instance[index]

        if len(result.shape) == 1:
            result = cp.reshape(result,(result.shape[0],1))

        result.index = new_index
        result.columns = new_columns
        result.cloc = cloccer(result)

        return result



class Variable(cp.Variable):

    def __init__(self,
                 shape  : tuple,
                 nonneg : bool,
                 index  : [pd.Index,pd.MultiIndex,list],
                 columns: [pd.Index,pd.MultiIndex,list]):

        super().__init__(shape = shape,
                         nonneg = nonneg)

        df = pd.DataFrame(np.zeros(shape),
                          index = index,
                          columns = columns)

        self.index   = df.index
        self.columns = df.columns

        self.cloc = cloccer(self)


class Parameter(cp.Parameter):

    def __init__(self,
                 shape  : tuple,
                 index  : [pd.Index,pd.MultiIndex,list],
                 columns: [pd.Index,pd.MultiIndex,list],
                 initialize : int = None,
                 value = None
                 ):

        super().__init__(shape=shape)

        df = pd.DataFrame(np.zeros(shape),
                          index = index,
                          columns = columns)

        if initialize is not None:
            self.value = np.ones(self.shape) * initialize

        self.value   = value
        self.index   = df.index
        self.columns = df.columns

        self.cloc = cloccer(self)


def is_vector(mat):

    if 1 not in mat.shape:
        return False

    return True




def matmul(a,b):
    '''
    matrix multiplication

    a,b should be indexed cvxpyModified objects
    MODIFICATIONS: isinstance(...)
    '''

    # print("MATMUL")
    # print(a.index)
    # print(a.columns)
    # print(b.index)
    # print(b.columns)
    # print("=*=*=*=*=*=*=*=*=*=*=*=*")


    try:
        result = a.cloc[a.index,a.columns] @ b.cloc[a.columns,:]
    except:
        result = a @ b

    result.index     = a.index
    result.columns   = b.columns

    result.cloc = cloccer(result)

    return result


def multiply(a,b):
    '''
    scalar or elementwise multiplication

    a should be indexed cvxpyModified objects
    b can be a scalar, a vector or a matrix

    if a is a matrix and b a vector (a has only one dimension equal to a
    dimension of b), b is replicated to have the same shape of a

    MODIFICATIONS: isinstance(...)

    '''

    if isinstance(a,(int,float)) or isinstance(b,(int,float)):
        try:
            index = a.index
            columns = a.columns
        except AttributeError:
            index = b.index
            columns = b.columns

        result = cp.multiply(a,b)

    elif a.shape != b.shape:
        if is_vector(a):
            # Horizontal vector
            index = b.index
            columns = b.columns
            if a.shape[0] == 1:
                result = cp.multiply(a.cloc[:,b.columns],b)
            else:
                result = cp.multiply(a.cloc[b.index,:],b)
        else:
            index = a.index
            columns = a.columns
            if b.shape[0] == 1:
                result = cp.multiply(a,b.cloc[:,a.columns])
            else:
                result = cp.multiply(a,b.cloc[a.index,:])

    else:

        result = cp.multiply(a,b)
        index = a.index
        columns = a.columns

    result.index     = index
    result.columns   = columns

    result.cloc = cloccer(result)



    return result


def summer(*args):
    '''
    sum function
    there should be at least of indexed cvxpyModified object in the inputs
    '''
    result = sum(args)

    index = args[0].index
    columns = args[0].columns

    for arg in args:
        if not index.equals(arg.index):
            raise ValueError()

        if not columns.equals(arg.columns):
            raise ValueError()

    for index,arg in enumerate(args):

        if hasattr(arg,'index'):
            result.index    = args[index].index
            result.columns  = args[index].columns
            result.cloc     = cloccer(result)
            return result
    raise AssertionError('non of arguments are indexed. use simple sum.')


def rcsum(a,
          dim : int
          ):

    '''
    column or row sum
    a is a cvxpyModified object
    a can be either a matrix or a vector

    '''

    if dim == 0:
        result = cp.sum(a,dim,keepdims=True)
        result.index = ['0']
        result.columns = a.columns
        result.cloc = cloccer(result)
        return result

    elif dim == 1:
        result = cp.sum(a,dim,keepdims=True)
        result.index = a.index
        result.columns = ['0']
        result.cloc = cloccer(result)
        return result

    else:
        raise AssertionError('dim can be: 0 -> rows sum ; 1 -> columns sum')


def trsp(a):
    '''
    vector or matrix transposition

    a should be indexed cvxpyModified objects
    a should be a vector or a matrix

    '''
    result = cp.transpose(a)

    result.index     = a.columns
    result.columns   = a.index

    result.cloc = cloccer(result)

    return result


def diag(a):
    '''
    it diagonalizes a vector or extracts the main diagonal from a square matrix

    a should be indexed cvxpyModified objects
    a should be a vector or a matrix

    '''
    if a.shape[0] > 1 and a.shape[1] > 1 and a.shape[0] != a.shape[1]:
        raise ValueError('the function only accepts square matrices or vectors')

    else:

        if a.shape[0] == a.shape[1]:
            result = cp.diag(a)
            result.index = a.index
            result.columns = a.columns

        elif a.shape[0] == 1:
            result = cp.diag(a)
            result.index = a.columns
            result.columns = a.columns

        elif a.shape[1] == 1:
            result = cp.diag(a)
            result.index = a.index
            result.columns = a.index

    result.cloc = cloccer(result)

    return result


def stack(obj_list : list,
          mode : str,
          ):

    '''
    it stacks (vertically or horizontally) a list of cvxpyModified objects,
    preserving index and columns labels

    '''

    if mode == 'v':
        obj_stacked = 10
    elif mode == 'h':
        obj_stacked = 10
    else:
        raise ValueError('allowed mode options: vertically (mode = v) or horizontally (mode = h)')

    return obj_stacked



#%%
'''
A proof of concept example

A dummy optimization prblem

'''
if __name__ == '__main__':
    import numpy as np
    A_index = [['IT','IT','DU','DU'],['t.1','t.2','t.1','t.2']]
    X_index = [['DU','DU','IT','IT'],['t.1','t.2','t.1','t.2']]

    A = Parameter(shape = (4,4), index = A_index, columns = A_index)

    X = Variable(shape = (4,1), index = X_index, columns = ['production'], nonneg = True)

    A.value = [
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12],
        [13,14,15,16],
    ]
    #AX = matmul(A,X)
    x_val = np.arange(start=1,stop=5)
    x_val.shape = X.shape
    constr = [
        X.cloc[A.index,:]>=x_val
        #X.cloc[("IT","t.1"),:]>=3
    ]

    problem = cp.Problem(cp.Minimize(cp.sum(X)),constr)
    problem.solve()


#%%

#     E_tld_diag = Parameter(shape = (4,1), index = A_index ,columns = ['demand_diag'])

#     # building the optimization problem
#     objective = cp.Minimize(cp.sum(X))

#     # Definiing some constriants
#     constraints = []

#     # constraints without cloc
#     constraints.append(X >= A @ X + E_tld_diag )

#     # A constraint with cloc function
#     # putting a high value on IT, t.2
#     constraints.append(X.cloc[('IT','t.2'),:] >= 500)

#     # putting a high value on whole DU X
#     constraints.append(X.cloc['DU'] >= 3500)


#     # solving the problem
#     problem = cp.Problem(objective,constraints)

#     # Assigning the values to A and Y
#     A.value = 0.1*np.ones((4,4))

#     E_tld.value = 5*np.ones((4,4))
#     EE = pd.DataFrame(E_tld.value,index=E_tld.index,columns=E_tld.columns)
#     EE_2 = pd.DataFrame(0,index=E_tld.index,columns=['0'])
#     EE_2.iloc[0,0] = EE.iloc[0,0]

#     E_tld_diag.value = EE_2.values
#     problem.solve(solver=cp.GUROBI,verbose=True)

#     if problem.status == 'optimal':
#         # printing out the data frames
#         print('X = ', cDataFrame(X))
#         print('E_tld_diag = ', cDataFrame(E_tld_diag))

#     #%%


# #%%
#     import pandas as pd


#     A_index = pd.MultiIndex.from_product([['a','b'],['1','2']])
#     B_index = pd.MultiIndex.from_product([['a','b'],['1','2'],[2015]])













# %%
