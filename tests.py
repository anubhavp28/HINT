from zweeBasic import *
# ------------------------------------------------------
# Test Conditions here 
# ------------------------------------------------------

'''
Test for ML model
'''
#print(TestLang.ACC)
#assert TestLang.ACC < 1
#print("PREV LOSS :")
#print(TestLang.prev.LOSS)
#assert TestLang.LOSS < 0
assert TestLang.ACC - TestLang.prev.ACC > 0
assert abs(TestLang.NO_OF_CORRECT_PREDICTIONS - TestLang.prev.NO_OF_CORRECT_PREDICTIONS) < 0.01*TestLang.NO_OF_CORRECT_PREDICTIONS


'''
Test on Dataset
'''


