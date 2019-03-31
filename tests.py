from zweeBasic import * 
# ------------------------------------------------------
# Test conditions goes here 
# ------------------------------------------------------

'''
Test for ML model
'''
use_prev(5)
print(TestLang.ACC)
print(TestLang.prev[0].ACC)
#assert TestLang.ACC < 1
#print("PREV LOSS :")
#print(TestLang.prev.LOSS)
#assert TestLang.LOSS < 0
assert TestLang.ACC - TestLang.prev[0].ACC >= 0
#assert abs(TestLang.NO_OF_CORRECT_PREDICTIONS - TestLang.prev[0].NO_OF_CORRECT_PREDICTIONS) < 0.01*TestLang.NO_OF_CORRECT_PREDICTIONS

