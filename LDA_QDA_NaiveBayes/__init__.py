'''To run the entire models we can use this __init__.py'''

from lda_main import main

lda_model = main()

from qda_main import main
print("\n")
qda_Model = main()

from nb_main import main
print("\n")
nb_model = main()
print("\n")
