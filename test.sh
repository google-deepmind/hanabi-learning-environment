pylint $(git ls-files '*.py' ':!:hanabi_learning_environment/*' ':!:examples/*')
python3 -m unittest
