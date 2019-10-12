add_library (pyhanabi SHARED pyhanabi.cc)
target_link_libraries (pyhanabi LINK_PUBLIC hanabi)

install(TARGETS pyhanabi LIBRARY DESTINATION hanabi_learning_environment)
install(FILES __init__.py DESTINATION hanabi_learning_environment)
install(FILES rl_env.py DESTINATION hanabi_learning_environment)
install(FILES pyhanabi.py DESTINATION hanabi_learning_environment)
install(FILES pyhanabi.h DESTINATION hanabi_learning_environment)
