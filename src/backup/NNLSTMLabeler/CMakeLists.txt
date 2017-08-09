
include_directories(
  model
)

add_executable(NNLSTMLabeler NNCNNLabeler.cpp)
target_link_libraries(NNLSTMLabeler ${LIBS})
