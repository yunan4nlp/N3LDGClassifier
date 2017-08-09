
include_directories(
  model
)

add_executable(NNLSTMCharLabeler NNCNNLabeler.cpp)
target_link_libraries(NNLSTMCharLabeler ${LIBS})
