
include_directories(
  model
)

add_executable(NNLSTMpLabeler NNCNNLabeler.cpp)
target_link_libraries(NNLSTMpLabeler ${LIBS})
