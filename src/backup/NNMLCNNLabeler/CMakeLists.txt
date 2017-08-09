
include_directories(
  model
)

add_executable(NNMLCNNLabeler NNCNNLabeler.cpp)
target_link_libraries(NNMLCNNLabeler ${LIBS})
