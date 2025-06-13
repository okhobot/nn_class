cmake_minimum_required(VERSION 3.22)
project(
  NN_Class
  DESCRIPTION "my_nn"
  LANGUAGES C CXX
)


file(
  GLOB_RECURSE SOURCES CONFIGURE_DEPENDS 
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)



add_library(${PROJECT_NAME} STATIC ${SOURCES})
target_compile_definitions(${PROJECT_NAME} PRIVATE NNC_EXPORTS=1)

target_link_libraries(${PROJECT_NAME} OCLW)
target_include_directories(${PROJECT_NAME} PRIVATE include)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_23)
